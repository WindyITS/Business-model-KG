import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from runtime.query_planner import QueryPlanEnvelope, QueryPlanPayload, compile_query_plan
from training.query_planner import (
    build_dataset_manifest,
    build_dataset_splits,
    build_synthetic_company_graphs,
    write_dataset_splits,
)
from training.query_planner import dataset as dataset_module
from training.query_planner.graphs import evaluate_query_plan, matching_graph_ids_for_plan


class QueryPlannerDatasetTests(unittest.TestCase):
    def test_build_synthetic_company_graphs_returns_five_deep_graphs(self):
        companies = build_synthetic_company_graphs()

        self.assertEqual(len(companies), 5)
        self.assertTrue(all(len(company.segments) == 4 for company in companies))

        all_root_offerings = {
            offering.name
            for company in companies
            for segment in company.segments
            for offering in segment.offerings
        }
        self.assertIn("Cloud Platform", all_root_offerings)
        self.assertIn("Marketplace Hub", all_root_offerings)
        self.assertIn("Analytics Studio", all_root_offerings)

    def test_build_dataset_splits_is_deterministic_and_split_scoped(self):
        first = build_dataset_splits(train_size=36, validation_size=18, release_eval_size=24, seed=3)
        second = build_dataset_splits(train_size=36, validation_size=18, release_eval_size=24, seed=3)

        self.assertEqual(first, second)

        allowed_graphs = {
            "train": {"aurora", "redwood", "lattice"},
            "validation": {"nimbus"},
            "release_eval": {"vector"},
        }
        for split_name, examples in first.items():
            self.assertEqual(len(examples), {"train": 36, "validation": 18, "release_eval": 24}[split_name])
            for example in examples:
                self.assertIn(example.route_label, {"local_safe", "strong_model_candidate", "refuse"})
                self.assertTrue(example.case_id)
                self.assertTrue(example.template_id)
                self.assertTrue(example.variant_id)
                self.assertEqual(example.supervision_target["route_label"], example.route_label)
                self.assertEqual(example.supervision_target["plan"], example.target)
                self.assertTrue(set(example.metadata["source_graph_ids"]).issubset(allowed_graphs[split_name]))

        local_safe = next(example for example in first["train"] if example.route_label == "local_safe")
        self.assertTrue(local_safe.target["answerable"])
        self.assertIsNotNone(local_safe.gold_cypher)
        self.assertTrue(local_safe.gold_rows)

        non_local = next(example for example in first["train"] if example.route_label != "local_safe")
        self.assertFalse(non_local.target["answerable"])
        self.assertIsNone(non_local.gold_cypher)
        self.assertEqual(non_local.gold_rows, [])

        self.assertFalse(set(example.case_id for example in first["train"]).intersection(example.case_id for example in first["validation"]))
        self.assertFalse(set(example.case_id for example in first["train"]).intersection(example.case_id for example in first["release_eval"]))
        self.assertFalse(set(example.case_id for example in first["validation"]).intersection(example.case_id for example in first["release_eval"]))

    def test_descendant_revenue_multi_model_matches_runtime_semantics(self):
        companies = build_synthetic_company_graphs()
        company_tuple = tuple(companies)

        positive_plan = QueryPlanEnvelope(
            answerable=True,
            family="companies_by_descendant_revenue",
            payload=QueryPlanPayload(
                companies=["Aurora Systems"],
                offerings=["Cloud Platform"],
                revenue_models=["subscription", "consumption-based"],
                hierarchy_mode="descendant",
            ),
        )
        positive_compiled = compile_query_plan(positive_plan)
        self.assertTrue(positive_compiled.answerable)
        self.assertEqual(evaluate_query_plan(company_tuple, positive_plan), [{"company": "Aurora Systems"}])
        self.assertEqual(matching_graph_ids_for_plan(company_tuple, positive_plan), ("aurora",))

        negative_plan = QueryPlanEnvelope(
            answerable=True,
            family="companies_by_descendant_revenue",
            payload=QueryPlanPayload(
                companies=["Aurora Systems"],
                offerings=["Analytics Studio"],
                revenue_models=["subscription", "licensing"],
                hierarchy_mode="descendant",
            ),
        )
        negative_compiled = compile_query_plan(negative_plan)
        self.assertTrue(negative_compiled.answerable)
        self.assertEqual(evaluate_query_plan(company_tuple, negative_plan), [])
        self.assertEqual(matching_graph_ids_for_plan(company_tuple, negative_plan), ())

    def test_local_safe_metadata_tracks_actual_matching_graph_ids(self):
        splits = build_dataset_splits(train_size=120, validation_size=30, release_eval_size=45, seed=13)

        multi_graph_example = next(
            example
            for example in splits["train"]
            if example.route_label == "local_safe"
            and "company" in example.gold_rows[0]
            and len(example.metadata["source_graph_ids"]) > 1
        )
        self.assertEqual(
            set(multi_graph_example.metadata["source_graph_ids"]),
            {
                {"Aurora Systems": "aurora", "Redwood Retail": "redwood", "Lattice Finance": "lattice"}[row["company"]]
                for row in multi_graph_example.gold_rows
            },
        )

    def test_surface_generation_avoids_misleading_cross_segment_and_malformed_boolean_phrasing(self):
        companies = build_synthetic_company_graphs()
        company_map = {company.graph_id: company for company in companies}
        train_companies = tuple(company_map[graph_id] for graph_id in dataset_module.GRAPH_IDS_BY_SPLIT["train"])
        local_safe_cases = dataset_module._canonical_local_safe_cases(train_companies)

        disallowed_cross_segment_phrases = {
            "across different segments",
            "across multiple segments",
            "split across segments",
        }
        cross_segment_surfaces = [
            question
            for case in local_safe_cases["companies_by_cross_segment_filters"][:10]
            for _, _, question in dataset_module._surface_pool_for_case(case, "train")
        ]
        self.assertTrue(cross_segment_surfaces)
        for question in cross_segment_surfaces:
            lowered = question.casefold()
            self.assertFalse(any(phrase in lowered for phrase in disallowed_cross_segment_phrases))

        boolean_surfaces = [
            question
            for case in local_safe_cases["boolean_exists"][:10]
            for _, _, question in dataset_module._surface_pool_for_case(case, "train")
        ]
        self.assertTrue(boolean_surfaces)
        for question in boolean_surfaces:
            self.assertNotIn("Is there a match where", question)
            self.assertTrue(question.endswith("?"))
            stem = question.split(", ", 1)[-1].strip().lower()
            self.assertTrue(stem.startswith(("does ", "do ", "is ", "are ", "can ", "would ")))
            self.assertNotIn(" that serve ", question.casefold())
            self.assertNotIn(" that sell through ", question.casefold())
            self.assertNotIn("matched to a matching segment", question.casefold())

        boolean_answers = {
            bool(evaluate_query_plan(train_companies, case.plan)[0]["is_match"])
            for case in local_safe_cases["boolean_exists"]
        }
        self.assertEqual(boolean_answers, {False, True})

    def test_local_safe_surfaces_preserve_company_segment_and_place_scope(self):
        companies = build_synthetic_company_graphs()
        company_map = {company.graph_id: company for company in companies}
        train_companies = tuple(company_map[graph_id] for graph_id in dataset_module.GRAPH_IDS_BY_SPLIT["train"])
        local_safe_cases = dataset_module._canonical_local_safe_cases(train_companies)

        company_scoped_families = {
            "companies_by_segment_filters",
            "segments_by_segment_filters",
            "companies_by_cross_segment_filters",
            "companies_by_descendant_revenue",
            "segments_by_place_and_segment_filters",
            "companies_by_partner",
        }
        for family in company_scoped_families:
            for case in local_safe_cases[family]:
                payload = case.plan.payload
                assert payload is not None
                if payload.companies:
                    company_tokens = [company.casefold() for company in payload.companies]
                    for _, _, question in dataset_module._surface_pool_for_case(case, "train"):
                        lowered = question.casefold()
                        self.assertTrue(
                            all(token in lowered for token in company_tokens),
                            msg=f"{family} omitted company scope for {payload.model_dump(exclude_none=True)} -> {question}",
                        )

        for family in {"companies_by_segment_filters", "segments_by_segment_filters"}:
            for case in local_safe_cases[family]:
                payload = case.plan.payload
                assert payload is not None
                if payload.segments:
                    segment_tokens = [segment.casefold() for segment in payload.segments]
                    for _, _, question in dataset_module._surface_pool_for_case(case, "train"):
                        lowered = question.casefold()
                        self.assertTrue(
                            all(token in lowered for token in segment_tokens),
                            msg=f"{family} omitted segment scope for {payload.model_dump(exclude_none=True)} -> {question}",
                        )

        for case in local_safe_cases["companies_by_descendant_revenue"]:
            payload = case.plan.payload
            assert payload is not None
            if payload.places:
                place_aliases = {
                    alias.casefold()
                    for place in payload.places
                    for alias in dataset_module.PLACE_SURFACES.get(place, (place,))
                }
                for _, _, question in dataset_module._surface_pool_for_case(case, "train"):
                    self.assertTrue(
                        any(alias in question.casefold() for alias in place_aliases),
                        msg=f"companies_by_descendant_revenue omitted place scope for {payload.model_dump(exclude_none=True)} -> {question}",
                    )

        for case in local_safe_cases["segments_by_place_and_segment_filters"]:
            payload = case.plan.payload
            assert payload is not None
            for _, _, question in dataset_module._surface_pool_for_case(case, "train"):
                self.assertNotIn("business segments operating in", question.casefold())
                self.assertNotIn("segments operating in", question.casefold())
                if payload.companies:
                    lowered = question.casefold()
                    company_name = payload.companies[0].casefold()
                    self.assertNotIn(f"at {company_name} at companies", lowered)
                    self.assertNotIn(f"at {company_name} for companies", lowered)

    def test_limit_surfaces_explicitly_mention_result_caps(self):
        companies = build_synthetic_company_graphs()
        company_map = {company.graph_id: company for company in companies}
        for split_name, graph_ids in dataset_module.GRAPH_IDS_BY_SPLIT.items():
            split_companies = tuple(company_map[graph_id] for graph_id in graph_ids)
            local_safe_cases = dataset_module._canonical_local_safe_cases(split_companies)
            for family, cases in local_safe_cases.items():
                for case in cases:
                    payload = case.plan.payload
                    assert payload is not None
                    if payload.limit is None:
                        continue
                    for _, _, question in dataset_module._surface_pool_for_case(case, split_name):
                        self.assertTrue(
                            dataset_module._limit_mentioned(question, payload.limit),
                            msg=f"{family} omitted explicit cap for limit={payload.limit}: {question}",
                        )

    def test_materialized_surfaces_avoid_known_bad_phrasing_patterns(self):
        splits = build_dataset_splits(train_size=240, validation_size=72, release_eval_size=108, seed=19)

        forbidden_substrings = (
            "business segments that serves",
            "business segments that sells",
            "satisfy serve ",
            "satisfy sell through ",
            "whose descendants of ",
            "systems's",
            "company portfolio",
            "business-wide request",
            "within the company scope of",
            "when the company scope is",
            "among companies scoped to",
            "descendant offerings in the ",
        )
        for split_name, examples in splits.items():
            for example in examples:
                lowered = example.question.casefold()
                for substring in forbidden_substrings:
                    self.assertNotIn(
                        substring,
                        lowered,
                        msg=f"{split_name} emitted malformed phrase {substring!r}: {example.question}",
                    )
                if example.family == "offerings_by_company" and len(example.target.get("payload", {}).get("companies", [])) > 1:
                    self.assertNotIn(
                        " does ",
                        f" {lowered} ",
                        msg=f"{split_name} emitted singular company wording for multi-company offerings question: {example.question}",
                    )

    def test_materialized_examples_avoid_single_company_plural_company_prompts_and_bad_count_grammar(self):
        splits = build_dataset_splits(train_size=240, validation_size=72, release_eval_size=108, seed=29)
        company_families = {
            "companies_by_segment_filters",
            "companies_by_cross_segment_filters",
            "companies_by_descendant_revenue",
            "companies_by_partner",
        }
        for split_name, examples in splits.items():
            for example in examples:
                payload = example.target.get("payload", {})
                companies = payload.get("companies", [])
                lowered = example.question.casefold()
                if example.family in company_families and len(companies) == 1:
                    self.assertNotIn("which companies", lowered, msg=f"{split_name}: {example.question}")
                    self.assertNotIn("list the companies", lowered, msg=f"{split_name}: {example.question}")
                    self.assertNotIn("name the companies", lowered, msg=f"{split_name}: {example.question}")
                    self.assertNotIn("what companies", lowered, msg=f"{split_name}: {example.question}")
                    self.assertNotIn("identify the companies", lowered, msg=f"{split_name}: {example.question}")
                    company_name = companies[0].casefold()
                    self.assertNotIn(f"for {company_name},", lowered, msg=f"{split_name}: {example.question}")
                    self.assertNotIn(f", for {company_name},", lowered, msg=f"{split_name}: {example.question}")
                    self.assertNotIn(f"list {company_name} if", lowered, msg=f"{split_name}: {example.question}")
                    self.assertNotIn(f"return {company_name} if", lowered, msg=f"{split_name}: {example.question}")
                    self.assertNotIn(f"show {company_name} if", lowered, msg=f"{split_name}: {example.question}")
                    self.assertNotIn(f"name {company_name} if", lowered, msg=f"{split_name}: {example.question}")
                    self.assertNotIn(f"identify {company_name} if", lowered, msg=f"{split_name}: {example.question}")
                if example.family == "count_aggregate":
                    self.assertNotIn("what is the companies count", lowered, msg=f"{split_name}: {example.question}")
                    self.assertNotIn("what is the offerings count", lowered, msg=f"{split_name}: {example.question}")
                    self.assertNotIn("what is the business segments count", lowered, msg=f"{split_name}: {example.question}")
                    self.assertNotIn("for cases that", lowered, msg=f"{split_name}: {example.question}")
                    self.assertNotIn("match the condition to", lowered, msg=f"{split_name}: {example.question}")

    def test_release_eval_segment_portfolio_surfaces_are_grammatical(self):
        splits = build_dataset_splits(train_size=240, validation_size=72, release_eval_size=108, seed=37)
        bad_substrings = (
            "which segment portfolio belong",
            "segment portfolio associated",
            "segment portfolio that sit",
            "identify up to 6 segment portfolio",
        )
        for example in splits["release_eval"]:
            lowered = example.question.casefold()
            for substring in bad_substrings:
                self.assertNotIn(substring, lowered, msg=example.question)

    def test_train_materialization_includes_multi_company_examples_for_company_selection_families(self):
        splits = build_dataset_splits(train_size=360, validation_size=108, release_eval_size=162, seed=41)
        target_families = {
            "companies_by_segment_filters",
            "companies_by_cross_segment_filters",
            "companies_by_descendant_revenue",
            "companies_by_partner",
        }
        counts = Counter()
        for example in splits["train"]:
            payload = example.target.get("payload", {})
            if example.family in target_families and len(payload.get("companies", [])) > 1:
                counts[example.family] += 1
        for family in target_families:
            self.assertGreater(counts[family], 0, msg=f"train did not materialize multi-company examples for {family}")

    def test_materialized_ranking_examples_avoid_tie_only_single_company_company_count_metrics(self):
        splits = build_dataset_splits(train_size=240, validation_size=72, release_eval_size=108, seed=31)
        tie_only_metrics = {"customer_type_by_company_count", "revenue_model_by_company_count"}
        companies = build_synthetic_company_graphs()
        company_map = {company.graph_id: company for company in companies}
        for split_name, examples in splits.items():
            split_companies = tuple(company_map[graph_id] for graph_id in dataset_module.GRAPH_IDS_BY_SPLIT[split_name])
            for example in examples:
                if example.family != "ranking_topk":
                    continue
                payload = example.target.get("payload", {})
                metric = (payload.get("aggregate_spec") or {}).get("ranking_metric")
                if metric in tie_only_metrics:
                    self.assertFalse(
                        payload.get("companies"),
                        msg=f"{split_name} selected tie-only company-scoped ranking example: {example.question}",
                    )
                plan = QueryPlanEnvelope(
                    answerable=True,
                    family="ranking_topk",
                    payload=QueryPlanPayload.model_validate(payload),
                )
                rows = evaluate_query_plan(split_companies, plan)
                self.assertGreaterEqual(len(rows), 2, msg=f"{split_name}: {example.question} -> {rows}")
                score_key = dataset_module._ranking_score_key(metric)
                self.assertIsNotNone(score_key)
                self.assertGreaterEqual(
                    len({row[score_key] for row in rows if score_key in row}),
                    2,
                    msg=f"{split_name}: {example.question} -> {rows}",
                )

    def test_ranking_surfaces_preserve_company_and_place_scope(self):
        companies = build_synthetic_company_graphs()

        company_case = dataset_module._make_local_safe_case(
            family="ranking_topk",
            bucket="ranking",
            payload=QueryPlanPayload(
                companies=["Aurora Systems"],
                aggregate_spec={"kind": "ranking", "ranking_metric": "customer_type_by_company_count"},
                limit=3,
            ),
            source_graph_ids=["aurora"],
        )
        self.assertIsNotNone(company_case)
        for _, _, question in dataset_module._surface_pool_for_case(company_case, "train"):
            self.assertIn("Aurora Systems", question)

        place_case = dataset_module._make_local_safe_case(
            family="ranking_topk",
            bucket="ranking",
            payload=QueryPlanPayload(
                places=["Europe"],
                aggregate_spec={"kind": "ranking", "ranking_metric": "channel_by_segment_count"},
                limit=3,
            ),
            source_graph_ids=["aurora", "redwood", "lattice"],
        )
        self.assertIsNotNone(place_case)
        europe_aliases = {surface.casefold() for surface in dataset_module.PLACE_SURFACES["Europe"]}
        for _, _, question in dataset_module._surface_pool_for_case(place_case, "train"):
            self.assertTrue(any(alias in question.casefold() for alias in europe_aliases))

        combined_case = dataset_module._make_local_safe_case(
            family="ranking_topk",
            bucket="ranking",
            payload=QueryPlanPayload(
                companies=["Aurora Systems"],
                places=["Europe"],
                aggregate_spec={"kind": "ranking", "ranking_metric": "revenue_model_by_company_count"},
                limit=3,
            ),
            source_graph_ids=["aurora"],
        )
        self.assertIsNotNone(combined_case)
        for _, _, question in dataset_module._surface_pool_for_case(combined_case, "train"):
            lowered = question.casefold()
            self.assertIn("aurora systems", lowered)
            self.assertTrue(any(alias in lowered for alias in europe_aliases))

        matched_segment_place_case = dataset_module._make_local_safe_case(
            family="ranking_topk",
            bucket="ranking",
            payload=QueryPlanPayload(
                customer_types=["developers"],
                places=["Europe"],
                aggregate_spec={"kind": "ranking", "ranking_metric": "company_by_matched_segment_count"},
                limit=3,
            ),
            source_graph_ids=["aurora", "redwood", "lattice"],
        )
        self.assertIsNotNone(matched_segment_place_case)
        for _, _, question in dataset_module._surface_pool_for_case(matched_segment_place_case, "train"):
            lowered = question.casefold()
            self.assertTrue(any(alias in lowered for alias in europe_aliases))
            self.assertNotIn("segments that serve developers and operate in europe", lowered)
            self.assertNotIn("segments that serve developers and operate in the european market", lowered)
            self.assertNotIn("segments that serve developers and operate in european geographies", lowered)

    def test_manifest_tracks_balance_holdouts_and_repetition_stats(self):
        manifest = build_dataset_manifest(train_size=800, validation_size=240, release_eval_size=360, seed=11)
        splits = build_dataset_splits(train_size=800, validation_size=240, release_eval_size=360, seed=11)

        self.assertEqual(manifest["split_sizes"]["train"], 800)
        self.assertEqual(manifest["graph_assignments"]["train"], ["aurora", "redwood", "lattice"])
        self.assertEqual(manifest["graph_assignments"]["validation"], ["nimbus"])
        self.assertEqual(manifest["graph_assignments"]["release_eval"], ["vector"])
        self.assertEqual(sum(manifest["route_targets"]["train"].values()), 800)
        self.assertEqual(
            sum(manifest["local_safe_family_targets"]["validation"].values()),
            manifest["route_targets"]["validation"]["local_safe"],
        )

        train_stats = manifest["split_stats"]["train"]
        self.assertEqual(train_stats["count"], 800)
        self.assertEqual(train_stats["duplicate_question_count"], 0)
        self.assertEqual(train_stats["duplicate_question_target_count"], 0)
        self.assertEqual(train_stats["unique_question_count"], 800)
        self.assertGreaterEqual(train_stats["unique_target_count"], 390)
        self.assertEqual(train_stats["route_counts"], train_stats["supervision_target_route_counts"])
        self.assertEqual(
            train_stats["strong_model_candidate_reason_counts"],
            {"beyond_local_coverage": manifest["route_targets"]["train"]["strong_model_candidate"]},
        )
        self.assertEqual(set(train_stats["boolean_answer_counts"]), {"false", "true"})
        self.assertGreater(train_stats["boolean_answer_counts"]["false"], 0)
        self.assertGreater(train_stats["boolean_answer_counts"]["true"], 0)
        self.assertEqual(
            sum(train_stats["ranking_scope_counts"].values()),
            train_stats["family_counts"].get("ranking_topk", 0),
        )
        self.assertEqual(
            set(train_stats["refusal_reason_counts"]),
            {
                "ambiguous_closed_label",
                "ambiguous_request",
                "beyond_local_coverage",
                "unsupported_metric",
                "unsupported_schema",
                "unsupported_time",
                "write_request",
            },
        )
        self.assertEqual(manifest["split_overlap_stats"]["train__validation"]["question_overlap_count"], 0)
        self.assertEqual(manifest["split_overlap_stats"]["train__validation"]["question_target_overlap_count"], 0)
        self.assertEqual(manifest["split_overlap_stats"]["train__release_eval"]["question_overlap_count"], 0)
        self.assertEqual(manifest["split_overlap_stats"]["train__release_eval"]["question_target_overlap_count"], 0)
        self.assertIn("common_offerings_between_segments", manifest["strong_model_candidate_feasible_families"]["validation"])
        self.assertIn("common_offerings_between_segments", manifest["strong_model_candidate_feasible_families"]["release_eval"])
        self.assertEqual(manifest["strong_model_candidate_missing_families"]["train"], [])
        self.assertEqual(manifest["strong_model_candidate_missing_families"]["validation"], [])
        self.assertEqual(manifest["strong_model_candidate_missing_families"]["release_eval"], [])
        self.assertEqual(manifest["split_overlap_stats"]["train__validation"]["case_id_overlap_count"], 0)
        self.assertEqual(manifest["split_overlap_stats"]["train__release_eval"]["case_id_overlap_count"], 0)
        self.assertEqual(manifest["split_overlap_stats"]["validation__release_eval"]["case_id_overlap_count"], 0)
        self.assertEqual(manifest["split_overlap_stats"]["validation__release_eval"]["question_overlap_count"], 0)
        self.assertEqual(manifest["split_overlap_stats"]["validation__release_eval"]["question_target_overlap_count"], 0)

        for split_name in ("validation", "release_eval"):
            boolean_counts = manifest["split_stats"][split_name]["boolean_answer_counts"]
            self.assertEqual(set(boolean_counts), {"false", "true"})
            self.assertGreater(boolean_counts["false"], 0)
            self.assertGreater(boolean_counts["true"], 0)

        for split_name in ("train", "validation", "release_eval"):
            self.assertGreater(manifest["split_stats"][split_name]["ranking_scope_counts"]["company+place"], 0)
            expected_bucket_counts = Counter(
                example.metadata["bucket"]
                for example in splits[split_name]
                if example.route_label == "local_safe"
            )
            expected_bucket_counts = {
                bucket: expected_bucket_counts.get(bucket, 0)
                for bucket in (
                    "inventory",
                    "same_segment",
                    "cross_segment",
                    "hierarchy",
                    "geography",
                    "partner",
                    "boolean",
                    "count",
                    "ranking",
                )
            }
            self.assertEqual(manifest["local_safe_bucket_targets"][split_name], expected_bucket_counts)
            self.assertEqual(manifest["split_stats"][split_name]["local_safe_bucket_counts"], expected_bucket_counts)

        overlap_pairs = (
            ("train", "validation", "train__validation"),
            ("train", "release_eval", "train__release_eval"),
            ("validation", "release_eval", "validation__release_eval"),
        )
        for left, right, manifest_key in overlap_pairs:
            expected_target_overlap = len(
                {
                    json.dumps(example.supervision_target, sort_keys=True, separators=(",", ":"))
                    for example in splits[left]
                }.intersection(
                    {
                        json.dumps(example.supervision_target, sort_keys=True, separators=(",", ":"))
                        for example in splits[right]
                    }
                )
            )
            self.assertEqual(
                manifest["split_overlap_stats"][manifest_key]["target_overlap_count"],
                expected_target_overlap,
            )

    def test_supervision_target_distinguishes_strong_candidates_from_refusals(self):
        splits = build_dataset_splits(train_size=120, validation_size=36, release_eval_size=54, seed=17)

        strong_candidate = next(example for example in splits["train"] if example.route_label == "strong_model_candidate")
        refusal = next(example for example in splits["train"] if example.route_label == "refuse")

        self.assertFalse(strong_candidate.supervision_target["plan"]["answerable"])
        self.assertFalse(refusal.supervision_target["plan"]["answerable"])
        self.assertEqual(strong_candidate.supervision_target["plan"]["reason"], "beyond_local_coverage")
        self.assertEqual(strong_candidate.supervision_target["route_label"], "strong_model_candidate")
        self.assertEqual(refusal.supervision_target["route_label"], "refuse")
        self.assertNotEqual(strong_candidate.supervision_target, refusal.supervision_target)

    def test_write_dataset_splits_writes_jsonl_and_support_files(self):
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            written = write_dataset_splits(output_dir, train_size=30, validation_size=15, release_eval_size=21, seed=5)

            self.assertEqual(set(written), {"train", "validation", "release_eval", "synthetic_graphs", "manifest"})
            self.assertTrue(written["train"].exists())
            self.assertTrue(written["validation"].exists())
            self.assertTrue(written["release_eval"].exists())
            self.assertTrue(written["synthetic_graphs"].exists())
            self.assertTrue(written["manifest"].exists())

            first_line = written["train"].read_text(encoding="utf-8").splitlines()[0]
            payload = json.loads(first_line)
            self.assertIn("question", payload)
            self.assertIn("route_label", payload)
            self.assertIn("metadata", payload)
            self.assertIn("supervision_target", payload)
            self.assertIn("case_id", payload)
            self.assertIn("template_id", payload)
            self.assertIn("variant_id", payload)
            self.assertEqual(payload["supervision_target"]["route_label"], payload["route_label"])

            manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["split_sizes"]["release_eval"], 21)


if __name__ == "__main__":
    unittest.main()
