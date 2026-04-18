import unittest

from runtime.query_planner import QueryPlanEnvelope, QueryPlanPayload, compile_query_plan


class QueryPlannerTests(unittest.TestCase):
    def test_compiles_segments_by_company_with_multi_company_filter(self):
        plan = QueryPlanEnvelope(
            answerable=True,
            family="segments_by_company",
            payload=QueryPlanPayload(companies=["Apple", "Microsoft"]),
        )

        result = compile_query_plan(plan)

        self.assertTrue(result.answerable)
        self.assertEqual(result.params, {"companies": ["Apple", "Microsoft"]})
        self.assertIn("c.name IN $companies", result.cypher or "")
        self.assertIn("RETURN DISTINCT c.name AS company, s.name AS segment", result.cypher or "")

    def test_compiles_same_segment_filters_with_closed_label_normalization(self):
        plan = QueryPlanEnvelope(
            answerable=True,
            family="companies_by_segment_filters",
            payload=QueryPlanPayload(
                customer_types=["healthcare firms"],
                channels=["resellers"],
            ),
        )

        result = compile_query_plan(plan)

        self.assertTrue(result.answerable)
        self.assertEqual(result.params["customer_type"], "healthcare organizations")
        self.assertEqual(result.params["channel"], "resellers")
        self.assertIn("MATCH (s)-[:SERVES]->(:CustomerType {name: $customer_type})", result.cypher or "")

    def test_compiles_cross_segment_company_query_with_distinct_segment_aliases(self):
        plan = QueryPlanEnvelope(
            answerable=True,
            family="companies_by_cross_segment_filters",
            payload=QueryPlanPayload(
                customer_types=["government agencies", "healthcare organizations"],
                binding_scope="across_segments",
            ),
        )

        result = compile_query_plan(plan)

        self.assertTrue(result.answerable)
        self.assertIn("(s1:BusinessSegment)", result.cypher or "")
        self.assertIn("(s2:BusinessSegment)", result.cypher or "")
        self.assertIn("$customer_type_1", result.cypher or "")
        self.assertIn("$customer_type_2", result.cypher or "")

    def test_compiles_place_query_with_rollup_properties(self):
        plan = QueryPlanEnvelope(
            answerable=True,
            family="companies_by_place",
            payload=QueryPlanPayload(places=["U.S."]),
        )

        result = compile_query_plan(plan)

        self.assertTrue(result.answerable)
        self.assertEqual(result.params["places"], ["United States"])
        self.assertIn("coalesce(place.includes_places, [])", result.cypher or "")
        self.assertIn("coalesce(place.within_places, [])", result.cypher or "")

    def test_compiles_boolean_wrapper_on_lookup_family(self):
        plan = QueryPlanEnvelope(
            answerable=True,
            family="boolean_exists",
            payload=QueryPlanPayload(
                base_family="companies_by_partner",
                partners=["Dell"],
            ),
        )

        result = compile_query_plan(plan)

        self.assertTrue(result.answerable)
        self.assertIn("COUNT(DISTINCT company.name) > 0 AS is_match", result.cypher or "")

    def test_compiles_count_wrapper_for_segment_lookup(self):
        plan = QueryPlanEnvelope(
            answerable=True,
            family="count_aggregate",
            payload=QueryPlanPayload(
                base_family="segments_by_company",
                companies=["Apple", "Microsoft"],
                aggregate_spec={
                    "kind": "count",
                    "base_family": "segments_by_company",
                    "count_target": "segment",
                },
            ),
        )

        result = compile_query_plan(plan)

        self.assertTrue(result.answerable)
        self.assertIn("COUNT(DISTINCT [c.name, s.name]) AS segment_count", result.cypher or "")

    def test_compiles_company_by_matched_segment_count_ranking(self):
        plan = QueryPlanEnvelope(
            answerable=True,
            family="ranking_topk",
            payload=QueryPlanPayload(
                customer_types=["developers"],
                aggregate_spec={
                    "kind": "ranking",
                    "ranking_metric": "company_by_matched_segment_count",
                },
                limit=3,
            ),
        )

        result = compile_query_plan(plan)

        self.assertTrue(result.answerable)
        self.assertIn("COUNT(DISTINCT s.name) AS segment_count", result.cypher or "")
        self.assertIn("LIMIT $limit", result.cypher or "")
        self.assertEqual(result.params["limit"], 3)

    def test_refuses_unknown_closed_label(self):
        plan = QueryPlanEnvelope(
            answerable=True,
            family="companies_by_segment_filters",
            payload=QueryPlanPayload(customer_types=["astronauts"]),
        )

        result = compile_query_plan(plan)

        self.assertFalse(result.answerable)
        self.assertEqual(result.reason, "ambiguous_closed_label")

    def test_refuses_unsupported_count_target_for_companies_list(self):
        plan = QueryPlanEnvelope(
            answerable=True,
            family="count_aggregate",
            payload=QueryPlanPayload(
                aggregate_spec={
                    "kind": "count",
                    "base_family": "companies_list",
                    "count_target": "segment",
                },
            ),
        )

        result = compile_query_plan(plan)

        self.assertFalse(result.answerable)
        self.assertEqual(result.reason, "beyond_local_coverage")


if __name__ == "__main__":
    unittest.main()
