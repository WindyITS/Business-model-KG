"""Evaluate extraction pipeline outputs against clean gold benchmark triples."""

from __future__ import annotations

import argparse
import json
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.output_layout import slugify_company_name


QUOTE_CHARS = "\"'`“”‘’ "
TRIPLE_FIELDS: tuple[str, ...] = ("subject", "subject_type", "relation", "object", "object_type")
SPLITS: tuple[str, ...] = ("dev", "test")


Triple = dict[str, str]
TripleKey = tuple[str, str, str, str, str]


@dataclass(frozen=True)
class EvaluationPaths:
    gold_path: Path
    prediction_path: Path
    output_dir: Path
    company: str
    company_slug: str
    pipeline: str
    split: str | None = None


def clean_entity_name(name: str) -> str:
    cleaned = unicodedata.normalize("NFKC", name).strip()
    cleaned = cleaned.strip(QUOTE_CHARS)
    return " ".join(cleaned.split())


def entity_key(name: str) -> str:
    cleaned = clean_entity_name(name).casefold()
    cleaned = cleaned.replace("’", "'")
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    return cleaned


def triple_key(triple: Triple) -> TripleKey:
    return (
        entity_key(triple["subject"]),
        triple["subject_type"].strip(),
        triple["relation"].strip(),
        entity_key(triple["object"]),
        triple["object_type"].strip(),
    )


def cleaned_triple(triple: dict[str, Any]) -> Triple:
    return {
        "subject": clean_entity_name(str(triple.get("subject", ""))),
        "subject_type": str(triple.get("subject_type", "")).strip(),
        "relation": str(triple.get("relation", "")).strip(),
        "object": clean_entity_name(str(triple.get("object", ""))),
        "object_type": str(triple.get("object_type", "")).strip(),
    }


def read_jsonl(path: Path) -> list[Triple]:
    rows: list[Triple] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object.")
        rows.append(cleaned_triple(payload))
    return rows


def read_prediction_triples(path: Path) -> list[Triple]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    triples = payload.get("triples") if isinstance(payload, dict) else None
    if not isinstance(triples, list):
        raise ValueError(f"{path} does not contain a top-level 'triples' list.")
    return [cleaned_triple(triple) for triple in triples if isinstance(triple, dict)]


def unique_by_key(triples: list[Triple]) -> dict[TripleKey, Triple]:
    records: dict[TripleKey, Triple] = {}
    for triple in triples:
        records.setdefault(triple_key(triple), triple)
    return records


def metric_payload(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def sort_triples(triples: list[Triple]) -> list[Triple]:
    return sorted(triples, key=lambda triple: tuple(triple[field] for field in TRIPLE_FIELDS))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Triple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def result_folder_has_files(path: Path) -> bool:
    return path.exists() and any(candidate.is_file() for candidate in path.rglob("*"))


def confirm_overwrite(path: Path) -> bool:
    response = input(
        f"There are already files in the results folder {path}. "
        "Proceeding with a new evaluation is going to overwrite them. Do you want to proceed? [Y/n] "
    ).strip()
    return response.casefold() not in {"n", "no"}


def prepare_result_folder(path: Path, *, assume_yes: bool = False) -> bool:
    if not result_folder_has_files(path):
        return True

    if not assume_yes and not confirm_overwrite(path):
        return False

    shutil.rmtree(path)
    return True


def evaluate_triples(gold_triples: list[Triple], predicted_triples: list[Triple]) -> dict[str, Any]:
    gold_by_key = unique_by_key(gold_triples)
    predicted_by_key = unique_by_key(predicted_triples)

    gold_keys = set(gold_by_key)
    predicted_keys = set(predicted_by_key)
    matched_keys = gold_keys & predicted_keys
    false_positive_keys = predicted_keys - gold_keys
    false_negative_keys = gold_keys - predicted_keys

    return {
        "metrics": metric_payload(
            tp=len(matched_keys),
            fp=len(false_positive_keys),
            fn=len(false_negative_keys),
        ),
        "counts": {
            "gold_triples": len(gold_triples),
            "gold_unique_triples": len(gold_by_key),
            "predicted_triples": len(predicted_triples),
            "predicted_unique_triples": len(predicted_by_key),
        },
        "matched": sort_triples([gold_by_key[key] for key in matched_keys]),
        "false_positives": sort_triples([predicted_by_key[key] for key in false_positive_keys]),
        "false_negatives": sort_triples([gold_by_key[key] for key in false_negative_keys]),
    }


def evaluate_company(paths: EvaluationPaths) -> dict[str, Any]:
    if not paths.prediction_path.is_file():
        result = {
            "company": paths.company,
            "company_slug": paths.company_slug,
            "pipeline": paths.pipeline,
            "split": paths.split,
            "status": "missing_prediction",
            "gold_path": str(paths.gold_path),
            "prediction_path": str(paths.prediction_path),
        }
        write_json(paths.output_dir / "metrics.json", result)
        return result

    gold_triples = read_jsonl(paths.gold_path)
    predicted_triples = read_prediction_triples(paths.prediction_path)
    result = evaluate_triples(gold_triples, predicted_triples)
    summary = {
        "company": paths.company,
        "company_slug": paths.company_slug,
        "pipeline": paths.pipeline,
        "split": paths.split,
        "status": "evaluated",
        "gold_path": str(paths.gold_path),
        "prediction_path": str(paths.prediction_path),
        **result["counts"],
        **result["metrics"],
    }

    write_json(paths.output_dir / "metrics.json", summary)
    write_jsonl(paths.output_dir / "matched.jsonl", result["matched"])
    write_jsonl(paths.output_dir / "false_positives.jsonl", result["false_positives"])
    write_jsonl(paths.output_dir / "false_negatives.jsonl", result["false_negatives"])
    return summary


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [result for result in results if result.get("status") == "evaluated"]
    tp = sum(int(result["true_positives"]) for result in evaluated)
    fp = sum(int(result["false_positives"]) for result in evaluated)
    fn = sum(int(result["false_negatives"]) for result in evaluated)
    return {
        "evaluated_company_count": len(evaluated),
        "missing_prediction_count": sum(1 for result in results if result.get("status") == "missing_prediction"),
        **metric_payload(tp, fp, fn),
    }


def prediction_path(outputs_root: Path, company_slug: str, pipeline: str) -> Path:
    return outputs_root / company_slug / pipeline / "latest" / "resolved_triples.json"


def company_name_from_gold_path(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").title()


def build_split_evaluation_paths(
    *,
    root: Path,
    outputs_root: Path,
    pipeline: str,
    split: str,
) -> list[EvaluationPaths]:
    clean_dir = root / "benchmarks" / split / "clean"
    result_root = root / "results" / pipeline / split
    gold_paths = sorted(path for path in clean_dir.glob("*.jsonl") if path.is_file())
    paths: list[EvaluationPaths] = []
    for gold_path in gold_paths:
        company_slug = slugify_company_name(gold_path.stem)
        company = company_name_from_gold_path(gold_path)
        paths.append(
            EvaluationPaths(
                gold_path=gold_path,
                prediction_path=prediction_path(outputs_root, company_slug, pipeline),
                output_dir=result_root / "companies" / company_slug,
                company=company,
                company_slug=company_slug,
                pipeline=pipeline,
                split=split,
            )
        )
    return paths


def find_cherry_pick_gold_path(root: Path, company: str) -> tuple[Path, str]:
    company_slug = slugify_company_name(company)
    matches: list[tuple[Path, str]] = []
    for split in SPLITS:
        candidate = root / "benchmarks" / split / "clean" / f"{company_slug}.jsonl"
        if candidate.is_file():
            matches.append((candidate, split))

    if not matches:
        raise FileNotFoundError(f"No clean gold benchmark found for company {company!r}. Expected {company_slug}.jsonl.")
    if len(matches) > 1:
        locations = ", ".join(str(path) for path, _split in matches)
        raise ValueError(f"Company {company!r} appears in multiple benchmark splits: {locations}")
    return matches[0]


def build_cherry_pick_evaluation_path(
    *,
    root: Path,
    outputs_root: Path,
    pipeline: str,
    company: str,
) -> EvaluationPaths:
    company_slug = slugify_company_name(company)
    gold_path, _split = find_cherry_pick_gold_path(root, company)
    return EvaluationPaths(
        gold_path=gold_path,
        prediction_path=prediction_path(outputs_root, company_slug, pipeline),
        output_dir=root / "results" / "cherry_picked" / pipeline / company_slug,
        company=company,
        company_slug=company_slug,
        pipeline=pipeline,
        split=None,
    )


def evaluate_paths(paths: list[EvaluationPaths], *, output_root: Path) -> dict[str, Any]:
    results = [evaluate_company(path) for path in paths]
    summary = {
        "result_count": len(results),
        "results": results,
        "aggregate": aggregate_metrics(results),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate extraction outputs against clean gold benchmark triples.")
    parser.add_argument("--pipeline", required=True, help="Extraction pipeline to evaluate, for example zero-shot or analyst.")
    parser.add_argument("--split", choices=SPLITS, default=None, help="Benchmark split for all-company evaluation.")
    parser.add_argument("--company", default=None, help="Single company to evaluate in cherry-picked mode.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Evaluation folder root. Defaults to the repo's evaluation/ folder.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs",
        help="Pipeline outputs root. Defaults to the repo's outputs/ folder.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Overwrite existing result files without prompting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    outputs_root = args.outputs_root.resolve()

    if args.company and args.split:
        raise SystemExit("--company and --split are separate modes. Use --company for cherry-picked evaluation or --split for all-company evaluation.")
    if not args.company and not args.split:
        raise SystemExit("Provide either --split for all-company evaluation or --company for cherry-picked evaluation.")

    if args.company:
        path = build_cherry_pick_evaluation_path(
            root=root,
            outputs_root=outputs_root,
            pipeline=args.pipeline,
            company=args.company,
        )
        output_root = path.output_dir
        if not prepare_result_folder(output_root, assume_yes=args.yes):
            print("Evaluation cancelled. Existing results were left unchanged.")
            return 0
        summary = evaluate_paths([path], output_root=output_root)
    else:
        paths = build_split_evaluation_paths(
            root=root,
            outputs_root=outputs_root,
            pipeline=args.pipeline,
            split=args.split,
        )
        output_root = root / "results" / args.pipeline / args.split
        if not prepare_result_folder(output_root, assume_yes=args.yes):
            print("Evaluation cancelled. Existing results were left unchanged.")
            return 0
        summary = evaluate_paths(paths, output_root=output_root)

    aggregate = summary["aggregate"]
    print(
        "Evaluated "
        f"{aggregate['evaluated_company_count']} companies "
        f"(missing predictions: {aggregate['missing_prediction_count']}). "
        f"F1={aggregate['f1']:.3f}, precision={aggregate['precision']:.3f}, recall={aggregate['recall']:.3f}"
    )
    print(f"Results: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
