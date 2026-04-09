import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finreflectkg_stage3 import Stage3TeacherAugmentor, triple_key
from stage3_prompt_profiles import DEFAULT_PROMPT_PROFILE, list_prompt_profiles
from stage3_smoke_cases import SMOKE_CASES, smoke_case_example


def _serialize_triples(triples):
    return sorted(triples, key=lambda triple: triple_key(triple))


def evaluate_case_report(case, report):
    actual = _serialize_triples(report["valid_triples"])
    expected = _serialize_triples(list(case.expected_triples))
    exact_match = actual == expected

    failure_reasons = []
    if report.get("error"):
        failure_reasons.append("teacher_call_error")
    if report.get("invalid_triple_count", 0):
        failure_reasons.append("invalid_triples_emitted")
    if report.get("duplicate_triple_count", 0):
        failure_reasons.append("duplicate_triples_emitted")
    if report.get("grounding_rejection_count", 0):
        failure_reasons.append("grounding_rejections")
    if not exact_match:
        failure_reasons.append("expected_triples_mismatch")

    return {
        "passed": not failure_reasons,
        "exact_match": exact_match,
        "expected_triples": expected,
        "actual_triples": actual,
        "failure_reasons": failure_reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed Stage 3 smoke cases against a local teacher model.")
    parser.add_argument("--base-url", type=str, default="http://localhost:1234/v1", help="Base URL for the local LLM endpoint.")
    parser.add_argument("--model", type=str, default="local-model", help="Model id sent to the local LLM endpoint.")
    parser.add_argument("--api-key", type=str, default="lm-studio", help="API key for the local LLM endpoint.")
    parser.add_argument("--prompt-profile", type=str, default=DEFAULT_PROMPT_PROFILE, choices=list_prompt_profiles(), help="Named Stage 3 prompt profile to use.")
    parser.add_argument("--max-retries", type=int, default=1, help="Max retries per case.")
    parser.add_argument(
        "--report-path",
        type=str,
        default="outputs/stage3_smoke_eval/report.json",
        help="Where to write the smoke evaluation report.",
    )
    args = parser.parse_args()

    augmentor = Stage3TeacherAugmentor(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        prompt_profile=args.prompt_profile,
    )

    case_reports = []
    passed_case_count = 0
    exact_match_case_count = 0
    for case in SMOKE_CASES:
        example = smoke_case_example(case)
        report = augmentor.run_relation(
            example,
            relation=case.relation,
            max_retries=args.max_retries,
        )
        evaluation = evaluate_case_report(case, report)
        passed = evaluation["passed"]
        if passed:
            passed_case_count += 1
        if evaluation["exact_match"]:
            exact_match_case_count += 1

        case_reports.append(
            {
                "case_id": case.case_id,
                "relation": case.relation,
                "passed": passed,
                "exact_match": evaluation["exact_match"],
                "expected_triples": evaluation["expected_triples"],
                "actual_triples": evaluation["actual_triples"],
                "failure_reasons": evaluation["failure_reasons"],
                "attempts_used": report["attempts_used"],
                "raw_response": report.get("raw_response", ""),
                "parsed_triples": report.get("parsed_triples", []),
                "invalid_triple_count": report["invalid_triple_count"],
                "duplicate_triple_count": report["duplicate_triple_count"],
                "grounding_rejection_count": report["grounding_rejection_count"],
                "grounding_rejections": report.get("grounding_rejections", []),
                "error": report.get("error"),
            }
        )

    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "prompt_profile": args.prompt_profile,
        "case_count": len(SMOKE_CASES),
        "exact_match_case_count": exact_match_case_count,
        "passed_case_count": passed_case_count,
        "failed_case_count": len(SMOKE_CASES) - passed_case_count,
        "pass_rate": (passed_case_count / len(SMOKE_CASES)) if SMOKE_CASES else 0.0,
        "cases": case_reports,
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Prompt profile: {args.prompt_profile}")
    print(f"Exact matches: {exact_match_case_count}/{len(SMOKE_CASES)}")
    print(f"Strict passes: {passed_case_count}/{len(SMOKE_CASES)} smoke cases")
    print(f"Wrote smoke report to {report_path}")
    return 0 if passed_case_count == len(SMOKE_CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
