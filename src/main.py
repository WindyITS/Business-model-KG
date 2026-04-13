import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from entity_resolver import resolve_entities
from llm_extractor import (
    CanonicalPipelineResult,
    ExtractionError,
    LLMExtractor,
)
from model_provider import resolve_model_settings
from ontology_validator import validate_triples

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _mode_name(args: argparse.Namespace) -> str:
    return f"{args.pipeline}_pipeline"


def _effective_use_schema(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "use_schema", False))


def _build_run_dir(output_dir: Path, source_file: Path, mode: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"{source_file.stem}_{mode}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _looks_like_company_name(candidate: str) -> bool:
    candidate = candidate.strip(" ,.-")
    if not candidate or len(candidate) > 80:
        return False

    tokens = [token.strip("()[]{}\"'.,") for token in candidate.split()]
    tokens = [token for token in tokens if token]
    if not 1 <= len(tokens) <= 8:
        return False

    connector_words = {"&", "and", "of", "the"}
    signal_tokens = 0
    for token in tokens:
        if token.casefold() in connector_words:
            continue
        if token[0].isupper() or "." in token or any(char.isdigit() for char in token):
            signal_tokens += 1
            continue
        return False

    return signal_tokens > 0


def _infer_company_name(source_file: Path, full_text: str) -> str:
    for line in full_text.splitlines()[:40]:
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("item ") or lowered.startswith("part "):
            continue
        marker = " is "
        if marker in lowered:
            company = stripped[: lowered.index(marker)].strip(" ,.-")
            if _looks_like_company_name(company):
                return company

    stem = source_file.stem.replace("_10k", "").replace("_", " ").strip()
    return stem.title()


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a business-model knowledge graph from SEC 10-K filings.")
    parser.add_argument("file_path", type=str, help="Path to the .txt file containing the 10-K business section.")
    parser.add_argument(
        "--provider",
        choices=["local", "opencode-go"],
        default="local",
        help="Provider preset. Use local for LM Studio/local OpenAI-compatible endpoints or opencode-go for hosted OpenCode Go models.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL root for the selected API. The resolver accepts either the API root or a full documented endpoint and normalizes common suffixes.",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name or ID to use.")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the selected provider. If omitted, provider-specific environment variables are used.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum completion tokens per model call. By default, opencode-go uses a capped value to reduce long-running requests; local leaves it unset.",
    )
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687", help="Neo4j connection URI.")
    parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Neo4j username.")
    parser.add_argument("--neo4j-password", type=str, default="password", help="Neo4j password.")
    parser.add_argument(
        "--pipeline",
        choices=["canonical"],
        default="canonical",
        help="Extraction pipeline to run. Only the canonical production pipeline is supported.",
    )
    parser.add_argument(
        "--no-schema",
        action="store_true",
        help="Deprecated compatibility flag. No-schema mode is already the default.",
    )
    parser.add_argument(
        "--use-schema",
        action="store_true",
        help="Enable response_format JSON Schema enforcement. By default the pipeline runs in no-schema mode.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum LLM retries per call.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory where run artifacts are written.")
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip Neo4j loading and only write extraction artifacts.")
    parser.add_argument("--clear-neo4j", action="store_true", help="Clear the Neo4j database before loading triples. Use with care.")
    args = parser.parse_args()

    source_file = Path(args.file_path)
    mode = _mode_name(args)
    use_schema = _effective_use_schema(args)
    ontology_version = "canonical"
    run_dir = _build_run_dir(Path(args.output_dir), source_file, mode)
    summary: dict[str, Any] = {
        "status": "running",
        "mode": mode,
        "ontology_version": ontology_version,
        "source_file": str(source_file),
        "run_dir": str(run_dir),
        "provider": args.provider,
        "requested_model": args.model,
        "requested_base_url": args.base_url,
        "requested_max_output_tokens": args.max_output_tokens,
        "requested_no_schema": args.no_schema,
        "requested_use_schema": args.use_schema,
        "use_schema": use_schema,
        "schema_mode": "schema" if use_schema else "no-schema",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_dir / "run_summary.json", summary)

    logger.info("Starting pipeline on %s", args.file_path)
    logger.info("Writing run artifacts to %s", run_dir)

    try:
        model_settings = resolve_model_settings(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_output_tokens=args.max_output_tokens,
        )
        summary.update(
            {
                "provider": model_settings.provider,
                "model": model_settings.model,
                "base_url": model_settings.base_url,
                "api_mode": model_settings.api_mode,
                "max_output_tokens": model_settings.max_output_tokens,
                "use_schema": use_schema,
                "schema_mode": "schema" if use_schema else "no-schema",
            }
        )
        _write_json(run_dir / "run_summary.json", summary)
        logger.info(
            "Run settings: pipeline=%s provider=%s model=%s schema_mode=%s max_output_tokens=%s base_url=%s",
            mode,
            model_settings.provider,
            model_settings.model,
            "schema" if use_schema else "no-schema",
            model_settings.max_output_tokens,
            model_settings.base_url,
        )

        logger.info("Reading full text without chunking...")
        full_text = source_file.read_text(encoding="utf-8")
        chunks = [full_text]
        logger.info("Prepared %s chunks.", len(chunks))
        company_name = _infer_company_name(source_file, full_text)
        _write_json(
            run_dir / "chunks.json",
            {
                "source_file": str(source_file),
                "mode": mode,
                "chunk_count": len(chunks),
                "chunks": [
                    {
                        "chunk_index": 0,
                        "text": full_text,
                        "character_count": len(full_text),
                        "word_count": len(full_text.split()),
                    }
                ],
            },
        )

        extractor = LLMExtractor(
            base_url=model_settings.base_url,
            api_key=model_settings.api_key,
            model=model_settings.model,
            provider=model_settings.provider,
            api_mode=model_settings.api_mode,
            max_output_tokens=model_settings.max_output_tokens,
        )
        logger.info("Starting LLM extraction in %s mode...", mode)

        chat_result: CanonicalPipelineResult = extractor.extract_canonical_pipeline(
            full_text=full_text,
            company_name=company_name,
            max_retries=args.max_retries,
            use_schema=use_schema,
        )
        if not chat_result.success:
            raise ExtractionError(chat_result.error or "Canonical pipeline failed.")
        _write_json(
            run_dir / "skeleton_extraction.json",
            {
                "triples": [t.model_dump() for t in chat_result.skeleton_extraction.triples],
                "extraction_notes": chat_result.skeleton_extraction.extraction_notes,
            },
        )
        _write_json(
            run_dir / "pass2_channels_extraction.json",
            {
                "triples": [t.model_dump() for t in chat_result.pass2_channels_extraction.triples],
                "extraction_notes": chat_result.pass2_channels_extraction.extraction_notes,
            },
        )
        _write_json(
            run_dir / "pass2_revenue_extraction.json",
            {
                "triples": [t.model_dump() for t in chat_result.pass2_revenue_extraction.triples],
                "extraction_notes": chat_result.pass2_revenue_extraction.extraction_notes,
            },
        )
        _write_json(
            run_dir / "pass2_commercial_extraction.json",
            {
                "triples": [t.model_dump() for t in chat_result.pass2_extraction.triples],
                "extraction_notes": chat_result.pass2_extraction.extraction_notes,
            },
        )
        _write_json(
            run_dir / "pass3_serves_extraction.json",
            {
                "triples": [t.model_dump() for t in chat_result.pass3_serves_extraction.triples],
                "extraction_notes": chat_result.pass3_serves_extraction.extraction_notes,
            },
        )
        _write_json(
            run_dir / "pass4_corporate_extraction.json",
            {
                "triples": [t.model_dump() for t in chat_result.pass4_corporate_extraction.triples],
                "extraction_notes": chat_result.pass4_corporate_extraction.extraction_notes,
            },
        )
        _write_json(
            run_dir / "pre_reflection_extraction.json",
            {
                "triples": [t.model_dump() for t in chat_result.pre_reflection_extraction.triples],
                "extraction_notes": chat_result.pre_reflection_extraction.extraction_notes,
            },
        )
        _write_json(
            run_dir / "reflection_extraction.json",
            {
                "triples": [t.model_dump() for t in chat_result.final_extraction.triples],
                "extraction_notes": chat_result.final_extraction.extraction_notes,
                "attempts_used": chat_result.final_reflection_attempts_used,
                "raw_response": chat_result.raw_final_reflection_response,
            },
        )
        extractions = [chat_result.final_extraction]
        extraction_payload = {
            "skeleton_extraction": chat_result.skeleton_extraction.model_dump(),
            "pass2_channels_extraction": chat_result.pass2_channels_extraction.model_dump(),
            "pass2_revenue_extraction": chat_result.pass2_revenue_extraction.model_dump(),
            "pass2_extraction": chat_result.pass2_extraction.model_dump(),
            "pass3_serves_extraction": chat_result.pass3_serves_extraction.model_dump(),
            "pass4_corporate_extraction": chat_result.pass4_corporate_extraction.model_dump(),
            "pre_reflection_extraction": chat_result.pre_reflection_extraction.model_dump(),
            "final_extraction": chat_result.final_extraction.model_dump(),
        }
        stage_audits = {
            "skeleton": chat_result.skeleton_audit,
            "pass2_channels": chat_result.pass2_channels_audit,
            "pass2_revenue": chat_result.pass2_revenue_audit,
            "pass2_commercial": chat_result.pass2_audit,
            "pass3_serves": chat_result.pass3_serves_audit,
            "pass4_corporate": chat_result.pass4_corporate_audit,
            "pre_reflection": chat_result.pre_reflection_audit,
            "reflection": chat_result.final_reflection_audit,
        }
        final_output_audit = chat_result.final_reflection_audit

        _write_json(run_dir / "extractions.json", extraction_payload)
        _write_json(run_dir / "extraction_audits.json", stage_audits)
        _write_json(run_dir / "final_output_validation_report.json", final_output_audit)

        logger.info("Resolving entities...")
        resolved_triples = resolve_entities(extractions)
        if not resolved_triples:
            raise ExtractionError("Extraction completed but produced zero resolved triples.")

        validation_report = validate_triples(
            [triple.model_dump() for triple in resolved_triples],
            source_text=full_text,
            require_text_grounding=False,
            dedupe=True,
            ontology_version=ontology_version,
        )
        _write_json(run_dir / "validation_report.json", validation_report)

        resolved_triples = [type(resolved_triples[0])(**triple) for triple in validation_report["valid_triples"]] if resolved_triples else []
        if not resolved_triples:
            raise ExtractionError("All resolved triples were rejected by ontology validation.")

        _write_json(
            run_dir / "resolved_triples.json",
            {
                "triple_count": len(resolved_triples),
                "triples": [triple.model_dump() for triple in resolved_triples],
            },
        )

        loaded_triples = 0
        if args.skip_neo4j:
            logger.info("Skipping Neo4j load by request.")
        else:
            logger.info("Loading triples into Neo4j...")
            from neo4j_loader import Neo4jLoader

            loader = Neo4jLoader(uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
            try:
                if args.clear_neo4j:
                    logger.warning("Clearing the entire Neo4j database before load because --clear-neo4j was provided.")
                    loader.clear_graph()
                loader.setup_constraints()
                loaded_triples = loader.load_triples(resolved_triples)
            finally:
                loader.close()

        summary.update(
            {
                "status": "success",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "chunk_count": len(chunks),
                "extraction_count": len(extractions),
                "final_output_raw_triple_count": int(final_output_audit.get("raw_triple_count", 0)),
                "final_output_malformed_triple_count": int(final_output_audit.get("malformed_triple_count", 0)),
                "final_output_ontology_rejected_triple_count": int(
                    final_output_audit.get("ontology_rejected_triple_count", 0)
                ),
                "final_output_duplicate_triple_count": int(final_output_audit.get("duplicate_triple_count", 0)),
                "final_output_kept_triple_count": int(final_output_audit.get("kept_triple_count", 0)),
                "resolved_triple_count": len(resolved_triples),
                "invalid_triple_count": validation_report["summary"]["invalid_triple_count"],
                "duplicate_triple_count": validation_report["summary"]["duplicate_triple_count"],
                "loaded_triple_count": loaded_triples,
                "skip_neo4j": args.skip_neo4j,
                "clear_neo4j": args.clear_neo4j,
            }
        )
        _write_json(run_dir / "run_summary.json", summary)
        logger.info("Pipeline complete. Resolved %s triples.", len(resolved_triples))
        return 0
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }
        )
        _write_json(run_dir / "run_summary.json", summary)
        logger.error("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
