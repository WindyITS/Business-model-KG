import argparse
import json
import logging
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from entity_resolver import resolve_entities
from llm_extractor import (
    CanonicalPipelineResult,
    ExtractionError,
    LLMExtractor,
)
from model_provider import resolve_model_settings
from ontology_validator import validate_triples

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("neo4j.notifications").disabled = True

TOTAL_STAGES = 10
CONSOLE_RULE = "=" * 50
CONSOLE_SEPARATOR = "-" * 50


def _console_print(line: str = "") -> None:
    print(line, flush=True)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"

    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def _format_token_visual(tokens: int, token_cap: int | None = None) -> str:
    if token_cap and token_cap > 0:
        return f"{tokens:,}/{token_cap:,}"
    return f"{tokens:,}"


class PipelineConsole:
    def __init__(self, total_stages: int = TOTAL_STAGES, printer: Callable[[str], None] = _console_print):
        self.total_stages = total_stages
        self._printer = printer
        self._run_started_at = perf_counter()
        self._header_printed = False
        self._current_stage: dict[str, Any] | None = None
        self._llm_token_cap: int | None = None

    @property
    def header_printed(self) -> bool:
        return self._header_printed

    def _print_detail(self, label: str, value: Any) -> None:
        self._printer(f"  {f'{label}:':<18}{value}")

    @staticmethod
    def _format_status_message(message: Any, *, max_length: int = 220) -> str:
        compact = " ".join(str(message).split())
        if len(compact) <= max_length:
            return compact
        return f"{compact[: max_length - 3]}..."

    def start_run(
        self,
        *,
        started_at: datetime,
        source_file: Path,
        run_dir: Path,
        pipeline: str,
        provider: str,
        model: str,
        neo4j_enabled: bool,
        llm_token_cap: int | None = None,
    ) -> None:
        self._run_started_at = perf_counter()
        self._header_printed = True
        self._llm_token_cap = llm_token_cap
        neo4j_status = "enabled (notifications disabled)" if neo4j_enabled else "skipped"

        self._printer(CONSOLE_RULE)
        self._printer("KG PIPELINE RUN")
        self._printer(CONSOLE_RULE)
        self._printer(f"Started:   {started_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')}")
        self._printer(f"Input:     {source_file}")
        self._printer(f"Run dir:   {run_dir}")
        self._printer(f"Pipeline:  {pipeline}")
        self._printer(f"Provider:  {provider}")
        self._printer(f"Model:     {model}")
        self._printer(f"Neo4j:     {neo4j_status}")
        self._printer("")

    def start_stage(
        self,
        index: int,
        title: str,
        *,
        extracts: str | None = None,
        details: list[tuple[str, Any]] | None = None,
    ) -> None:
        self._current_stage = {
            "index": index,
            "title": title,
            "started_at": perf_counter(),
            "llm": None,
        }
        self._printer(f"[{index:02d}/{self.total_stages:02d}] {title}")
        if extracts:
            self._print_detail("extracts", extracts)
        for label, value in details or []:
            self._print_detail(label, value)

    def record_llm_call(self, *, attempt: int, max_retries: int, tokens: int | None = None) -> None:
        if self._current_stage is None:
            return
        self._current_stage["llm"] = {
            "attempt": attempt,
            "max_retries": max_retries,
            "tokens": tokens,
        }

    def start_llm_attempt(self, *, attempt: int, max_retries: int) -> None:
        if self._current_stage is None:
            return
        self._print_detail("llm", f"starting attempt {attempt}/{max_retries}")

    def fail_llm_attempt(self, *, attempt: int, max_retries: int, error: Any, will_retry: bool) -> None:
        if self._current_stage is None:
            return

        retry_status = "retrying" if will_retry else "no retries left"
        self._print_detail(
            "llm",
            f"attempt {attempt}/{max_retries} failed, {retry_status}: {self._format_status_message(error)}",
        )

    def finish_stage(self, *, status: str = "done", details: list[tuple[str, Any]] | None = None) -> None:
        if self._current_stage is None:
            return

        stage = self._current_stage
        llm = stage.get("llm")
        if llm:
            llm_summary = f"attempt {llm['attempt']}/{llm['max_retries']}"
            if llm.get("tokens") is not None:
                llm_summary += f", tokens={_format_token_visual(llm['tokens'], self._llm_token_cap)}"
            self._print_detail("llm", llm_summary)

        for label, value in details or []:
            self._print_detail(label, value)

        self._print_detail("status", status)
        self._print_detail("time", _format_duration(perf_counter() - stage["started_at"]))
        self._printer("")
        self._current_stage = None

    def handle_progress(self, event: str, **payload: Any) -> None:
        if event == "stage_start":
            self.start_stage(
                payload["index"],
                payload["title"],
                extracts=payload.get("extracts"),
                details=payload.get("details"),
            )
            return
        if event == "llm_call_start":
            self.start_llm_attempt(
                attempt=payload["attempt"],
                max_retries=payload["max_retries"],
            )
            return
        if event == "llm_call_error":
            self.fail_llm_attempt(
                attempt=payload["attempt"],
                max_retries=payload["max_retries"],
                error=payload["error"],
                will_retry=payload["will_retry"],
            )
            return
        if event == "llm_call_complete":
            self.record_llm_call(
                attempt=payload["attempt"],
                max_retries=payload["max_retries"],
                tokens=payload.get("tokens"),
            )
            return
        if event == "stage_complete":
            self.finish_stage(status=payload.get("status", "done"), details=payload.get("details"))
            return
        if event == "stage_failed":
            self.finish_stage(status="failed", details=[("error", payload["error"])])

    def complete_run(
        self,
        *,
        status: str,
        artifacts: Path,
        resolved_triples: int | None = None,
        error: str | None = None,
    ) -> None:
        self._printer(CONSOLE_SEPARATOR)
        self._printer("RUN COMPLETE" if status == "success" else "RUN FAILED")
        self._print_detail("status", status)
        if resolved_triples is not None:
            self._print_detail("resolved triples", resolved_triples)
        if error:
            self._print_detail("error", error)
        self._print_detail("duration", _format_duration(perf_counter() - self._run_started_at))
        self._print_detail("artifacts", artifacts)
        self._printer(CONSOLE_RULE)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _mode_name(args: argparse.Namespace) -> str:
    return f"{args.pipeline}_pipeline"


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
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or ID to use. For opencode-go this repo supports kimi-k2.5, mimo-v2-pro, and minimax-m2.7.",
    )
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
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum LLM retries per call.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory where run artifacts are written.")
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip Neo4j loading and only write extraction artifacts.")
    parser.add_argument("--clear-neo4j", action="store_true", help="Clear the Neo4j database before loading triples. Use with care.")
    args = parser.parse_args()

    source_file = Path(args.file_path)
    mode = _mode_name(args)
    ontology_version = "canonical"
    run_started_at = datetime.now(timezone.utc)
    run_dir = _build_run_dir(Path(args.output_dir), source_file, mode)
    console = PipelineConsole()
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
        "started_at": run_started_at.isoformat(),
    }
    _write_json(run_dir / "run_summary.json", summary)

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
            }
        )
        _write_json(run_dir / "run_summary.json", summary)

        console.start_run(
            started_at=run_started_at,
            source_file=source_file,
            run_dir=run_dir,
            pipeline=args.pipeline,
            provider=model_settings.provider,
            model=model_settings.model,
            neo4j_enabled=not args.skip_neo4j,
            llm_token_cap=model_settings.max_output_tokens,
        )

        console.start_stage(1, "Read input")
        full_text = source_file.read_text(encoding="utf-8")
        chunks = [full_text]
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
        console.finish_stage(details=[("chunks", len(chunks))])

        extractor = LLMExtractor(
            base_url=model_settings.base_url,
            api_key=model_settings.api_key,
            model=model_settings.model,
            provider=model_settings.provider,
            api_mode=model_settings.api_mode,
            max_output_tokens=model_settings.max_output_tokens,
            progress_callback=console.handle_progress,
        )

        chat_result: CanonicalPipelineResult = extractor.extract_canonical_pipeline(
            full_text=full_text,
            company_name=company_name,
            max_retries=args.max_retries,
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
            run_dir / "rule_reflection_extraction.json",
            {
                "triples": [t.model_dump() for t in chat_result.rule_reflection_extraction.triples],
                "extraction_notes": chat_result.rule_reflection_extraction.extraction_notes,
                "attempts_used": chat_result.rule_reflection_attempts_used,
                "raw_response": chat_result.raw_rule_reflection_response,
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
            "rule_reflection_extraction": chat_result.rule_reflection_extraction.model_dump(),
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
            "rule_reflection": chat_result.rule_reflection_audit,
            "reflection": chat_result.final_reflection_audit,
        }
        final_output_audit = chat_result.final_reflection_audit

        _write_json(run_dir / "extractions.json", extraction_payload)
        _write_json(run_dir / "extraction_audits.json", stage_audits)
        _write_json(run_dir / "final_output_validation_report.json", final_output_audit)

        console.start_stage(9, "Resolve + validate")
        resolved_triples = resolve_entities(extractions)
        if not resolved_triples:
            raise ExtractionError("Extraction completed but produced zero resolved triples.")
        raw_resolved_triple_count = len(resolved_triples)

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
        console.finish_stage(
            details=[
                ("raw triples", raw_resolved_triple_count),
                ("resolved triples", len(resolved_triples)),
                ("invalid removed", validation_report["summary"]["invalid_triple_count"]),
                ("duplicates removed", validation_report["summary"]["duplicate_triple_count"]),
            ]
        )

        loaded_triples = 0
        console.start_stage(10, "Load Neo4j")
        if args.skip_neo4j:
            console.finish_stage(status="skipped", details=[("load", "skipped by request")])
        else:
            from neo4j_loader import Neo4jLoader

            loader = Neo4jLoader(uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
            graph_cleared = False
            try:
                if args.clear_neo4j:
                    loader.clear_graph()
                    graph_cleared = True
                loader.setup_constraints()
                loaded_triples = loader.load_triples(resolved_triples)
            finally:
                loader.close()

            neo4j_details: list[tuple[str, Any]] = []
            if graph_cleared:
                neo4j_details.append(("database", "cleared before load"))
            neo4j_details.append(("constraints", "checked"))
            neo4j_details.append(("triples loaded", loaded_triples))
            console.finish_stage(details=neo4j_details)

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
        console.complete_run(status="success", artifacts=run_dir, resolved_triples=len(resolved_triples))
        return 0
    except Exception as exc:
        if not console.header_printed:
            console.start_run(
                started_at=run_started_at,
                source_file=source_file,
                run_dir=run_dir,
                pipeline=args.pipeline,
                provider=str(summary.get("provider", args.provider)),
                model=str(summary.get("model") or summary.get("requested_model") or "<default>"),
                neo4j_enabled=not args.skip_neo4j,
                llm_token_cap=summary.get("max_output_tokens"),
            )
        console.finish_stage(status="failed", details=[("error", str(exc))])
        summary.update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }
        )
        _write_json(run_dir / "run_summary.json", summary)
        console.complete_run(status="failed", artifacts=run_dir, error=str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
