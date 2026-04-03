import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from tqdm import tqdm

from chunker import read_and_chunk_file
from entity_resolver import canonical_entity_key, clean_entity_name, resolve_entities
from llm_extractor import ExtractionError, KnowledgeGraphExtraction, LLMExtractor
from neo4j_loader import Neo4jLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MEMORY_ENTITY_LIMIT_PER_TYPE = 12
MEMORY_RECENT_TRIPLES_LIMIT = 24


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _mode_name(args: argparse.Namespace) -> str:
    if args.incremental:
        return "incremental"
    if args.zero_shot:
        return "zero_shot"
    return "chunked"


def _build_run_dir(output_dir: Path, source_file: Path, mode: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"{source_file.stem}_{mode}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _serialize_extractions(extractions: List[KnowledgeGraphExtraction]) -> list[dict]:
    return [extraction.model_dump() for extraction in extractions]


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
            if company:
                return company

    stem = source_file.stem.replace("_10k", "").replace("_", " ").strip()
    return stem.title()


def _choose_surface(current: str, candidate: str) -> str:
    if candidate == current:
        return current
    current_letters = [ch for ch in current if ch.isalpha()]
    candidate_letters = [ch for ch in candidate if ch.isalpha()]
    current_mixed = any(ch.isupper() for ch in current_letters) and any(ch.islower() for ch in current_letters)
    candidate_mixed = any(ch.isupper() for ch in candidate_letters) and any(ch.islower() for ch in candidate_letters)
    current_score = (1 if current_mixed else 0, len(current))
    candidate_score = (1 if candidate_mixed else 0, len(candidate))
    return candidate if candidate_score > current_score else current


def _empty_memory_state(company_name: str) -> dict[str, Any]:
    return {
        "company_name": company_name,
        "entity_names": {},
        "entity_order": defaultdict(list),
        "recent_triples": [],
        "triples_seen": set(),
    }


def _update_memory_state(memory_state: dict[str, Any], extraction: KnowledgeGraphExtraction) -> None:
    for triple in extraction.triples:
        for value, node_type in (
            (triple.subject, triple.subject_type),
            (triple.object, triple.object_type),
        ):
            cleaned = clean_entity_name(value)
            if not cleaned:
                continue
            entity_key = (node_type, canonical_entity_key(cleaned))
            current = memory_state["entity_names"].get(entity_key)
            memory_state["entity_names"][entity_key] = cleaned if current is None else _choose_surface(current, cleaned)
            if entity_key not in memory_state["entity_order"][node_type]:
                memory_state["entity_order"][node_type].append(entity_key)

        triple_key = (
            triple.subject_type,
            canonical_entity_key(clean_entity_name(triple.subject)),
            triple.relation,
            triple.object_type,
            canonical_entity_key(clean_entity_name(triple.object)),
        )
        if triple_key in memory_state["triples_seen"]:
            continue

        memory_state["triples_seen"].add(triple_key)
        memory_state["recent_triples"].append(
            {
                "subject": clean_entity_name(triple.subject),
                "subject_type": triple.subject_type,
                "relation": triple.relation,
                "object": clean_entity_name(triple.object),
                "object_type": triple.object_type,
            }
        )
        if len(memory_state["recent_triples"]) > MEMORY_RECENT_TRIPLES_LIMIT:
            memory_state["recent_triples"] = memory_state["recent_triples"][-MEMORY_RECENT_TRIPLES_LIMIT:]


def _memory_snapshot(memory_state: dict[str, Any]) -> dict[str, Any]:
    known_entities = {}
    for node_type, ordered_keys in memory_state["entity_order"].items():
        surfaces = [memory_state["entity_names"][key] for key in ordered_keys[:MEMORY_ENTITY_LIMIT_PER_TYPE]]
        if surfaces:
            known_entities[node_type] = surfaces

    return {
        "company_name": memory_state["company_name"],
        "known_entities": known_entities,
        "recent_triples": memory_state["recent_triples"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a business-model knowledge graph from SEC 10-K filings.")
    parser.add_argument("file_path", type=str, help="Path to the .txt file containing the 10-K business section.")
    parser.add_argument("--base-url", type=str, default="http://localhost:1234/v1", help="Base URL for the LM Studio API.")
    parser.add_argument("--model", type=str, default="local-model", help="Model name required by the local LLM endpoint.")
    parser.add_argument("--api-key", type=str, default="lm-studio", help="API key for the local LLM service.")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687", help="Neo4j connection URI.")
    parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Neo4j username.")
    parser.add_argument("--neo4j-password", type=str, default="password", help="Neo4j password.")
    parser.add_argument("--zero-shot", action="store_true", help="Process the entire file in one prompt.")
    parser.add_argument("--incremental", action="store_true", help="Process the full document iteratively with memory.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum LLM retries per chunk or iteration.")
    parser.add_argument("--max-iterations", type=int, default=20, help="Maximum iterations for incremental mode.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory where run artifacts are written.")
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip Neo4j loading and only write extraction artifacts.")
    args = parser.parse_args()

    source_file = Path(args.file_path)
    mode = _mode_name(args)
    run_dir = _build_run_dir(Path(args.output_dir), source_file, mode)
    summary: dict[str, Any] = {
        "status": "running",
        "mode": mode,
        "source_file": str(source_file),
        "run_dir": str(run_dir),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_dir / "run_summary.json", summary)

    logger.info("Starting pipeline on %s", args.file_path)
    logger.info("Writing run artifacts to %s", run_dir)

    try:
        if args.zero_shot or args.incremental:
            logger.info("Reading full text without chunking...")
            full_text = source_file.read_text(encoding="utf-8")
            chunks = [full_text]
        else:
            logger.info("Chunking text with semantic chunker...")
            chunks = read_and_chunk_file(args.file_path)
            full_text = "\n\n".join(chunks)
        logger.info("Prepared %s chunks.", len(chunks))
        company_name = _infer_company_name(source_file, full_text)
        logger.info("Using company anchor: %s", company_name)

        _write_json(
            run_dir / "chunks.json",
            {
                "source_file": str(source_file),
                "mode": mode,
                "chunk_count": len(chunks),
                "chunks": [
                    {
                        "chunk_index": index,
                        "text": chunk,
                        "character_count": len(chunk),
                        "word_count": len(chunk.split()),
                    }
                    for index, chunk in enumerate(chunks)
                ],
            },
        )

        extractor = LLMExtractor(base_url=args.base_url, api_key=args.api_key, model=args.model)
        logger.info("Starting LLM extraction in %s mode...", mode)
        if args.incremental:
            extractions = extractor.extract_incremental(
                full_text=source_file.read_text(encoding="utf-8"),
                max_iterations=args.max_iterations,
                max_retries=args.max_retries,
            )
        else:
            extractions = []
            memory_state = _empty_memory_state(company_name)
            for chunk_index, chunk in enumerate(tqdm(chunks, desc="Processing Chunks"), start=1):
                logger.info("Extracting chunk %s/%s", chunk_index, len(chunks))
                memory_snapshot = None if args.zero_shot else _memory_snapshot(memory_state)
                extraction = extractor.extract_from_chunk(
                    chunk,
                    company_name=company_name,
                    memory=memory_snapshot,
                    extraction_mode=mode,
                    max_retries=args.max_retries,
                )
                extractions.append(extraction)
                if not args.zero_shot:
                    _update_memory_state(memory_state, extraction)

        extraction_payload = {"extractions": _serialize_extractions(extractions)}
        if not args.zero_shot and not args.incremental:
            extraction_payload["memory_snapshot"] = _memory_snapshot(memory_state)
        _write_json(run_dir / "extractions.json", extraction_payload)

        logger.info("Resolving entities...")
        resolved_triples = resolve_entities(extractions)
        if not resolved_triples:
            raise ExtractionError("Extraction completed but produced zero resolved triples.")

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
            loader = Neo4jLoader(uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
            try:
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
                "resolved_triple_count": len(resolved_triples),
                "loaded_triple_count": loaded_triples,
                "skip_neo4j": args.skip_neo4j,
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
