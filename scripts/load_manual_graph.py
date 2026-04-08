import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_extractor import Triple
from neo4j_loader import Neo4jLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Load a manual JSON graph into Neo4j.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/manual/manual_opus.json",
        help="Path to input JSON file.",
    )
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", type=str, default="neo4j")
    parser.add_argument("--neo4j-password", type=str, default="password")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("File %s not found.", args.input)
        return 1

    try:
        triples_data = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s as JSON: %s", args.input, exc)
        return 1

    mapped_triples = []
    for t in triples_data.get("triples", []):
        mapped_triples.append(Triple(
            subject=t["subject"],
            subject_type=t["subject_type"],
            relation=t.get("relation") or t.get("predicate"),
            object=t["object"],
            object_type=t["object_type"]
        ))

    logger.info("Parsed %s triples from %s", len(mapped_triples), args.input)

    loader = Neo4jLoader(uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
    try:
        loader.clear_graph()
        loader.setup_constraints()
        loaded_count = loader.load_triples(mapped_triples)
        logger.info("Inserted %s triples into Neo4j.", loaded_count)
    finally:
        loader.close()

    # Save validation and result to 'outputs/'
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir) / f"manual_load_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "success",
        "source_file": str(input_path),
        "triples_found": len(mapped_triples),
        "triples_loaded_to_neo4j": loaded_count,
        "timestamp": timestamp,
    }

    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Also dump the raw processed triples back for convenience
    (run_dir / "resolved_triples.json").write_text(
        json.dumps({"triples": [t.model_dump() for t in mapped_triples]}, indent=2),
        encoding="utf-8",
    )

    logger.info("Wrote execution summary to %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
