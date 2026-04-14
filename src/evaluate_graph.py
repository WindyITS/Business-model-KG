import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from entity_resolver import canonical_entity_key

TripleKey = Tuple[str, str, str, str, str]


def _triple_key(triple: dict) -> TripleKey:
    return (
        canonical_entity_key(triple["subject"]),
        triple["subject_type"],
        triple["relation"],
        canonical_entity_key(triple["object"]),
        triple["object_type"],
    )


def _load_triples(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        triples = payload
    else:
        if "triples" in payload:
            triples = payload["triples"]
        elif "resolved_triples" in payload:
            triples = payload["resolved_triples"]
        elif "valid_triples" in payload:
            triples = payload["valid_triples"]
        else:
            triples = []

    return [
        {
            "subject": triple["subject"],
            "subject_type": triple["subject_type"],
            "relation": triple["relation"],
            "object": triple["object"],
            "object_type": triple["object_type"],
        }
        for triple in triples
    ]


def _load_triples_from_json(path: str) -> set[TripleKey]:
    return {_triple_key(triple) for triple in _load_triples(Path(path))}


def evaluate(predicted: set[TripleKey], gold: set[TripleKey]) -> dict:
    true_positives = predicted & gold
    false_positives = predicted - gold
    false_negatives = gold - predicted

    precision = len(true_positives) / len(predicted) if predicted else 0.0
    recall = len(true_positives) / len(gold) if gold else 0.0
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    return {
        "predicted_count": len(predicted),
        "gold_count": len(gold),
        "true_positive_count": len(true_positives),
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _format_triples(triples: Iterable[dict]) -> Sequence[str]:
    return [
        f"[{triple['subject_type']}] {triple['subject']} --{triple['relation']}--> [{triple['object_type']}] {triple['object']}"
        for triple in triples
    ]


def compare_prediction_to_gold(predicted_path: Path, gold_path: Path, show_missing: int, show_extra: int) -> int:
    predicted = _load_triples(predicted_path)
    gold = _load_triples(gold_path)

    predicted_index = {_triple_key(triple): triple for triple in predicted}
    gold_index = {_triple_key(triple): triple for triple in gold}

    true_positives = set(predicted_index) & set(gold_index)
    false_positives = set(predicted_index) - set(gold_index)
    false_negatives = set(gold_index) - set(predicted_index)

    summary = evaluate(set(predicted_index), set(gold_index))
    precision = summary["precision"]
    recall = summary["recall"]
    f1 = summary["f1"]

    print(f"Predicted triples: {len(predicted_index)}")
    print(f"Gold triples:      {len(gold_index)}")
    print(f"True positives:    {len(true_positives)}")
    print(f"False positives:   {len(false_positives)}")
    print(f"False negatives:   {len(false_negatives)}")
    print(f"Precision:         {precision:.3f}")
    print(f"Recall:            {recall:.3f}")
    print(f"F1:                {f1:.3f}")

    if show_missing:
        print("\nMissing gold triples:")
        for line in _format_triples(gold_index[key] for key in list(sorted(false_negatives))[:show_missing]):
            print(f"- {line}")

    if show_extra:
        print("\nExtra predicted triples:")
        for line in _format_triples(predicted_index[key] for key in list(sorted(false_positives))[:show_extra]):
            print(f"- {line}")

    return 0 if not false_positives and not false_negatives else 1


def dump_neo4j(uri: str, user: str, password: str) -> int:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = (
        "MATCH (n)-[r]->(m) "
        "RETURN labels(n)[0] AS sub_type, n.name AS subject, type(r) AS relation, "
        "labels(m)[0] AS obj_type, m.name AS object"
    )

    with driver.session() as session:
        result = session.run(query)
        count = 0
        for record in result:
            print(
                f"[{record['sub_type']}] {record['subject']} --{record['relation']}--> "
                f"[{record['obj_type']}] {record['object']}"
            )
            count += 1
        print(f"\nTotal relationships: {count}")

    driver.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate extracted triples or inspect the Neo4j graph.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser("compare", help="Compare predicted triples to a gold file.")
    compare_parser.add_argument("predicted", type=Path, help="Predicted triples JSON file.")
    compare_parser.add_argument("gold", type=Path, help="Gold triples JSON file.")
    compare_parser.add_argument("--show-missing", type=int, default=10, help="How many missing gold triples to print.")
    compare_parser.add_argument("--show-extra", type=int, default=10, help="How many extra predicted triples to print.")

    dump_parser = subparsers.add_parser("dump-neo4j", help="Print the graph currently stored in Neo4j.")
    dump_parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687")
    dump_parser.add_argument("--neo4j-user", type=str, default="neo4j")
    dump_parser.add_argument("--neo4j-password", type=str, default="password")

    args = parser.parse_args()
    if args.command == "compare":
        return compare_prediction_to_gold(args.predicted, args.gold, args.show_missing, args.show_extra)
    return dump_neo4j(args.neo4j_uri, args.neo4j_user, args.neo4j_password)


if __name__ == "__main__":
    raise SystemExit(main())
