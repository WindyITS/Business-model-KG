import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Tuple
from urllib.parse import urlparse

from neo4j import GraphDatabase
from place_hierarchy import PLACE_INCLUDES_PROPERTY, PLACE_WITHIN_PROPERTY, place_query_property_rows


ALLOWED_NODE_LABELS = {
    "Company",
    "BusinessSegment",
    "Offering",
    "CustomerType",
    "Channel",
    "Place",
    "RevenueModel",
}

ALLOWED_RELATION_TYPES = {
    "HAS_SEGMENT",
    "OFFERS",
    "SERVES",
    "OPERATES_IN",
    "SELLS_THROUGH",
    "PARTNERS_WITH",
    "MONETIZES_VIA",
}

SCOPED_NODE_LABELS = {"BusinessSegment", "Offering"}
FIXTURE_NODE_RESERVED_KEYS = {"node_id", "label", "name", "properties"}

DISALLOWED_CLAUSE_PATTERNS = (
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bDELETE\b",
    r"\bDETACH\b",
    r"\bSET\b",
    r"\bREMOVE\b",
    r"\bCALL\b",
)

PARAM_PATTERN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
BROWSER_PORT_TO_BOLT_PORT = {
    7473: 7687,
    7474: 7687,
}

SINGLE_STRING_INTENTS = {
    "qf23_offering_segment_anchor",
    "qf23_offering_root_anchor",
    "qf24_offering_parent_list",
    "qf24_offering_root_ancestor",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return rows


def normalize_neo4j_uri(uri: str | None) -> str:
    raw = (
        (uri or "").strip()
        or os.getenv("NEO4J_URI", "").strip()
        or os.getenv("NEO4J_URL", "").strip()
        or DEFAULT_NEO4J_URI
    )

    if "://" not in raw:
        host, separator, port_text = raw.rpartition(":")
        if separator and host and port_text.isdigit():
            port = int(port_text)
            bolt_port = BROWSER_PORT_TO_BOLT_PORT.get(port, port)
            return f"bolt://{host}:{bolt_port}"
        return f"bolt://{raw}"

    parsed = urlparse(raw)
    if parsed.scheme in {"bolt", "neo4j", "bolt+s", "bolt+ssc", "neo4j+s", "neo4j+ssc"}:
        return raw

    if parsed.scheme in {"http", "https"}:
        if not parsed.hostname:
            raise ValueError(f"Neo4j URI {raw!r} is missing a hostname")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        bolt_port = BROWSER_PORT_TO_BOLT_PORT.get(port, port)
        return f"bolt://{parsed.hostname}:{bolt_port}"

    raise ValueError(
        "Unsupported Neo4j URI scheme. Use bolt/neo4j URIs directly, or pass a browser URL "
        "such as http://localhost:7474 and it will be normalized."
    )


def validate_fixture_shape(fixture: Dict[str, Any]) -> None:
    node_lookup = {}
    node_names: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node in fixture["nodes"]:
        if not isinstance(node, dict):
            raise ValueError(f"Fixture graph {fixture['graph_id']} contains a non-object node entry")
        for required_key in ("node_id", "label", "name"):
            if required_key not in node:
                raise ValueError(
                    f"Fixture graph {fixture['graph_id']} contains a node missing {required_key!r}"
                )
        properties = node.get("properties")
        if properties is not None and not isinstance(properties, dict):
            raise ValueError(
                f"Fixture graph {fixture['graph_id']} node {node['node_id']!r} has non-object properties"
            )
        label = node["label"]
        if label not in ALLOWED_NODE_LABELS:
            raise ValueError(f"Unsupported node label {label!r} in graph {fixture['graph_id']}")
        node_lookup[node["node_id"]] = node
        node_names[node["name"]].append(node)

    for edge in fixture["edges"]:
        if edge["type"] not in ALLOWED_RELATION_TYPES:
            raise ValueError(f"Unsupported relation type {edge['type']!r} in graph {fixture['graph_id']}")
        SyntheticGraphLoader._resolve_edge_endpoint(edge["from"], node_lookup, node_names, fixture["graph_id"])
        SyntheticGraphLoader._resolve_edge_endpoint(edge["to"], node_lookup, node_names, fixture["graph_id"])


def validate_read_only_cypher(cypher: str) -> List[str]:
    failures = []
    for pattern in DISALLOWED_CLAUSE_PATTERNS:
        if re.search(pattern, cypher, flags=re.IGNORECASE):
            failures.append(f"Query contains disallowed clause matching {pattern}")
    return failures


def validate_params_match(cypher: str, params: Dict[str, Any]) -> List[str]:
    failures = []
    referenced = sorted(set(PARAM_PATTERN.findall(cypher)))
    provided = sorted(params.keys())
    if referenced != provided:
        failures.append(
            f"Parameter mismatch. Referenced={referenced}, provided={provided}"
        )
    return failures


def value_matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return False


def _extract_node_properties(node: Dict[str, Any]) -> Dict[str, Any]:
    properties: Dict[str, Any] = {"node_id": node["node_id"], "name": node["name"]}

    nested_properties = node.get("properties")
    if isinstance(nested_properties, dict):
        properties.update(nested_properties)

    for key, value in node.items():
        if key not in FIXTURE_NODE_RESERVED_KEYS:
            properties[key] = value

    return properties


def _is_scoped_label(node_label: str) -> bool:
    return node_label in SCOPED_NODE_LABELS


def validate_result(
    intent_id: str,
    result_shape: List[Dict[str, Any]],
    columns: List[str],
    records: List[Dict[str, Any]],
) -> List[str]:
    failures = []
    expected_columns = [column["column"] for column in result_shape]
    if columns != expected_columns:
        failures.append(f"Column mismatch. Expected {expected_columns}, got {columns}")
        return failures

    if len(result_shape) == 1:
        expected = result_shape[0]
        column_name = expected["column"]
        expected_type = expected["type"]
        if expected_type == "boolean":
            if len(records) != 1:
                failures.append(f"Expected exactly 1 boolean row, got {len(records)}")
            elif records[0].get(column_name) is not True:
                failures.append(f"Expected boolean result True, got {records[0].get(column_name)!r}")
            return failures

        if expected_type == "integer":
            if len(records) != 1:
                failures.append(f"Expected exactly 1 integer row, got {len(records)}")
            else:
                value = records[0].get(column_name)
                if not value_matches_type(value, expected_type):
                    failures.append(f"Expected integer result for {column_name}, got {value!r}")
                elif value < 1:
                    failures.append(f"Expected positive integer result for {column_name}, got {value!r}")
            return failures

        if intent_id in SINGLE_STRING_INTENTS:
            if len(records) != 1:
                failures.append(f"Expected exactly 1 string row, got {len(records)}")

    if not records:
        failures.append("Expected at least one result row, got zero")
        return failures

    for record in records:
        for column in result_shape:
            column_name = column["column"]
            expected_type = column["type"]
            value = record.get(column_name)
            if value is None:
                failures.append(f"Column {column_name} returned null")
                continue
            if not value_matches_type(value, expected_type):
                failures.append(
                    f"Column {column_name} expected {expected_type}, got {type(value).__name__}"
                )
    return failures


class SyntheticGraphLoader:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(normalize_neo4j_uri(uri), auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def clear_graph(self) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n").consume()

    def setup_constraints(self) -> None:
        with self.driver.session() as session:
            for label in sorted(ALLOWED_NODE_LABELS):
                if _is_scoped_label(label):
                    session.run(f"DROP CONSTRAINT {label}_name IF EXISTS").consume()
                    session.run(
                        f"CREATE CONSTRAINT {label}_name_company IF NOT EXISTS "
                        f"FOR (node:{label}) REQUIRE (node.company_name, node.name) IS UNIQUE"
                    ).consume()
                else:
                    session.run(
                        f"CREATE CONSTRAINT {label}_name IF NOT EXISTS "
                        f"FOR (node:{label}) REQUIRE node.name IS UNIQUE"
                    ).consume()

    def load_graph(self, fixture: Dict[str, Any]) -> None:
        validate_fixture_shape(fixture)
        nodes_by_label: DefaultDict[Tuple[str, bool], List[Dict[str, Any]]] = defaultdict(list)
        node_lookup = {node["node_id"]: node for node in fixture["nodes"]}
        node_names: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        place_names: set[str] = set()

        for node in fixture["nodes"]:
            node_properties = _extract_node_properties(node)
            row = {
                "name": node["name"],
                "properties": node_properties,
                "node_id": node["node_id"],
            }
            has_company_name = _is_scoped_label(node["label"]) and "company_name" in node_properties
            if has_company_name:
                row["company_name"] = node_properties["company_name"]
            nodes_by_label[(node["label"], has_company_name)].append(row)
            node_names[node["name"]].append(node)
            if node["label"] == "Place":
                place_names.add(node["name"])

        edges_by_signature: DefaultDict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
        for edge in fixture["edges"]:
            source = self._resolve_edge_endpoint(edge["from"], node_lookup, node_names, fixture["graph_id"])
            target = self._resolve_edge_endpoint(edge["to"], node_lookup, node_names, fixture["graph_id"])
            edges_by_signature[(source["label"], edge["type"], target["label"])].append(
                {
                    "from_node_id": source["node_id"],
                    "to_node_id": target["node_id"],
                }
            )

        with self.driver.session() as session:
            for (label, has_company_name), rows in nodes_by_label.items():
                if _is_scoped_label(label) and has_company_name:
                    session.run(
                        f"""
                        UNWIND $rows AS row
                        MERGE (n:{label} {{name: row.name, company_name: row.company_name}})
                        SET n += row.properties
                        """,
                        rows=rows,
                    ).consume()
                else:
                    session.run(
                        f"""
                        UNWIND $rows AS row
                        MERGE (n:{label} {{name: row.name}})
                        SET n += row.properties
                        """,
                        rows=rows,
                    ).consume()

            for (from_label, rel_type, to_label), rows in edges_by_signature.items():
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (source:{from_label} {{node_id: row.from_node_id}})
                    MATCH (target:{to_label} {{node_id: row.to_node_id}})
                    MERGE (source)-[:{rel_type}]->(target)
                    """,
                    rows=rows,
                ).consume()

            if place_names:
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (place:Place {{name: row.name}})
                    SET place.{PLACE_WITHIN_PROPERTY} = row.{PLACE_WITHIN_PROPERTY},
                        place.{PLACE_INCLUDES_PROPERTY} = row.{PLACE_INCLUDES_PROPERTY}
                    """,
                    rows=list(place_query_property_rows(place_names)),
                ).consume()

    @staticmethod
    def _resolve_edge_endpoint(
        reference: str,
        node_lookup: Dict[str, Dict[str, Any]],
        node_names: Dict[str, List[Dict[str, Any]]],
        graph_id: str,
    ) -> Dict[str, Any]:
        if reference in node_lookup:
            return node_lookup[reference]

        candidates = node_names.get(reference, [])
        if len(candidates) == 1:
            return candidates[0]

        if not candidates:
            raise ValueError(f"Unknown edge endpoint {reference!r} in graph {graph_id}")

        raise ValueError(f"Ambiguous edge endpoint {reference!r} in graph {graph_id}; use node_id references")

    def run_query(self, cypher: str, params: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            columns = list(result.keys())
            records = [record.data() for record in result]
        return columns, records


def validate_example(
    loader: SyntheticGraphLoader,
    example: Dict[str, Any],
) -> Dict[str, Any]:
    failures: List[str] = []
    if not example["answerable"]:
        if example["gold_cypher"] is not None:
            failures.append("Refusal row should have gold_cypher=null")
        if example["result_shape"] is not None:
            failures.append("Refusal row should have result_shape=null")
        return {
            "example_id": example["example_id"],
            "graph_id": example["graph_id"],
            "passed": not failures,
            "failures": failures,
            "row_count": 0,
            "columns": [],
        }

    cypher = example["gold_cypher"]
    params = example["params"]
    result_shape = example["result_shape"]

    failures.extend(validate_read_only_cypher(cypher))
    failures.extend(validate_params_match(cypher, params))

    if not failures:
        try:
            columns, records = loader.run_query(cypher, params)
        except Exception as exc:  # pragma: no cover - exercised only with Neo4j.
            failures.append(f"Execution failed: {exc}")
            columns = []
            records = []
        else:
            failures.extend(validate_result(example["intent_id"], result_shape, columns, records))
    else:
        columns = []
        records = []

    return {
        "example_id": example["example_id"],
        "graph_id": example["graph_id"],
        "passed": not failures,
        "failures": failures,
        "row_count": len(records),
        "columns": columns,
    }


def build_report(
    fixture_rows: Iterable[Dict[str, Any]],
    example_rows: Iterable[Dict[str, Any]],
    loader: SyntheticGraphLoader,
) -> Dict[str, Any]:
    fixtures = list(fixture_rows)
    examples = list(example_rows)
    fixture_by_graph_id = {fixture["graph_id"]: fixture for fixture in fixtures}
    examples_by_graph_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    refusal_rows: List[Dict[str, Any]] = []

    for example in examples:
        graph_id = example["graph_id"]
        if example["answerable"]:
            if not graph_id:
                raise ValueError(f"Answerable row {example['example_id']} is missing graph_id")
            if graph_id not in fixture_by_graph_id:
                raise ValueError(f"Unknown graph_id {graph_id!r} for example {example['example_id']}")
            examples_by_graph_id[graph_id].append(example)
        else:
            refusal_rows.append(example)

    per_example_results = []
    for refusal in refusal_rows:
        per_example_results.append(validate_example(loader, refusal))

    for graph_id, graph_examples in examples_by_graph_id.items():
        loader.clear_graph()
        loader.load_graph(fixture_by_graph_id[graph_id])
        for example in graph_examples:
            per_example_results.append(validate_example(loader, example))

    failed_results = [result for result in per_example_results if not result["passed"]]
    return {
        "summary": {
            "fixtures_loaded": len(fixtures),
            "examples_checked": len(examples),
            "passed": len(per_example_results) - len(failed_results),
            "failed": len(failed_results),
        },
        "failures": failed_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate text-to-Cypher bound seed examples against synthetic Neo4j fixtures."
    )
    parser.add_argument(
        "--fixtures-path",
        type=Path,
        default=Path("datasets/text2cypher/v3/source/fixture_instances.jsonl"),
    )
    parser.add_argument(
        "--examples-path",
        type=Path,
        default=Path("datasets/text2cypher/v3/source/bound_seed_examples.jsonl"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("datasets/text2cypher/v3/reports/bound_seed_validation_report.json"),
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL") or DEFAULT_NEO4J_URI,
    )
    parser.add_argument("--neo4j-user", type=str, default="neo4j")
    parser.add_argument("--neo4j-password", type=str, default="password")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixtures = load_jsonl(args.fixtures_path)
    examples = load_jsonl(args.examples_path)

    neo4j_uri = normalize_neo4j_uri(args.neo4j_uri)
    loader = SyntheticGraphLoader(neo4j_uri, args.neo4j_user, args.neo4j_password)
    try:
        loader.setup_constraints()
        report = build_report(fixtures, examples, loader)
    finally:
        loader.close()

    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = report["summary"]
    print(
        json.dumps(
            {
                "fixtures_loaded": summary["fixtures_loaded"],
                "examples_checked": summary["examples_checked"],
                "passed": summary["passed"],
                "failed": summary["failed"],
                "neo4j_uri": neo4j_uri,
                "report_path": str(args.report_path),
            },
            indent=2,
        )
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
