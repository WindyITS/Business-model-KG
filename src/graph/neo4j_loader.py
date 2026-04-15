import logging
from collections import defaultdict
from typing import DefaultDict, List, Tuple

from neo4j import GraphDatabase

from llm_extraction.models import Triple
from ontology.place_hierarchy import PLACE_INCLUDES_PROPERTY, PLACE_WITHIN_PROPERTY, place_query_property_rows

logger = logging.getLogger(__name__)

ALLOWED_NODE_TYPES = {"Company", "BusinessSegment", "Offering", "CustomerType", "Channel", "Place", "RevenueModel"}
SCOPED_NODE_TYPES = {"BusinessSegment", "Offering"}
SCOPED_NODE_COMPANY_PROPERTY = "company_name"
ALLOWED_RELATION_TYPES = {
    "HAS_SEGMENT",
    "OFFERS",
    "SERVES",
    "OPERATES_IN",
    "SELLS_THROUGH",
    "PARTNERS_WITH",
    "MONETIZES_VIA",
}


class Neo4jLoader:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def clear_graph(self) -> None:
        """Delete all nodes and relationships in the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n").consume()
        logger.info("Cleared all nodes and relationships from Neo4j.")

    def setup_constraints(self) -> None:
        """Set up uniqueness constraints, scoping segments and offerings by company."""
        unscoped_node_types = ["Company", "CustomerType", "Channel", "Place", "RevenueModel"]
        with self.driver.session() as session:
            for constraint_name in (
                "BusinessSegment_name",
                "BusinessSegment_name_company",
                "Offering_name",
                "Offering_name_company",
            ):
                try:
                    session.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS").consume()
                except Exception as exc:  # pragma: no cover - exercised only with Neo4j.
                    logger.warning("Could not drop scoped constraint %s: %s", constraint_name, exc)
            for node_type in unscoped_node_types:
                try:
                    query = (
                        f"CREATE CONSTRAINT {node_type}_name IF NOT EXISTS "
                        f"FOR (node:{node_type}) REQUIRE node.name IS UNIQUE"
                    )
                    session.run(query)
                except Exception as exc:  # pragma: no cover - exercised only with Neo4j.
                    logger.warning("Could not create constraint for %s: %s", node_type, exc)
            for node_type in sorted(SCOPED_NODE_TYPES):
                try:
                    query = (
                        f"CREATE CONSTRAINT {node_type}_name_company IF NOT EXISTS "
                        f"FOR (node:{node_type}) REQUIRE (node.{SCOPED_NODE_COMPANY_PROPERTY}, node.name) IS UNIQUE"
                    )
                    session.run(query)
                except Exception as exc:  # pragma: no cover - exercised only with Neo4j.
                    logger.warning("Could not create scoped constraint for %s: %s", node_type, exc)
            logger.info("Checked/created uniqueness constraints.")

    def load_triples(self, triples: List[Triple], company_name: str, batch_size: int = 200) -> int:
        grouped_rows: DefaultDict[Tuple[str, str, str], List[dict]] = defaultdict(list)
        place_names: set[str] = set()
        skipped_triples = 0
        for triple in triples:
            if (
                triple.subject_type not in ALLOWED_NODE_TYPES
                or triple.object_type not in ALLOWED_NODE_TYPES
                or triple.relation not in ALLOWED_RELATION_TYPES
            ):
                skipped_triples += 1
                logger.warning("Skipping invalid triple type combination: %s", triple)
                continue
            grouped_rows[(triple.subject_type, triple.relation, triple.object_type)].append(
                {
                    "subject_name": triple.subject,
                    "subject_company_name": company_name,
                    "object_name": triple.object,
                    "object_company_name": company_name,
                }
            )
            if triple.subject_type == "Place":
                place_names.add(triple.subject)
            if triple.object_type == "Place":
                place_names.add(triple.object)

        total_loaded = 0
        with self.driver.session() as session:
            for (subject_type, relation, object_type), rows in grouped_rows.items():
                subject_merge = _merge_node_clause(
                    alias="subject",
                    node_type=subject_type,
                    name_field="subject_name",
                    company_field="subject_company_name",
                )
                object_merge = _merge_node_clause(
                    alias="object",
                    node_type=object_type,
                    name_field="object_name",
                    company_field="object_company_name",
                )
                query = f"""
                UNWIND $rows AS row
                {subject_merge}
                {object_merge}
                MERGE (subject)-[:{relation}]->(object)
                """
                for start in range(0, len(rows), batch_size):
                    batch = rows[start:start + batch_size]
                    try:
                        session.run(query, rows=batch).consume()
                    except Exception as exc:  # pragma: no cover - exercised only with Neo4j.
                        logger.error(
                            "Failed to load batch for %s-%s-%s starting at index %s: %s",
                            subject_type,
                            relation,
                            object_type,
                            start,
                            exc,
                        )
                        raise
                    total_loaded += len(batch)

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

        if skipped_triples:
            logger.warning("Skipped %s triples due to invalid node/relation types.", skipped_triples)
        logger.info("Successfully loaded %s triples into Neo4j.", total_loaded)
        return total_loaded


def _merge_node_clause(alias: str, node_type: str, name_field: str, company_field: str) -> str:
    if node_type in SCOPED_NODE_TYPES:
        return (
            f"MERGE ({alias}:{node_type} "
            f"{{name: row.{name_field}, {SCOPED_NODE_COMPANY_PROPERTY}: row.{company_field}}})"
        )
    return f"MERGE ({alias}:{node_type} {{name: row.{name_field}}})"
