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
UNLOAD_COMPANY_RELATION_TYPES = ("HAS_SEGMENT", "OFFERS", "OPERATES_IN", "PARTNERS_WITH")
UNLOAD_ORPHAN_PRUNE_LABELS = ("Company", "CustomerType", "Channel", "Place", "RevenueModel")


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

    def graph_counts(self) -> dict[str, int]:
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS node_count").single()["node_count"] or 0
            relationship_count = (
                session.run("MATCH ()-[r]->() RETURN count(r) AS relationship_count").single()["relationship_count"] or 0
            )
        return {
            "node_count": int(node_count),
            "relationship_count": int(relationship_count),
        }

    def list_loaded_companies(self) -> list[str]:
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (company:Company)
                RETURN DISTINCT company.name AS company_name
                UNION
                MATCH (node)
                WHERE (node:BusinessSegment OR node:Offering)
                  AND node.company_name IS NOT NULL
                  AND trim(node.company_name) <> ""
                RETURN DISTINCT node.company_name AS company_name
                """
            ).data()
        company_names = sorted(
            {
                row["company_name"].strip()
                for row in rows
                if isinstance(row.get("company_name"), str) and row["company_name"].strip()
            }
        )
        return company_names

    def company_graph_counts(self, company_name: str) -> dict[str, int]:
        with self.driver.session() as session:
            counts = session.run(
                """
                OPTIONAL MATCH (company:Company {name: $company_name})
                WITH count(company) AS company_node_count
                OPTIONAL MATCH (scoped)
                WHERE (scoped:BusinessSegment OR scoped:Offering) AND scoped.company_name = $company_name
                WITH company_node_count, count(scoped) AS scoped_node_count
                OPTIONAL MATCH (root)
                WHERE (root:Company AND root.name = $company_name)
                   OR ((root:BusinessSegment OR root:Offering) AND root.company_name = $company_name)
                OPTIONAL MATCH (root)-[rel]-()
                RETURN
                    company_node_count,
                    scoped_node_count,
                    count(DISTINCT rel) AS relationship_count
                """,
                company_name=company_name,
            ).single()
        company_node_count = counts["company_node_count"] or 0
        scoped_node_count = counts["scoped_node_count"] or 0
        relationship_count = counts["relationship_count"] or 0
        return {
            "company_node_count": int(company_node_count),
            "scoped_node_count": int(scoped_node_count),
            "relationship_count": int(relationship_count),
        }

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

    def unload_company(self, company_name: str) -> dict[str, int | str]:
        """Remove one company's graph footprint while preserving unrelated shared graph state."""
        candidate_rows: list[dict[str, str]] = []
        with self.driver.session() as session:
            candidate_rows = session.run(
                """
                MATCH (root)
                WHERE (root:Company AND root.name = $company_name)
                   OR ((root:BusinessSegment OR root:Offering) AND root.company_name = $company_name)
                OPTIONAL MATCH (root)--(neighbor)
                WHERE neighbor IS NOT NULL
                  AND NOT (
                    ((neighbor:BusinessSegment OR neighbor:Offering) AND neighbor.company_name = $company_name)
                    OR (neighbor:Company AND neighbor.name = $company_name)
                  )
                WITH DISTINCT neighbor
                RETURN labels(neighbor) AS labels, neighbor.name AS name
                """,
                company_name=company_name,
            ).data()

            scoped_relationships_deleted = (
                session.run(
                    """
                    MATCH (node)
                    WHERE (node:BusinessSegment OR node:Offering) AND node.company_name = $company_name
                    OPTIONAL MATCH (node)-[rel]-()
                    RETURN count(DISTINCT rel) AS relationship_count
                    """,
                    company_name=company_name,
                ).single()["relationship_count"]
                or 0
            )
            scoped_nodes_deleted = (
                session.run(
                    """
                    MATCH (node)
                    WHERE (node:BusinessSegment OR node:Offering) AND node.company_name = $company_name
                    RETURN count(node) AS node_count
                    """,
                    company_name=company_name,
                ).single()["node_count"]
                or 0
            )
            session.run(
                """
                MATCH (node)
                WHERE (node:BusinessSegment OR node:Offering) AND node.company_name = $company_name
                DETACH DELETE node
                """,
                company_name=company_name,
            ).consume()

            company_relationships_deleted = (
                session.run(
                    """
                    MATCH (:Company {name: $company_name})-[rel]->()
                    WHERE type(rel) IN $relation_types
                    RETURN count(rel) AS relationship_count
                    """,
                    company_name=company_name,
                    relation_types=list(UNLOAD_COMPANY_RELATION_TYPES),
                ).single()["relationship_count"]
                or 0
            )
            session.run(
                """
                MATCH (:Company {name: $company_name})-[rel]->()
                WHERE type(rel) IN $relation_types
                DELETE rel
                """,
                company_name=company_name,
                relation_types=list(UNLOAD_COMPANY_RELATION_TYPES),
            ).consume()

            company_node_deleted = (
                session.run(
                    """
                    MATCH (company:Company {name: $company_name})
                    WHERE NOT (company)--()
                    WITH collect(company) AS companies
                    FOREACH (company IN companies | DELETE company)
                    RETURN size(companies) AS deleted_count
                    """,
                    company_name=company_name,
                ).single()["deleted_count"]
                or 0
            )

            orphan_candidates = _orphan_prune_candidates(candidate_rows)
            orphan_nodes_deleted = 0
            if orphan_candidates:
                orphan_nodes_deleted = (
                    session.run(
                        """
                        UNWIND $candidates AS candidate
                        MATCH (node {name: candidate.name})
                        WHERE candidate.label IN labels(node)
                          AND NOT (node)--()
                        WITH collect(DISTINCT node) AS nodes
                        FOREACH (node IN nodes | DELETE node)
                        RETURN size(nodes) AS deleted_count
                        """,
                        candidates=orphan_candidates,
                    ).single()["deleted_count"]
                    or 0
                )

        summary = {
            "company_name": company_name,
            "scoped_nodes_deleted": scoped_nodes_deleted,
            "scoped_relationships_deleted": scoped_relationships_deleted,
            "company_relationships_deleted": company_relationships_deleted,
            "company_node_deleted": company_node_deleted,
            "orphan_nodes_deleted": orphan_nodes_deleted,
        }
        logger.info("Unloaded Neo4j footprint for %s: %s", company_name, summary)
        return summary


def _merge_node_clause(alias: str, node_type: str, name_field: str, company_field: str) -> str:
    if node_type in SCOPED_NODE_TYPES:
        return (
            f"MERGE ({alias}:{node_type} "
            f"{{name: row.{name_field}, {SCOPED_NODE_COMPANY_PROPERTY}: row.{company_field}}})"
        )
    return f"MERGE ({alias}:{node_type} {{name: row.{name_field}}})"


def _orphan_prune_candidates(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        for label in row.get("labels", []):
            if label not in UNLOAD_ORPHAN_PRUNE_LABELS:
                continue
            key = (label, name)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"label": label, "name": name})
    return candidates
