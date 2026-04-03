import logging
from collections import defaultdict
from typing import DefaultDict, List, Tuple

from neo4j import GraphDatabase

from llm_extractor import Triple

logger = logging.getLogger(__name__)

ALLOWED_NODE_TYPES = {"Company", "BusinessSegment", "Offering", "CustomerType", "Channel", "Place", "RevenueModel"}
ALLOWED_RELATION_TYPES = {
    "HAS_SEGMENT",
    "OFFERS",
    "SERVES",
    "OPERATES_IN",
    "SELLS_THROUGH",
    "PARTNERS_WITH",
    "SUPPLIED_BY",
    "MONETIZES_VIA",
    "PART_OF",
}


class Neo4jLoader:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def setup_constraints(self) -> None:
        """Set up uniqueness constraints on the `name` property for all node types."""
        node_types = ["Company", "BusinessSegment", "Offering", "CustomerType", "Channel", "Place", "RevenueModel"]
        with self.driver.session() as session:
            for node_type in node_types:
                try:
                    query = (
                        f"CREATE CONSTRAINT {node_type}_name IF NOT EXISTS "
                        f"FOR (node:{node_type}) REQUIRE node.name IS UNIQUE"
                    )
                    session.run(query)
                except Exception as exc:  # pragma: no cover - exercised only with Neo4j.
                    logger.warning("Could not create constraint for %s: %s", node_type, exc)
            logger.info("Checked/created uniqueness constraints.")

    def load_triples(self, triples: List[Triple], batch_size: int = 200) -> int:
        grouped_rows: DefaultDict[Tuple[str, str, str], List[dict]] = defaultdict(list)
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
                    "object_name": triple.object,
                }
            )

        total_loaded = 0
        with self.driver.session() as session:
            for (subject_type, relation, object_type), rows in grouped_rows.items():
                query = f"""
                UNWIND $rows AS row
                MERGE (subject:{subject_type} {{name: row.subject_name}})
                MERGE (object:{object_type} {{name: row.object_name}})
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

        if skipped_triples:
            logger.warning("Skipped %s triples due to invalid node/relation types.", skipped_triples)
        logger.info("Successfully loaded %s triples into Neo4j.", total_loaded)
        return total_loaded
