import logging
from collections import defaultdict
from typing import DefaultDict, List, Tuple

from neo4j import GraphDatabase

from llm_extractor import Triple

logger = logging.getLogger(__name__)

ALLOWED_NODE_TYPES = {"Company", "BusinessSegment", "Offering", "CustomerType", "Channel", "Place", "RevenueModel"}
COMPANY_SCOPED_NODE_TYPES = {"BusinessSegment", "Offering"}
SCOPED_NODE_KEY = Tuple[str, str]
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

    @staticmethod
    def _merge_pattern(alias: str, node_type: str) -> str:
        if node_type in COMPANY_SCOPED_NODE_TYPES:
            return f"{alias}:{node_type} {{name: row.{alias}_name, company: row.{alias}_company}}"
        return f"{alias}:{node_type} {{name: row.{alias}_name}}"

    @staticmethod
    def _set_scope(
        scopes: DefaultDict[SCOPED_NODE_KEY, set[str]],
        node_type: str,
        node_name: str,
        company_name: str,
    ) -> bool:
        key = (node_type, node_name)
        prior_size = len(scopes[key])
        scopes[key].add(company_name)
        return len(scopes[key]) != prior_size

    @classmethod
    def _infer_scoped_node_companies(cls, triples: List[Triple]) -> dict[SCOPED_NODE_KEY, str]:
        scopes: DefaultDict[SCOPED_NODE_KEY, set[str]] = defaultdict(set)
        seen_scoped_nodes: set[SCOPED_NODE_KEY] = set()
        reporting_companies = {triple.subject for triple in triples if triple.subject_type == "Company"}

        for triple in triples:
            if triple.subject_type in COMPANY_SCOPED_NODE_TYPES:
                seen_scoped_nodes.add((triple.subject_type, triple.subject))
            if triple.object_type in COMPANY_SCOPED_NODE_TYPES:
                seen_scoped_nodes.add((triple.object_type, triple.object))

        changed = True
        while changed:
            changed = False
            for triple in triples:
                if (
                    triple.relation == "HAS_SEGMENT"
                    and triple.subject_type == "Company"
                    and triple.object_type == "BusinessSegment"
                ):
                    changed |= cls._set_scope(scopes, "BusinessSegment", triple.object, triple.subject)
                    continue

                if triple.relation != "OFFERS" or triple.object_type != "Offering":
                    continue

                if triple.subject_type == "Company":
                    changed |= cls._set_scope(scopes, "Offering", triple.object, triple.subject)
                    continue

                if triple.subject_type not in COMPANY_SCOPED_NODE_TYPES:
                    continue

                for company_name in scopes.get((triple.subject_type, triple.subject), set()):
                    changed |= cls._set_scope(scopes, "Offering", triple.object, company_name)

        default_company = next(iter(reporting_companies)) if len(reporting_companies) == 1 else None
        resolved_scopes: dict[SCOPED_NODE_KEY, str] = {}
        unresolved_nodes: list[str] = []
        ambiguous_nodes: list[str] = []

        for node_key in sorted(seen_scoped_nodes):
            companies = set(scopes.get(node_key, set()))
            if not companies and default_company is not None:
                companies = {default_company}

            if len(companies) == 1:
                resolved_scopes[node_key] = next(iter(companies))
            elif not companies:
                unresolved_nodes.append(f"{node_key[0]}:{node_key[1]}")
            else:
                ambiguous_nodes.append(f"{node_key[0]}:{node_key[1]} -> {sorted(companies)}")

        if unresolved_nodes or ambiguous_nodes:
            details = []
            if unresolved_nodes:
                details.append(f"unresolved scoped nodes: {', '.join(unresolved_nodes)}")
            if ambiguous_nodes:
                details.append(f"ambiguous scoped nodes: {', '.join(ambiguous_nodes)}")
            raise ValueError("Could not derive company scope from triples; " + "; ".join(details))

        return resolved_scopes

    def setup_constraints(self) -> None:
        """Set up uniqueness constraints on the `name` property for all node types."""
        name_only_node_types = ["Company", "CustomerType", "Channel", "Place", "RevenueModel"]
        with self.driver.session() as session:
            for node_type in name_only_node_types:
                try:
                    query = (
                        f"CREATE CONSTRAINT {node_type}_name IF NOT EXISTS "
                        f"FOR (node:{node_type}) REQUIRE node.name IS UNIQUE"
                    )
                    session.run(query)
                except Exception as exc:  # pragma: no cover - exercised only with Neo4j.
                    logger.warning("Could not create constraint for %s: %s", node_type, exc)
            for node_type in sorted(COMPANY_SCOPED_NODE_TYPES):
                legacy_constraint_name = f"{node_type}_name"
                scoped_constraint_name = f"{node_type}_name_company"
                try:
                    session.run(f"DROP CONSTRAINT {legacy_constraint_name} IF EXISTS").consume()
                    query = (
                        f"CREATE CONSTRAINT {scoped_constraint_name} IF NOT EXISTS "
                        f"FOR (node:{node_type}) REQUIRE (node.name, node.company) IS UNIQUE"
                    )
                    session.run(query)
                except Exception as exc:  # pragma: no cover - exercised only with Neo4j.
                    logger.warning("Could not create company-scoped constraint for %s: %s", node_type, exc)
            logger.info("Checked/created uniqueness constraints.")

    def load_triples(self, triples: List[Triple], batch_size: int = 200) -> int:
        node_scopes = self._infer_scoped_node_companies(triples)
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
                    **(
                        {"subject_company": node_scopes[(triple.subject_type, triple.subject)]}
                        if triple.subject_type in COMPANY_SCOPED_NODE_TYPES
                        else {}
                    ),
                    **(
                        {"object_company": node_scopes[(triple.object_type, triple.object)]}
                        if triple.object_type in COMPANY_SCOPED_NODE_TYPES
                        else {}
                    ),
                }
            )

        total_loaded = 0
        with self.driver.session() as session:
            for (subject_type, relation, object_type), rows in grouped_rows.items():
                subject_merge_pattern = self._merge_pattern("subject", subject_type)
                object_merge_pattern = self._merge_pattern("object", object_type)
                query = f"""
                UNWIND $rows AS row
                MERGE ({subject_merge_pattern})
                MERGE ({object_merge_pattern})
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
