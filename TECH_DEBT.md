# Technical Debt

## Compatibility Wrapper Cleanup

Status: deferred

Context:
The repo currently keeps several top-level modules in `src/` that re-export implementations from the package-based layout under `src/runtime`, `src/graph`, `src/ontology`, and `src/llm`.

Examples:
- `src/main.py`
- `src/llm_extractor.py`
- `src/entity_resolver.py`
- `src/model_provider.py`
- `src/neo4j_loader.py`
- `src/evaluate_graph.py`
- `src/ontology_config.py`
- `src/ontology_validator.py`
- `src/place_hierarchy.py`
- `src/query_cypher.py`

Why this likely exists:
These files appear to preserve older import paths and entry-point behavior after the project was reorganized into clearer package directories.

Why revisit:
The compatibility layer adds cognitive overhead when reading the codebase and makes the source layout look more duplicated than it really is.

Suggested future cleanup:
- Identify whether any scripts, tests, notebooks, or external users still rely on the top-level import paths.
- Confirm which wrappers are still needed for packaging or CLI compatibility.
- Update imports and packaging metadata in `pyproject.toml` if the wrappers are no longer required.
- Remove the wrapper modules once compatibility is no longer needed.

Guardrail:
Do not remove these files until compatibility usage has been checked explicitly.
