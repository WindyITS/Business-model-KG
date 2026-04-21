# Technical Debt

## Compatibility Wrapper Cleanup

Status: completed

Resolution:
- Internal callers and package entry points were migrated to the package-based module surface.
- The top-level compatibility shim modules in `src/` were removed.
- Packaging metadata no longer exports the legacy top-level module names.

Canonical surface:
- `runtime.*`
- `graph.*`
- `ontology.*`
- `llm.*`

Validation:
- Repo-wide internal usage was checked before removal.
- Repo/package smoke tests now assert the old top-level module names are no longer packaged.
