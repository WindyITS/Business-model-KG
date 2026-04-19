# Query Planner Curation Rubric

1. `question` matches the exact semantics of `supervision_target`.
2. No hidden scope, hidden limit, hidden company/place constraint, or inclusive/exclusive mismatch.
3. `local_safe` wording is faithful to runtime semantics.
4. `strong_model_candidate` vs `refuse` boundary is clean.
5. Prompt is grammatical and natural enough for a small local model.
6. No synthetic scaffold patterns that weaken supervision.
7. No contradiction between `question`, `route_label`, `family`, and `gold_rows`.
