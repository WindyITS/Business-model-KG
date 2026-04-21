# Analyst Pipeline Prompt Assets

This folder contains the prompt files for the analyst-style extraction pipeline.

The analyst pipeline is intentionally separate from the zero-shot baseline:

- stage 1A builds the foundational analyst memo as structured plain text
- stage 1B augments that memo with defensible detail, still as structured plain text
- stage 2 compiles the memo into ontology-valid graph output
- a short critique pass removes overreach and weak structure
