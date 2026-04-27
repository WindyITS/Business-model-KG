# Evaluation

This folder contains the benchmark data and evaluation code for the extraction pipelines.

The benchmark is split into two sections:

- `benchmarks/dev/`: development benchmark used while building and debugging the evaluator
- `benchmarks/test/`: final held-out benchmark used for presentation metrics

Each section has two subfolders:

- `raw/`: manually annotated CSV files
- `clean/`: normalized benchmark files produced from raw CSVs

Raw benchmark CSVs should use this five-column schema:

```csv
subject,subject_type,relation,object,object_type
```

Clean benchmark files will use JSONL, with one triple per line:

```json
{"subject":"Microsoft","subject_type":"Company","relation":"HAS_SEGMENT","object":"Intelligent Cloud","object_type":"BusinessSegment"}
```

Convert raw CSV files into clean JSONL files with:

```bash
./venv/bin/python -m evaluation.scripts.prepare_gold --split all
```

Useful variants:

```bash
./venv/bin/python -m evaluation.scripts.prepare_gold --split dev
./venv/bin/python -m evaluation.scripts.prepare_gold --split test
```

The converter writes:

- one `.jsonl` file in the matching `clean/` folder for each raw CSV
- one `manifest.json` per converted split with file and triple counts

The evaluation scripts should compare clean gold triples against pipeline outputs from:

```text
outputs/<company>/<pipeline>/latest/resolved_triples.json
```

Primary score:

- strict normalized typed-triple precision, recall, and F1

Secondary score:

- alias-normalized typed-triple precision, recall, and F1, using only manually approved aliases
