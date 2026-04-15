# MLX LoRA Fine-Tuning

This repo includes a local Apple Silicon fine-tuning path for the text-to-Cypher dataset using `mlx-lm` and `google/gemma-4-E4B-it`.

The default workflow is:

1. build the local `v3` dataset workspace
2. prepare the `v3` chat dataset as MLX-ready `train.jsonl` / `test.jsonl`
3. run LoRA fine-tuning against `google/gemma-4-E4B-it`
4. evaluate the adapter on the held-out `v3` set with JSON, Cypher, params, and optional Neo4j execution checks

Recommended first run:

1. `./venv/bin/python scripts/build_text2cypher_dataset.py`
2. `./venv/bin/python scripts/prepare_text2cypher_mlx_dataset.py --force`
3. `./venv/bin/python scripts/train_text2cypher_mlx_lora.py`
4. `./venv/bin/python scripts/evaluate_text2cypher_mlx_adapter.py --adapter-path outputs/text2cypher_mlx/gemma4_e4b/adapters --force`

## Install

Keep the runtime dependencies and the training dependencies separate:

```bash
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-mlx-lora.txt
```

The MLX LoRA workflow follows the official `mlx-lm` training path:

- install with `pip install "mlx-lm[train]"`
- train with `mlx_lm.lora`
- evaluate test loss with `mlx_lm.lora --test`
- generate with `mlx_lm.generate`

Reference: [MLX LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)

## Prepare The Dataset

Build the local dataset workspace first, then prepare the MLX-ready chat dataset:

```bash
./venv/bin/python scripts/build_text2cypher_dataset.py
./venv/bin/python scripts/prepare_text2cypher_mlx_dataset.py --force
```

By default this writes:

- `outputs/text2cypher_mlx/gemma4_e4b/dataset/train.jsonl`
- `outputs/text2cypher_mlx/gemma4_e4b/dataset/test.jsonl`

Each row keeps the model-facing `messages` list as the training signal and preserves prompt/completion metadata for later scoring.

## Train Gemma 4 E4B

Launch a first LoRA run:

```bash
./venv/bin/python scripts/train_text2cypher_mlx_lora.py
```

The default training shape is conservative for a laptop:

- model: `google/gemma-4-E4B-it`
- fine-tune type: `lora`
- iterations: `5000`
- batch size: `1`
- gradient accumulation: `4`
- tuned layers: `8`
- prompt masking: enabled
- gradient checkpointing: enabled

Prompt masking is intentional. The dataset is a `chat` corpus and the loss should land on the assistant JSON only, not on the system or user text.

The default adapter output path is `outputs/text2cypher_mlx/gemma4_e4b/adapters`.

For a quick smoke run:

```bash
./venv/bin/python scripts/train_text2cypher_mlx_lora.py --iters 250 --dry-run
```

To run MLX test loss on the held-out set after training:

```bash
./venv/bin/python scripts/train_text2cypher_mlx_lora.py --run-test-loss
```

## Evaluate The Adapter

Run held-out generation and score the predictions:

```bash
./venv/bin/python scripts/evaluate_text2cypher_mlx_adapter.py \
  --adapter-path outputs/text2cypher_mlx/gemma4_e4b/adapters \
  --force
```

This writes:

- `outputs/text2cypher_mlx/gemma4_e4b/evaluation/predictions.jsonl`
- `outputs/text2cypher_mlx/gemma4_e4b/evaluation/summary.json`

The evaluator reports:

- valid JSON rate
- answerable accuracy
- normalized Cypher match
- params exact match
- structured match against the gold JSON contract
- optional execution match against the synthetic Neo4j fixture graph

It also reports grouped summaries by `example_id` and `intent_id`, which matters because the held-out set contains paraphrase expansions of a smaller number of unique held-out cases.

To include execution-based checking:

```bash
./venv/bin/python scripts/evaluate_text2cypher_mlx_adapter.py \
  --adapter-path outputs/text2cypher_mlx/gemma4_e4b/adapters \
  --neo4j-uri http://localhost:7474 \
  --force
```

## Thinking Mode

The training pipeline defaults to `enable_thinking=False` at generation time during evaluation.

That is deliberate:

- the supervised target is strict JSON only
- the dataset never supervises an intermediate reasoning block
- keeping thinking off makes the generated shape line up with the target contract

If you want to experiment with inference-time reasoning anyway, the evaluator supports:

```bash
./venv/bin/python scripts/evaluate_text2cypher_mlx_adapter.py \
  --adapter-path outputs/text2cypher_mlx/gemma4_e4b/adapters \
  --enable-thinking \
  --force
```

The evaluator will still try to recover the final JSON object from a longer response, but the default recommendation is to keep thinking disabled for the baseline run.
