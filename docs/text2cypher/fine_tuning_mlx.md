# MLX LoRA Fine-Tuning

This repo includes a local Apple Silicon fine-tuning path for the text-to-Cypher dataset using `mlx-lm` and `Qwen/Qwen3-4B`.

The default workflow is:

1. build the local `v3` dataset workspace
2. prepare the `v3` chat dataset as MLX-ready `train.jsonl` / `valid.jsonl` / `test.jsonl`
3. run LoRA fine-tuning against `Qwen/Qwen3-4B`
4. evaluate the adapter on the held-out `v3` set with JSON, Cypher, params, and optional Neo4j execution checks

Recommended first run:

1. `./venv/bin/text2cypher-build`
2. `./venv/bin/text2cypher-prepare-mlx --force`
3. `./venv/bin/text2cypher-train-mlx`
4. `./venv/bin/text2cypher-evaluate-mlx --adapter-path outputs/text2cypher_mlx/qwen3_4b/adapters --force`

## Install

Keep the runtime dependencies and the training dependencies separate:

```bash
source venv/bin/activate
pip install -e .
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
./venv/bin/text2cypher-build
./venv/bin/text2cypher-prepare-mlx --force
```

By default this writes:

- `outputs/text2cypher_mlx/qwen3_4b/dataset/train.jsonl`
- `outputs/text2cypher_mlx/qwen3_4b/dataset/valid.jsonl`
- `outputs/text2cypher_mlx/qwen3_4b/dataset/test.jsonl`

Each row keeps the model-facing `messages` list as the training signal and preserves prompt/completion metadata for later scoring.

The current local split shape is `4904` train rows, `100` validation rows, and `512` held-out test rows.

## Train Qwen 3 4B

Launch a first LoRA run:

```bash
./venv/bin/text2cypher-train-mlx
```

The default training shape is conservative for a laptop:

- model: `Qwen/Qwen3-4B`
- fine-tune type: `lora`
- iterations: `5000`
- batch size: `1`
- gradient accumulation: `4`
- tuned layers: `8`
- prompt masking: enabled
- gradient checkpointing: enabled

Prompt masking is intentional. The dataset is a `chat` corpus and the loss should land on the assistant JSON only, not on the system or user text.

The default adapter output path is `outputs/text2cypher_mlx/qwen3_4b/adapters`.

For a quick smoke run:

```bash
./venv/bin/text2cypher-train-mlx --iters 250 --dry-run
```

To run MLX test loss on the held-out set after training:

```bash
./venv/bin/text2cypher-train-mlx --run-test-loss
```

Because the prepared dataset now includes `valid.jsonl`, MLX can also use the validation split during training instead of warning that validation data is missing.

## Evaluate The Adapter

Run held-out generation and score the predictions:

```bash
./venv/bin/text2cypher-evaluate-mlx \
  --adapter-path outputs/text2cypher_mlx/qwen3_4b/adapters \
  --force
```

This writes:

- `outputs/text2cypher_mlx/qwen3_4b/evaluation/predictions.jsonl`
- `outputs/text2cypher_mlx/qwen3_4b/evaluation/summary.json`

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
./venv/bin/text2cypher-evaluate-mlx \
  --adapter-path outputs/text2cypher_mlx/qwen3_4b/adapters \
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
./venv/bin/text2cypher-evaluate-mlx \
  --adapter-path outputs/text2cypher_mlx/qwen3_4b/adapters \
  --enable-thinking \
  --force
```

The evaluator will still try to recover the final JSON object from a longer response, but the default recommendation is to keep thinking disabled for the baseline run.
