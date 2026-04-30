# Fine-Tuning Study Notes

This document explains the fine-tuning part of the project in presentation-friendly language. It is written to help defend the choices behind the dataset, the two model pipeline, the training settings, and the evaluation logic during a Q&A.

The main idea is simple: this project does not fine-tune a model to memorize business facts. It fine-tunes a small local query stack so that a user can ask natural-language questions about the business-model knowledge graph, and the system can decide how to handle the question.

The fine-tuning work is focused on two jobs:

1. A router decides where the question should go.
2. A planner turns safe local questions into a compact plan that the application can convert into a database query.

The result is published as a local bundle under `runtime_assets/query_stack/`. The main runtime consumes that bundle; it does not directly import the fine-tuning code.

## The Big Picture

The project builds a business-model knowledge graph from company filings. The graph contains companies, business segments, offerings, customer types, sales channels, revenue models, places, and partners.

Once that graph exists, users want to ask questions such as:

- Which companies sell to developers through direct sales?
- What offerings does a company have?
- How many offerings are linked to a segment?
- Which companies operate in a given place?
- Which companies partner with a given organization?

Some of those questions can be answered locally and safely, because the application already knows how to convert them into a deterministic database lookup. Other questions are too broad, too explanatory, or outside the graph structure. For example, asking for employee-level details, revenue trends, or why two business segments differ may require a stronger model or a refusal.

That is why the fine-tuning design is split into two models instead of one.

The router is the traffic controller. It sees the user question and chooses one of three paths:

- `local`: the local planner should try to answer it.
- `api_fallback`: send it to a stronger hosted model.
- `refuse`: do not answer because the request is unsupported or unsafe.

The planner only runs after the router has chosen `local`. Its job is narrower: it must produce a small JSON plan that says what kind of lookup is needed and what names or filters should be used. The planner does not write Cypher directly, and it does not make refusal decisions. Python code later compiles the plan into Cypher in a controlled way.

This separation is important. It makes the system easier to evaluate, safer to deploy, and cheaper to run.

## Why Two Fine-Tuned Components?

The two-component design is a deliberate safety and reliability choice.

The router protects the local planner from questions it should not handle. If a question asks about employees, dates, revenue numbers, suppliers, explanations, or unsupported comparisons, the router can stop it before the planner produces a bad plan.

The planner is kept intentionally narrow. It only learns supported local query families. That means its output space is smaller and more predictable. Instead of asking it to invent a whole database query, we ask it to fill in a structured form.

This is easier to defend than a single all-purpose model because each component has a clear responsibility:

- The router answers: "Should this question be local, fallback, or refused?"
- The planner answers: "If it is local, what exact structured plan should we use?"
- The runtime answers: "Can this plan be compiled into safe read-only database logic?"

If something goes wrong, the system can fall back to the hosted path rather than silently trusting a bad local plan.

## Dataset Structure

The retained dataset lives under:

`data/query_planner_curated/v1_final/`

It is called the query planner final dataset. Each row is like a teaching card. The row shows a user question and the correct behavior the system should learn.

At the source level, each row contains these main parts:

| Field | Plain-language meaning |
|---|---|
| `question` | The natural-language question a user might ask. |
| `route_label` | The high-level decision: answer locally, send to stronger model, or refuse. |
| `family` | The kind of question, such as partner lookup, offering lookup, count, ranking, geography, or refusal. |
| `supervision_target` | The main lesson used for training. It includes the route and, when local, the correct plan. |
| `target` | A compatibility copy of the runtime-facing target. |
| `gold_cypher` | The reference database query for checking the example. |
| `gold_params` | The exact filters that should be passed into that query. |
| `gold_rows` | The expected answer rows from the synthetic graph. |
| `metadata` | Extra annotations, such as the bucket, graph source, and whether the row uses geography, aggregation, or hierarchy. |

The most important field is `supervision_target`. That is what teaches the model what to do.

For a local question, the supervision target looks conceptually like this:

```json
{
  "route_label": "local_safe",
  "plan": {
    "answerable": true,
    "family": "companies_by_partner",
    "payload": {
      "companies": ["Aurora Systems", "Redwood Retail"],
      "partners": ["Fujitsu Limited"]
    }
  }
}
```

For a non-local question, the target is intentionally simpler:

```json
{
  "route_label": "refuse",
  "plan": {
    "answerable": false,
    "reason": "unsupported_schema"
  }
}
```

The planner is trained only on locally answerable rows. The router sees all rows.

## Dataset Splits

The source dataset has three splits:

| Split | Rows | Purpose |
|---|---:|---|
| `train` | 8,000 | Used to teach the models. |
| `validation` | 1,200 | Used during tuning and calibration. |
| `release_eval` | 1,800 | Held back as a final exam. |

The split boundaries are important. We should not mix them. The validation and release-evaluation rows are meant to test whether the system generalizes beyond the rows it studied.

The dataset also uses different synthetic graph assignments across splits:

- Training: `aurora`, `redwood`, `lattice`
- Validation: `nimbus`
- Release evaluation: `vector`

This matters because the goal is not to memorize a company name. The goal is to learn the shape of the question and the correct query behavior.

## Route Distribution

The router dataset contains all rows and maps them into three runtime labels:

- `local_safe` becomes `local`
- `strong_model_candidate` becomes `api_fallback`
- `refuse` stays `refuse`

The distribution is:

| Split | Local | API fallback | Refuse | Total |
|---|---:|---:|---:|---:|
| Train | 5,000 | 1,500 | 1,500 | 8,000 |
| Validation | 750 | 225 | 225 | 1,200 |
| Release eval | 900 | 450 | 450 | 1,800 |

In the training split, local questions are 62.5 percent of the dataset, while fallback and refusal are each 18.75 percent. The release-evaluation split is harder and more balanced: 50 percent local, 25 percent fallback, and 25 percent refusal.

This distribution is useful because it gives the router many positive local examples while still forcing it to learn when not to use the local planner.

## Local Question Types

The locally answerable examples are divided into question families. These are the families the planner can produce:

| Family | What it means |
|---|---|
| `companies_list` | List companies in the graph. |
| `segments_by_company` | Find business segments for a company. |
| `offerings_by_company` | Find offerings owned by a company. |
| `offerings_by_segment` | Find offerings attached to a business segment. |
| `companies_by_segment_filters` | Find companies whose segments match customer, channel, or revenue-model filters. |
| `segments_by_segment_filters` | Find segments that match customer, channel, or revenue-model filters. |
| `companies_by_cross_segment_filters` | Find companies matching filters across multiple segments. |
| `descendant_offerings_by_root` | Walk an offering hierarchy from a root offering to its descendants. |
| `companies_by_descendant_revenue` | Find companies linked to descendant offerings and revenue models. |
| `companies_by_place` | Find companies operating in a place. |
| `segments_by_place_and_segment_filters` | Combine place filters with segment-level filters. |
| `companies_by_partner` | Find companies that partner with a named organization. |
| `boolean_exists` | Answer whether a supported graph pattern exists. |
| `count_aggregate` | Count companies, segments, or offerings for a supported family. |
| `ranking_topk` | Return top-k rankings for a small set of approved ranking metrics. |

The local-safe training distribution before balancing is:

| Local family | Train rows |
|---|---:|
| `companies_by_cross_segment_filters` | 700 |
| `companies_by_segment_filters` | 600 |
| `segments_by_segment_filters` | 600 |
| `boolean_exists` | 400 |
| `companies_by_descendant_revenue` | 350 |
| `count_aggregate` | 350 |
| `descendant_offerings_by_root` | 350 |
| `segments_by_place_and_segment_filters` | 380 |
| `offerings_by_segment` | 270 |
| `companies_by_place` | 220 |
| `segments_by_company` | 210 |
| `companies_by_partner` | 200 |
| `offerings_by_company` | 180 |
| `ranking_topk` | 150 |
| `companies_list` | 40 |

Some families are naturally rarer than others. That is why the planner training set is balanced after preparation.

Concrete examples make the family names easier to defend in a presentation:

| Family | Example user question |
|---|---|
| `companies_list` | "From what we have here, list up to 1 company represented in the knowledge graph." |
| `segments_by_company` | "List up to 4 business segments for Redwood Retail." |
| `offerings_by_company` | "Based on this graph, name as many as 4 offerings owned by Aurora Systems, Lattice Finance, and Redwood Retail." |
| `offerings_by_segment` | "From what we have here, list up to 4 offerings in the Compliance Services segment of Lattice Finance." |
| `companies_by_segment_filters` | "List the companies that have a business segment that serves small businesses and sells through marketplaces." |
| `segments_by_segment_filters` | "Which business segments monetize via subscription?" |
| `companies_by_cross_segment_filters` | "From what we have here, list the companies that can serve industrial companies and merchant businesses across their segments." |
| `descendant_offerings_by_root` | "From what we have here, name as many as 2 offerings in the Loyalty Cloud family at Redwood Retail." |
| `companies_by_descendant_revenue` | "In this graph, within Aurora Systems and Redwood Retail, which companies use consumption-based for offerings under Cloud Platform, up to 2 results?" |
| `companies_by_place` | "From what we have here, name as many as 5 companies with presence in Mexico." |
| `segments_by_place_and_segment_filters` | "In this graph, list the business segments for companies operating in the United Kingdom that serve government agencies." |
| `companies_by_partner` | "Based on this graph, which of Aurora Systems and Redwood Retail partner with Fujitsu Limited?" |
| `boolean_exists` | "Could Lattice Finance qualify through a business segment that sells through OEMs?" |
| `count_aggregate` | "Based on this graph, how many offerings does Aurora Systems have?" |
| `ranking_topk` | "In this graph, top 3 channels by segment count within segments at Aurora Systems." |

## Non-Local Question Types

The dataset also teaches the system when not to use the local planner.

The `strong_model_candidate` rows are questions that may still be useful, but are beyond the local planner's limited contract. These include:

- explaining why a segment matches something
- comparing segments by channels or customer types
- comparing companies by channels or customer types
- finding common or unique offerings between segments
- weighted ranking requests

The `refuse` rows are questions the system should not answer locally or through the normal fallback path. The refusal reasons include:

| Reason | Meaning |
|---|---|
| `unsupported_schema` | The question asks for something not represented in the graph, such as employees or suppliers. |
| `unsupported_metric` | The question asks for unsupported numbers, such as revenue amount, price, growth, or other metrics. |
| `unsupported_time` | The question depends on time, trends, dates, or year-over-year analysis. |
| `ambiguous_closed_label` | The question uses a label that cannot be confidently mapped to a supported category. |
| `ambiguous_request` | The request is too vague to form a safe query. |
| `write_request` | The user asks to edit, delete, or mutate data. |
| `beyond_local_coverage` | The question is outside the supported local families. |

This negative coverage is important. Without it, a model could become over-eager and try to answer questions that the local graph cannot support.

## Data Preparation Pipeline

The command `prepare-data` converts the source dataset into the exact shapes needed by the two training tasks.

It creates one dataset for the router and one for the planner.

For the router, each row becomes:

- the user question
- the simplified runtime label: `local`, `api_fallback`, or `refuse`
- the split name
- the source route label
- the question family

This means the router is trained like a classification task. It reads a question and chooses one label.

For the planner, only `local_safe` rows are kept. Each planner row becomes a small conversation:

- system message: the frozen planner instructions
- user message: the question
- assistant message: the compact JSON plan

The planner sees examples in the same form it will see at runtime. That is helpful because the training format matches the deployment format.

The preparation step also creates:

- a raw planner dataset
- a balanced planner dataset
- frozen prompt files
- summary files with counts by split and family

The balanced planner dataset repeats rarer local families so they do not get drowned out by the more common families. In the current fresh-adapter prepared dataset, the raw planner train split has 5,090 rows, including 90 planner-only augmentation rows. The balanced train split has 9,480 rows.

## Why Balance The Planner Data?

The planner does not just need to be good at the most common question type. It needs to know all supported local families.

Without balancing, rare families such as `companies_list`, `ranking_topk`, or `companies_by_partner` would receive much less training signal. The model might learn the common families well and perform poorly on narrower but still important tasks.

Balancing is done by repeating underrepresented families, not by changing the validation or release-evaluation splits. That distinction matters:

- Training is adjusted to help learning.
- Evaluation remains untouched so the score is honest.

## Router Fine-Tuning

The router model is:

`microsoft/deberta-v3-small`

Its job is to classify the question into one of three runtime outcomes:

- `local`
- `api_fallback`
- `refuse`

The main router settings are:

| Setting | Value | Why it makes sense |
|---|---:|---|
| Base model | `microsoft/deberta-v3-small` | Small, fast, strong for short text classification. |
| Max input length | 256 | User questions are short; longer context is not needed. |
| Train batch size | 16 | Practical batch size for local training. |
| Eval batch size | 32 | Evaluation can be batched more aggressively. |
| Learning rate | 0.00002 | Conservative update size for fine-tuning an existing language model. |
| Weight decay | 0.01 | Helps prevent overfitting. |
| Epochs | Up to 6 | Enough passes over the data, with early stopping. |
| Early stopping patience | 2 | Stops if validation quality stops improving. |
| Seed | 7 | Makes the run more reproducible. |
| Minimum local precision gate | 0.97 | The local route should be trusted only when it is precise enough. |

The router uses class weighting during training. This means mistakes on smaller classes are not ignored just because the local class is larger.

The router is evaluated using macro F1 and accuracy. Macro F1 is important because it gives each class a fair voice. If we only looked at accuracy, a model could look good by over-predicting the largest class.

## Router Runtime Policy

At runtime, the router does not simply choose the largest score in every case. It uses a conservative local threshold:

`Use local only when P(local) >= 0.97. Otherwise choose between api_fallback and refuse.`

This is a practical reliability choice. A false local decision can send an unsupported question into the planner. A false fallback is less dangerous, because the hosted path can still try to handle the question. So the system intentionally requires high confidence before allowing local execution.

In the published runtime bundle, the router threshold file records:

- temperature calibration: about 1.0
- local threshold: 0.97
- planner gate open: true

On release evaluation, the router policy achieves:

| Metric | Result |
|---|---:|
| Overall policy accuracy | 96.4 percent |
| Local precision | 97.8 percent |
| Local recall | 100.0 percent |
| API fallback precision | 100.0 percent |
| API fallback recall | 86.2 percent |
| Refuse precision | 90.9 percent |
| Refuse recall | 99.3 percent |

The key defense point is that the router is optimized for safe local routing, not just raw label accuracy. The policy is designed so the local planner only receives questions the router is confident it can handle.

## Planner Fine-Tuning

The planner model is:

`mlx-community/Qwen3-4B-Instruct-2507-4bit`

The planner is fine-tuned using QLoRA-style adapter training in MLX. In plain language, this means we do not retrain the whole model. We keep the base model mostly fixed and train a smaller set of adapter weights that teach it our specific planning format.

This choice is practical for local development:

- The 4-bit model is small enough to run locally on Apple Silicon.
- The model is still instruction-following enough to emit structured JSON.
- Adapter training is much cheaper than full model training.
- The resulting adapter can be published and loaded by the runtime.

The planner does not learn to answer the user's question in prose. It learns to produce compact JSON such as:

```json
{
  "answerable": true,
  "family": "offerings_by_company",
  "payload": {
    "companies": ["Aurora Systems"],
    "limit": 4
  }
}
```

The runtime then validates and compiles that plan into database logic.

## Planner Settings

The main planner settings are:

| Setting | Value | Why it makes sense |
|---|---:|---|
| Base model | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | Local, compact, instruction-following model. |
| Data variant | Balanced | Protects rare query families. |
| Adapter rank | 16 | Gives the adapter enough capacity without becoming too large. |
| Adapter alpha | 32 | Controls adapter scaling. |
| Dropout | 0.05 | Mild regularization to reduce overfitting. |
| Layers adapted | 16 | Adapts a meaningful portion of the model while staying lightweight. |
| Learning rate | 0.0001 | Typical adapter fine-tuning scale. |
| Max generated tokens | 256 | The output should be compact JSON, not a long answer. |
| Mask prompt | true | The model is trained on the answer, not rewarded for repeating the prompt. |
| Checkpoint interval | 500 iterations | Allows restart and comparison between checkpoints. |
| Seed | 7 | Improves reproducibility. |

The currently published planner adapter was trained from the fresh-adapter recipe with:

- balanced planner data
- batch size 4
- gradient accumulation 4
- effective batch size 16
- max sequence length 4096
- 7,110 total iterations in the published training configuration
- checkpoints every 500 iterations
- resume support from a full checkpoint

There are older or alternate config files in `finetuning/config/` for continuing or resuming experiments. They do not change the conceptual design. They exist so training can be restarted from adapter files or full checkpoints without starting from zero.

## Frozen Planner Prompt

The planner is trained with a frozen system prompt. This prompt tells it:

- return compact JSON only
- do not write Cypher
- do not explain
- do not return refusals
- copy open-class names exactly as written
- normalize only closed vocabulary labels
- use only the supported query families

This matters because some values are open names, while others are fixed labels.

Open names include:

- company names
- segment names
- offering names
- partner names
- place names

These should be copied from the question rather than paraphrased.

Closed labels include:

- customer types
- channels
- revenue models

These should be normalized to the exact vocabulary. For example, "public sector" maps to "government agencies", and "direct" maps to "direct sales".

This is one of the central reasons for fine-tuning the planner: generic models are often tempted to paraphrase. The local query stack needs exact structured values.

## Planner Evaluation

The planner is evaluated on local-safe validation and release-evaluation rows. It is not evaluated on fallback or refusal rows because the router should prevent those from reaching the planner.

The planner evaluation checks five things:

| Metric | Meaning |
|---|---|
| JSON parse rate | Did the model produce readable JSON? |
| Contract valid rate | Does the JSON match the expected plan shape? |
| Family accuracy | Did it choose the right query family? |
| Exact plan match rate | Did it match the full gold plan exactly? |
| Correct output rate | If the predicted plan is executed against the synthetic graph, does it return the same final rows as the gold answer? |

The latest saved planner evaluation for the published 2,500-iteration adapter scores:

| Split | JSON parse | Contract valid | Family accuracy | Exact plan match | Correct output |
|---|---:|---:|---:|---:|---:|
| Validation | 100.0 percent | 100.0 percent | 99.9 percent | 90.7 percent | 91.5 percent |
| Release eval | 100.0 percent | 100.0 percent | 99.4 percent | 76.6 percent | 78.1 percent |

The exact match metric is intentionally strict. It is useful for finding remaining issues, but it can undercount useful behavior because a plan may differ from the gold JSON while still returning the same graph rows. The correct output rate is therefore the best presentation metric for final planner usefulness: validation returns the correct graph rows in 686 of 750 cases, and release evaluation returns the correct graph rows in 703 of 900 cases.

## Publishing And Runtime Handoff

After training and evaluation, `publish-query-stack` creates the runtime bundle:

`runtime_assets/query_stack/`

That bundle contains:

- router model files
- router threshold metadata
- planner adapter files
- planner frozen system prompt
- a manifest that tells the runtime where everything is

The main runtime loads the bundle and follows this process:

1. The router scores the question.
2. If local confidence is at least 0.97, the local planner is allowed to run.
3. If local confidence is lower, the runtime chooses between fallback and refusal.
4. If the planner produces an invalid plan, the runtime falls back.
5. If the plan is valid, Python compiles it into read-only Cypher.

This means the final system has multiple guardrails:

- router confidence threshold
- planner contract validation
- deterministic compiler
- read-only query validation
- fallback if local planning fails

## Why This Is Defensible

The fine-tuning design is defensible because it is not trying to make a small local model do everything.

The router handles decision-making. The planner handles only a narrow, structured planning task. The database compiler handles actual query generation. The fallback path handles broader requests.

This division avoids a common mistake: asking one model to classify, plan, generate database queries, enforce safety, and answer users all at once.

Instead, each part is small enough to test:

- Router: did it choose the right path?
- Planner: did it produce the right plan shape?
- Compiler: did the plan become valid read-only Cypher?
- Runtime: did the result come from local, fallback, or refusal?

That makes the architecture easier to explain and easier to improve.

## Known Limitations

The dataset is strong, but not perfect.

One limitation is that some distribution balance is numerical rather than semantic. For example, a phrase like "how many" appears mostly in supported local count questions. That can make the router too trusting of count-style questions unless there are enough hard negative examples.

Another limitation is that refusals are mostly business-domain refusals involving known graph-like language, such as unsupported employees, suppliers, revenue, or time-series facts. The dataset has fewer completely out-of-domain refusals.

Some planner families are also harder than others. Hierarchy wording, possessive phrasing, and combinations of geography with segment filters can be tricky. The project notes suggest the next dataset iteration should add more hard paraphrases, more hard refusals, and more coverage for validation and release-evaluation patterns.

The most important future improvement is probably the router dataset. If the router sends a valid local question to fallback, the planner never gets a chance. If the router sends an unsupported question to local, the planner may fail before fallback catches it. Better router negatives and paraphrases would improve the whole system.

## Common Q&A Defense Points

If asked why we did not train one model end-to-end, the answer is that the split architecture is safer and easier to validate. The router, planner, and compiler each have one job.

If asked whether the model memorizes the dataset, the answer is no, not in the intended sense. The dataset uses synthetic graph names and held-out splits. The goal is to learn query behavior, not real-world company facts.

If asked why gold Cypher is in the dataset, the answer is that it provides a reference for checking correctness. The planner is not trained to write Cypher. It is trained to write the compact plan that the runtime compiles.

If asked why the router has a 0.97 local threshold, the answer is that local execution should be high precision. It is safer to fall back than to force an unsupported question into local planning.

If asked why the planner is trained only on local rows, the answer is that the router owns refusal and fallback. The planner should not learn to make routing decisions because that would blur responsibilities.

If asked why DeBERTa is used for the router, the answer is that routing is a short-text classification problem. A compact classifier is faster and cheaper than using a generative model for every routing decision.

If asked why Qwen3 4B 4-bit is used for the planner, the answer is that the planner needs instruction-following and JSON generation, but it must still run locally. The 4-bit model plus adapter training gives a good balance between capability and local practicality.

If asked why QLoRA or adapters are used, the answer is that the project only needs to teach a narrow behavior. Adapter fine-tuning is much cheaper than retraining the full model and produces a small deployable artifact.

If asked why the planner exact match is lower than family accuracy, the answer is that exact match is stricter. A model can choose the right family and produce valid JSON while still differing in one payload detail. Exact match is useful for debugging, while contract validity and family accuracy show whether the planner is generally behaving correctly.

If asked how the system avoids unsafe database behavior, the answer is that the planner does not write database code directly. It emits a plan, the runtime validates the plan, compiles it deterministically, checks read-only constraints, and falls back if something is invalid.

If asked what the biggest weakness is, the answer is router coverage. The local planner depends on the router. Improving hard paraphrases and hard negatives in the router dataset would likely give the highest return.

## One-Sentence Summary For The Presentation

The fine-tuning pipeline teaches a small local query stack to route business-graph questions safely and, only when appropriate, translate them into compact structured plans that the runtime can validate and compile into database queries.
