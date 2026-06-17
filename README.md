# Semantic Mirror

Semantic Mirror turns a source repository into a path-preserving semantic mirror:
reviewable Markdown plus machine-readable JSON sidecars that describe what the
code does, with source-line evidence for every claim.

The core idea is simple: code review and model training both need something
better than a prose summary. Semantic Mirror preserves behavior, side effects,
control flow, data dependencies, failure modes, and Data/ML mechanics in a
structured representation that can be inspected by humans and scored by
deterministic validators.

This repository is a CLI-first research prototype for building, evaluating, and
training those semantic mirrors.

## Why This Exists

Large models are useful at explaining code, but ungrounded explanations are hard
to trust. Semantic Mirror treats explanations as artifacts that must prove where
they came from.

The goal is a system that can:

- generate a mirror repo beside the source repo, preserving paths and symbols
- make every generated claim cite concrete source spans
- show uncertainty instead of silently inventing missing behavior
- produce diff mirrors that highlight changed behavior in a PR
- build datasets from verified mirrors, hard negatives, teacher models, and
  focused human review
- train an open model to emit faithful semantic IR for code review workflows

The first domain is Python Data/ML code: training loops, model definitions,
losses, metrics, dataloaders, optimizers, checkpoint logic, tensor/device
behavior, and experiment configuration.

## What A Mirror Looks Like

Source code:

```python
def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = loss_fn(logits, batch["y"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
```

Semantic Mirror output is not a paragraph like "this trains a model." It is a
structured representation with evidence and review-facing detail:

```json
{
  "unit_id": "train_epoch",
  "symbol_type": "function",
  "source_spans": [{"path": "train.py", "start_line": 1, "end_line": 11}],
  "control_flow": ["iterates over loader batches"],
  "calls": ["model.train", "optimizer.zero_grad", "model", "loss_fn", "loss.backward", "optimizer.step", "loss.item", "len"],
  "returns": ["average loss across loader batches"],
  "state_mutations": ["sets model to training mode", "updates optimizer parameters"],
  "data_ml_details": {
    "losses": ["loss_fn(logits, batch[\"y\"])"],
    "training_loops": ["one optimizer update per batch"],
    "optimizer_scheduler_behavior": ["zero_grad before backward, step after backward"],
    "metrics": ["accumulates loss.item()"],
    "tensor_shapes": [],
    "checkpointing": []
  },
  "uncertainty": []
}
```

The real generated files include richer metadata, static facts, source evidence,
confidence, schema validation, and Markdown rendering for human review.

## Architecture

```mermaid
flowchart TD
    Repo["Source repository"] --> Parse["Tree-sitter + AST extraction"]
    Parse --> Facts["Static facts: symbols, spans, calls, returns, writes, hazards"]
    Facts --> Mirror["Path-preserving mirror"]
    Mirror --> Markdown["*.sir.md review files"]
    Mirror --> JSON["*.sir.json sidecars"]
    Mirror --> Manifest["manifest.json"]

    JSON --> Score["Deterministic scoring"]
    JSON --> Dataset["Dataset sampling"]
    JSON --> Review["Reviewer packs"]

    Dataset --> Teacher["Teacher model requests"]
    Teacher --> Verify["Verifier-backed ingest"]
    Verify --> Training["SFT / DPO / RL batches"]
    Training --> Model["Open semantic mirror model"]
    Model --> Eval["Held-out candidate gates"]
    Eval --> Mirror
```

## Review And Diff Flow

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant CLI as semantic-mirror
    participant Mirror as Mirror Repo
    participant Reviewer as Reviewer
    participant Eval as Gates

    Dev->>CLI: build repo --profile data_ml --zoom L4
    CLI->>Mirror: write path-preserving *.sir.md and *.sir.json
    Dev->>CLI: diff repo --base main --head feature
    CLI->>Mirror: mark changed and context semantic units
    CLI->>Reviewer: generate evidence-backed review pack
    Reviewer->>CLI: conduct timed source-only / mirror-first study
    CLI->>Eval: score coverage, accuracy, speedup, visibility
```

## Training Loop

```mermaid
flowchart LR
    Silver["Silver static examples"] --> Curate["Human curation"]
    HardNeg["Hard negatives"] --> Curate
    Teacher["OpenAI / Anthropic / Gemini teacher outputs"] --> Ingest["Verifier ingest"]
    Ingest --> Critic["Critic labels"]
    Critic --> Prefs["Preference pairs"]
    Curate --> Batch["Training batch"]
    Prefs --> Batch
    Batch --> SFT["SFT"]
    SFT --> DPO["DPO"]
    DPO --> RL["Deterministic-reward RL"]
    RL --> Gates["Schema, coverage, faithfulness, hallucination gates"]
```

The durable asset is not a single hosted model. It is the schema, verifier,
dataset workflow, reward suite, curation loop, and trained open-model path.

## Quick Start

Install with `uv` from the repo root:

```powershell
uv sync
uv run pytest
```

Build a semantic mirror for a Python repository:

```powershell
uv run semantic-mirror build <repo> --out <mirror_repo> --profile data_ml --zoom L4
```

Score the mirror against the source:

```powershell
uv run semantic-mirror score <mirror_repo> --repo <repo>
```

Generate a semantic diff for a PR or commit range:

```powershell
uv run semantic-mirror diff <repo> --base <ref> --head <ref> --out <mirror_repo>/diffs/<id>
```

Create reviewer-facing questions from a mirror:

```powershell
uv run semantic-mirror review pack <mirror_repo> --out <review_pack_dir>
uv run semantic-mirror review study <review_pack_dir> --out <human_study_dir>
uv run semantic-mirror review conduct-study <human_study_dir> --out <answers.jsonl> --reviewer <name>
uv run semantic-mirror eval human-study <human_study_dir> --answers <answers.jsonl> --out <report.json>
```

## Command Map

### Mirror Generation

```powershell
uv run semantic-mirror build <repo> --out <mirror_repo> --profile data_ml --zoom L4
uv run semantic-mirror diff <repo> --base <ref> --head <ref> --out <mirror_repo>/diffs/<id>
uv run semantic-mirror score <mirror_repo> --repo <repo>
```

### Corpus And Dataset Creation

```powershell
uv run semantic-mirror corpus collect --repo <repo_or_git_url> --repo <repo_or_git_url> --out <corpus_dir> --max-units-per-repo 100
uv run semantic-mirror dataset sample <repo> --out <dataset_dir> --max-units 200 --review-budget 50
uv run semantic-mirror dataset promote-gold <dataset_dir> --record-id <silver_or_review_record_id> --label verified_behavior --reviewer <name>
```

### Evaluation

```powershell
uv run semantic-mirror eval mirror <mirror_repo> --repo <repo> --out <report.json>
uv run semantic-mirror eval dataset <dataset_dir> --out <report.json>
uv run semantic-mirror eval compare <baseline_report.json> <current_report.json>
uv run semantic-mirror eval candidates <dataset_dir> --candidates <model_outputs.jsonl> --model-name <run_name> --out <report.json>
uv run semantic-mirror eval model-compare <baseline_eval.json> <current_eval.json> --stage sft
uv run semantic-mirror eval review-pack <review_pack_dir> --mirror <mirror_repo> --out <report.json>
```

### Review Studies

```powershell
uv run semantic-mirror review pack <mirror_repo> --out <review_pack_dir>
uv run semantic-mirror review study <review_pack_dir> --out <human_study_dir>
uv run semantic-mirror review conduct-study <human_study_dir> --out <answers.jsonl> --reviewer <name>
uv run semantic-mirror eval human-study <human_study_dir> --answers <answers.jsonl> --out <report.json>
```

### Teacher And Critic Pipeline

```powershell
uv run semantic-mirror teacher export <dataset_dir> --out <teacher_requests_dir>
uv run semantic-mirror teacher run --requests <candidate_requests.jsonl> --out <responses.jsonl> --max-requests 5
uv run semantic-mirror teacher run --provider anthropic --model claude-sonnet-4-5-20250929 --requests <candidate_requests.jsonl> --out <responses.jsonl> --max-requests 5
uv run semantic-mirror teacher run --provider gemini --model gemini-2.5-flash --requests <candidate_requests.jsonl> --out <responses.jsonl> --max-requests 5
uv run semantic-mirror teacher ingest <dataset_dir> --requests <candidate_requests.jsonl> --responses <responses.jsonl> --out <teacher_results_dir>
uv run semantic-mirror teacher run-critic --requests <teacher_results_dir>/critic_requests.jsonl --out <critic_responses.jsonl> --max-requests 5
uv run semantic-mirror teacher ingest-critic <teacher_results_dir> --responses <critic_responses.jsonl> --out <teacher_results_dir>
uv run semantic-mirror teacher pipeline <dataset_dir> --out <teacher_pipeline_dir> --provider openai --provider anthropic --provider gemini --max-units 5
```

### Training

```powershell
uv run semantic-mirror train prepare <dataset_dir> --out <training_dir>
uv run semantic-mirror train prepare <dataset_dir> --teacher-results <teacher_pipeline_dir> --out <training_dir>
uv run semantic-mirror train validate <training_dir>
uv run semantic-mirror train audit <training_dir> --env-file .env
uv run semantic-mirror train package <training_dir> --out <training_bundle_dir> --env-file .env
uv run semantic-mirror train run-sft <training_dir> --output-dir <sft_model_dir> --max-steps 300 --dry-run
uv run semantic-mirror train run-dpo <training_dir> --model-name-or-path <sft_model_dir> --output-dir <dpo_model_dir> --max-steps 120 --dry-run
uv run semantic-mirror train run-rl <training_dir> --model-name-or-path <dpo_model_dir> --output-dir <rl_model_dir> --max-steps 120 --dry-run
```

Packaged GPU bundles also include `launch/run_full_training_eval.sh`, which
runs SFT, DPO, RL, held-out candidate generation, candidate scoring, SFT-vs-
baseline comparison, and RL-vs-SFT comparison.

## Output Files

| File | Purpose |
| --- | --- |
| `*.sir.md` | Human-readable semantic IR beside a path-preserving source structure. |
| `*.sir.json` | Schema-valid sidecars for training, validation, scoring, and reward computation. |
| `manifest.json` | Repo-level inventory, symbol graph, verifier metadata, confidence, coverage, and optional diff metadata. |
| `silver.jsonl` | Automatically sampled positive examples. |
| `hard_negative.jsonl` | Deterministic contrastive examples with known defects. |
| `review_queue.jsonl` | High-impact or high-disagreement examples for human curation. |
| `gold.jsonl` | Human-promoted examples with labels, reviewer metadata, and notes. |
| `teacher_candidates.jsonl` | Eval-ready teacher model candidates after verifier ingest. |
| `preference_pairs.jsonl` | DPO/RL preference data from hard negatives, critic labels, and teacher outputs. |
| `human_study/` | Source-only, mirror-first, and visibility tasks for usefulness evaluation. |

## Semantic Zoom Levels

| Level | Intended Use | Detail Budget |
| --- | --- | --- |
| `L1` | Repo and module orientation | Intent, major flows, high-level responsibilities. |
| `L2` | Everyday review | Function/class behavior, calls, returns, side effects. |
| `L3` | Behavioral review | Branch predicates, data dependencies, mutation order. |
| `L4` | Data/ML and safety-sensitive review | Implementation details, training mechanics, hazards, uncertainty. |

## Verifiable Rewards

Semantic Mirror is built around measurable faithfulness rather than subjective
"good explanations."

| Reward Area | Positive Signal | Penalty |
| --- | --- | --- |
| Static faithfulness | Preserved calls, returns, writes, state mutations, failure modes. | Invented calls, writes, errors, behavior, or unsupported evidence. |
| Control flow | Preserved branches and guard ordering. | Collapsed predicates when the zoom level requires exact behavior. |
| Data/ML detail | Preserved losses, optimizer behavior, metrics, checkpoints, tensor/device assumptions. | Missing training mechanics or hallucinated model behavior. |
| Review usefulness | Faster accurate mirror-first answers and changed-behavior detection. | Hidden uncertainty or unsupported areas presented as facts. |

Release gates are meant to be concrete:

- at least 90% parsed function/class coverage or an explicit unsupported reason
- 100% generated claims with source-span evidence
- zero verifier-detected invented side effects on gold examples
- diff changed-unit recall of at least 95% on curated PRs
- no more than 1% score regression between accepted releases
- SFT/RL candidates must preserve schema validity, held-out unit coverage,
  static faithfulness, and hallucination penalties against baselines

## Current Implementation Scope

Implemented locally:

- Python AST extraction with Tree-sitter parser metadata.
- Path-preserving whole-repo mirror generation.
- Git ref diff mode over changed files.
- Evidence validation for generated claims.
- Data/ML detectors for losses, model components, training loops, optimizers,
  schedulers, metrics, checkpoints, tensor/device operations, and hazards.
- Deterministic reward scoring for calls, returns, writes, state mutations,
  failure modes, invented facts, and invalid evidence.
- Dataset sampling with silver records, hard negatives, review queues, and gold
  promotion.
- Multi-repository corpus collection for aggregate training batches.
- Reviewer packs for whole-repo and diff-mode review.
- Human usefulness study generation, interactive answer collection, and
  evaluator gates.
- Provider-neutral teacher request export, response ingest, critic execution,
  and critic-label ingest.
- Optional OpenAI, Anthropic, and Gemini teacher execution paths.
- Training preparation for SFT, contrastive repair, preference pairs, and RL
  prompts.
- Runtime audit, dry-run launch gates, and portable Linux/WSL/Colab training
  bundle packaging.

## Training Status

The training path currently targets a Qwen3-family local-fit LoRA configuration
through generated Unsloth/TRL scripts. The current repo can prepare, validate,
package, and dry-run launch commands on Windows, but actual GPU training expects
a Linux or WSL CUDA environment with Python `>=3.11,<3.14` and the training
dependencies installed.

Known local artifact status from the development run:

- compact SFT and repaired RL candidate gates pass for the guarded model
  pipeline
- longer compact SFT, DPO, and deterministic-reward RL runs completed in WSL
- repaired SFT and RL candidate evaluations pass held-out coverage and schema
  validity gates
- raw model completions are not yet a pure contract: deterministic repair still
  fills required schema and source-backed static facts after generation
- human-study artifacts exist, but real timed reviewer logs are still needed
  before claiming human usefulness gates

## Repository Layout

```text
src/semantic_mirror/
  builder.py       mirror generation orchestration
  cli.py           command-line interface
  corpus.py        multi-repository corpus collection
  dataset.py       silver, hard-negative, review-queue, and gold datasets
  evaluation.py    mirror, dataset, candidate, and regression gates
  extractors.py    static extraction and Data/ML detectors
  gitdiff.py       changed-file and changed-unit diff support
  render.py        Markdown rendering
  review.py        review packs and human-study tasks
  rewards.py       deterministic scoring
  schema.py        semantic IR schema validation
  teacher.py       provider-neutral teacher and critic pipeline
  training.py      training batch, audit, package, and launch helpers
tests/
  test_semantic_mirror.py
```

## Example Goal

A strong end-to-end use case should look like this:

1. A developer opens a PR that changes a training loop.
2. `semantic-mirror diff` identifies the changed semantic units, not just the
   changed lines.
3. The mirror shows that the PR moved `optimizer.zero_grad()` after
   `loss.backward()`, marks the training-loop hazard, and cites the exact source
   lines.
4. The reviewer answers behavior questions from the mirror faster than from
   source-only review.
5. The generated IR is accepted into a gold or silver dataset.
6. Future SFT/DPO/RL training improves on that case without losing source-span
   evidence or inventing behavior.

That is the product target: a faithful semantic representation that is useful
for review, measurable enough for regression gates, and structured enough to
train an open model.

## Environment Notes

- Normal CLI development and validation works on Windows with `uv`.
- GPU training is expected to run in Linux or WSL CUDA, not native Windows
  Python 3.14.
- `.env` is used for provider keys during teacher and critic runs. It is not
  copied into packaged training bundles.
- Training artifacts under local output directories are generated artifacts and
  should not be treated as source.

## Project Maturity

This is an active research prototype. The core CLI and guarded pipeline are in
place, but the final research claims still depend on two remaining gates:

- pure raw-output model quality without deterministic repair as a required
  post-processing step
- real timed human-study logs showing mirror-first review usefulness

Until those gates are satisfied, the honest claim is: Semantic Mirror is a
working evidence-backed semantic IR pipeline with training and evaluation
machinery, not a finished model product.
