# Semantic Mirror

Semantic Mirror builds a path-preserving mirror repository containing static,
evidence-backed semantic IR for source files.

The first implementation is a CLI-first slice of `SEMANTIC_MIRROR_PLAN.md`.
It supports Python repositories with deterministic AST extraction and emits both
human-readable Markdown and machine-readable JSON sidecars.

## Commands

```powershell
uv run semantic-mirror build <repo> --out <mirror_repo> --profile data_ml --zoom L4
uv run semantic-mirror diff <repo> --base <ref> --head <ref> --out <mirror_repo>/diffs/<id>
uv run semantic-mirror score <mirror_repo> --repo <repo>
uv run semantic-mirror corpus collect --repo <repo_or_git_url> --repo <repo_or_git_url> --out <corpus_dir> --max-units-per-repo 100
uv run semantic-mirror dataset sample <repo> --out <dataset_dir> --max-units 200 --review-budget 50
uv run semantic-mirror dataset promote-gold <dataset_dir> --record-id <silver_or_review_record_id> --label verified_behavior --reviewer <name>
uv run semantic-mirror eval mirror <mirror_repo> --repo <repo> --out <report.json>
uv run semantic-mirror eval dataset <dataset_dir> --out <report.json>
uv run semantic-mirror eval compare <baseline_report.json> <current_report.json>
uv run semantic-mirror eval candidates <dataset_dir> --candidates <model_outputs.jsonl> --model-name <run_name> --out <report.json>
uv run semantic-mirror eval model-compare <baseline_eval.json> <current_eval.json> --stage sft
uv run semantic-mirror review pack <mirror_repo> --out <review_pack_dir>
uv run semantic-mirror eval review-pack <review_pack_dir> --mirror <mirror_repo> --out <report.json>
uv run semantic-mirror review study <review_pack_dir> --out <human_study_dir>
uv run semantic-mirror review conduct-study <human_study_dir> --out <answers.jsonl> --reviewer <name>
uv run semantic-mirror eval human-study <human_study_dir> --answers <answers.jsonl> --out <report.json>
uv run semantic-mirror train prepare <dataset_dir> --out <training_dir>
uv run semantic-mirror train prepare <dataset_dir> --teacher-results <teacher_pipeline_dir> --out <training_dir>
uv run semantic-mirror train validate <training_dir>
uv run semantic-mirror train audit <training_dir> --env-file .env
uv run semantic-mirror train package <training_dir> --out <training_bundle_dir> --env-file .env
uv run semantic-mirror train run-sft <training_dir> --output-dir <sft_model_dir> --dry-run
uv run semantic-mirror train run-dpo <training_dir> --model-name-or-path <sft_model_dir> --output-dir <dpo_model_dir> --dry-run
uv run semantic-mirror train run-rl <training_dir> --model-name-or-path <dpo_model_dir> --output-dir <rl_model_dir> --dry-run
# Packaged GPU bundles also include `launch/run_full_training_eval.sh` for SFT/DPO/RL plus held-out candidate scoring and model-compare gates.
uv run semantic-mirror teacher export <dataset_dir> --out <teacher_requests_dir>
uv run semantic-mirror teacher run --requests <candidate_requests.jsonl> --out <responses.jsonl> --max-requests 5
uv run semantic-mirror teacher run --provider anthropic --model claude-sonnet-4-5-20250929 --requests <candidate_requests.jsonl> --out <responses.jsonl> --max-requests 5
uv run semantic-mirror teacher run --provider gemini --model gemini-2.5-flash --requests <candidate_requests.jsonl> --out <responses.jsonl> --max-requests 5
uv run semantic-mirror teacher ingest <dataset_dir> --requests <candidate_requests.jsonl> --responses <responses.jsonl> --out <teacher_results_dir>
uv run semantic-mirror eval candidates <dataset_dir> --candidates <teacher_results_dir>/teacher_candidates.jsonl --model-name teacher-baseline --out <teacher_baseline_eval.json>
uv run semantic-mirror teacher run-critic --requests <teacher_results_dir>/critic_requests.jsonl --out <critic_responses.jsonl> --max-requests 5
uv run semantic-mirror teacher ingest-critic <teacher_results_dir> --responses <critic_responses.jsonl> --out <teacher_results_dir>
uv run semantic-mirror teacher pipeline <dataset_dir> --out <teacher_pipeline_dir> --provider openai --provider anthropic --provider gemini --max-units 5
```

## Output

- `*.sir.md`: reviewable semantic IR beside a path-preserving source structure.
- `*.sir.json`: schema-valid sidecars for training, validation, and rewards.
- `manifest.json`: repo-level inventory, symbol graph, verifier metadata,
  confidence, coverage, and optional diff metadata.

## Current Scope

- Python AST extraction.
- Tree-sitter Python parsing before AST fact extraction, with parser metadata
  stored in IR sidecars, manifests, datasets, teacher requests, and training prompts.
- Explicit semantic zoom policy: `L1` keeps high-level intent and major flows,
  `L2` adds function/class behavior and side effects, `L3` adds data
  dependencies and order annotations, and `L4` includes implementation-sensitive
  details and Data/ML mechanics.
- Whole-repo build mode.
- Git ref diff mode over changed files.
- Evidence validation for generated claims.
- Data/ML detectors for training loops, losses, optimizers, metrics,
  checkpoints, model components, tensor/device operations, and related hazards.
- Schema validation enforces the required `data_ml_details` categories:
  losses, model architecture, tensor shapes, training loops,
  optimizer/scheduler behavior, metrics, and checkpointing.
- Deterministic reward scoring for preserved calls, returns, writes, state
  mutations, failure modes, invented facts, and invalid evidence.
- Dataset sampling that emits `silver.jsonl`, deterministic `hard_negative.jsonl`,
  an active-learning `review_queue.jsonl`, and an initially empty `gold.jsonl`
  for curator-promoted examples.
- Gold-set promotion for reviewed silver or review-queue records, with curation
  labels, reviewer metadata, and notes stored alongside the promoted training
  example.
- Corpus collection that references local repositories or shallow-clones Git
  URLs, samples each repository independently, and emits an aggregate dataset
  under `<corpus_dir>/aggregate` for multi-repository training preparation.
- Gate evaluation for mirror quality, dataset curation, diff changed-unit
  recall, and no-more-than-1% score regression comparisons.
- Reviewer packs that turn whole-repo mirrors and diff mirrors into
  evidence-backed repo questions, changed-behavior tasks, and explicit
  unsupported or low-confidence visibility lists for human usefulness gates.
- Human usefulness study artifacts that pair source-only and mirror-first timed
  tasks, plus evaluator gates for answer coverage, mirror answer accuracy,
  mirror speedup, changed-behavior accuracy, and visibility-marker
  acknowledgement.
- Interactive human study runner that presents source-only, mirror-first, or
  visibility tasks, records elapsed time, asks the reviewer to score correctness
  after answering, and writes evaluator-ready answer JSONL without overwriting
  prior logs by default.
- Held-out model candidate evaluation and teacher/SFT/RL comparison gates for
  schema validity, held-out unit coverage, static faithfulness, and
  hallucination penalties, including aggregate-corpus scoring against each
  record's original source repository.
- SFT/RL preparation that converts curated data into chat-style SFT records,
  contrastive repair records, preference pairs, RL prompts, and Unsloth-oriented
  QLoRA/reward configs, with optional teacher-ingest preference pairs merged
  into the DPO/RL preference set.
- Generated training scripts for Unsloth/TRL SFT, TRL DPO preference training,
  deterministic-reward policy optimization, model candidate generation, and
  deterministic candidate scoring, plus `train validate` to check batch shape
  before GPU training.
- Runtime training audit and guarded SFT/DPO/RL launch commands that validate the
  prepared batch, Unsloth-compatible Python version, dependency imports, CUDA
  availability, optional Hugging Face token presence, and dry-run launch
  commands before starting a GPU job.
- Portable training bundle packaging for Linux/WSL/Colab handoff, including the
  validated batch, generated launch scripts, Semantic Mirror runtime source for
  candidate scoring, CUDA-oriented requirements, bootstrap scripts for a Python
  3.11-3.13 CUDA environment, and a sanitized current-machine audit without
  copying `.env`.
- Packaged full-run GPU handoff script for SFT, DPO, RL, held-out candidate
  generation, candidate scoring, SFT-vs-baseline comparison, and RL-vs-SFT
  comparison reports.
- Provider-neutral teacher request export and response ingest for frontier-model
  candidate generation, verifier-backed auto-rejection, critic requests,
  eval-ready `teacher_candidates.jsonl` baseline outputs, and teacher
  preference pairs.
- Teacher and critic API runners persist each response or error incrementally so
  long provider jobs can be monitored and salvaged after rate limits or parse
  failures.
- Provider-neutral critic execution and ingest that turns `critic_requests.jsonl`
  into structured error labels, labeled review queues, and optional DPO/RL
  preference metadata for "label errors instead of rewrite" curation.
- Optional OpenAI-backed teacher request execution using `OPENAI_API_KEY` and
  `SEMANTIC_MIRROR_TEACHER_MODEL` from `.env`, with test coverage using a fake
  transport so local validation does not spend API calls.
- Anthropic and Gemini teacher execution paths using `ANTHROPIC_API_KEY` and
  `GEMINI_API_KEY`, also covered by fake transports.
- A teacher pipeline command that exports provider-scoped requests, runs selected
  providers, combines responses, and immediately ingests them through the
  deterministic verifier to create candidate results, critic requests,
  preference pairs, and a review queue.
