"""Command-line interface for Semantic Mirror."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_mirror.builder import build_repository, diff_repository
from semantic_mirror.corpus import collect_corpus
from semantic_mirror.dataset import promote_gold_records, sample_dataset
from semantic_mirror.evaluation import (
    compare_regression_reports,
    compare_model_evaluations,
    evaluate_dataset,
    evaluate_mirror,
    evaluate_model_candidates,
)
from semantic_mirror.rewards import score_mirror
from semantic_mirror.review import (
    conduct_human_usefulness_study,
    create_human_study_collection_plan,
    create_human_usefulness_study,
    create_review_pack,
    evaluate_human_usefulness_study,
    evaluate_review_pack,
    summarize_human_study_answer_coverage,
    summarize_human_usefulness_studies,
)
from semantic_mirror.schema import SUPPORTED_PROFILES, SUPPORTED_ZOOMS
from semantic_mirror.teacher import (
    SUPPORTED_TEACHER_PROVIDERS,
    export_teacher_requests,
    ingest_critic_responses,
    ingest_teacher_responses,
    run_critic_requests,
    run_teacher_pipeline,
    run_teacher_requests,
)
from semantic_mirror.training import (
    DEFAULT_BASE_MODEL,
    audit_training_environment,
    create_sample_inspection,
    generate_training_diagnostics,
    generate_training_package_source_freshness,
    launch_training_job,
    package_training_bundle,
    prepare_training_data,
    summarize_full_eval_contract_status,
    validate_training_batch,
)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        manifest = build_repository(
            Path(args.repo),
            Path(args.out),
            profile=args.profile,
            zoom=args.zoom,
        )
    elif args.command == "diff":
        manifest = diff_repository(
            Path(args.repo),
            Path(args.out),
            base=args.base,
            head=args.head,
            profile=args.profile,
            zoom=args.zoom,
        )
    elif args.command == "score":
        report = score_mirror(
            Path(args.mirror),
            repo_path=Path(args.repo) if args.repo is not None else None,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    elif args.command == "corpus" and args.corpus_command == "collect":
        manifest = collect_corpus(
            args.repo,
            Path(args.out),
            profile=args.profile,
            zoom=args.zoom,
            max_units_per_repo=args.max_units_per_repo,
            review_budget_per_repo=args.review_budget_per_repo,
            hard_negatives_per_unit=args.hard_negatives_per_unit,
        )
    elif args.command == "dataset" and args.dataset_command == "sample":
        manifest = sample_dataset(
            Path(args.repo),
            Path(args.out),
            profile=args.profile,
            zoom=args.zoom,
            max_units=args.max_units,
            review_budget=args.review_budget,
            hard_negatives_per_unit=args.hard_negatives_per_unit,
        )
    elif args.command == "dataset" and args.dataset_command == "promote-gold":
        manifest = promote_gold_records(
            Path(args.dataset),
            args.record_id,
            labels=args.label,
            reviewer=args.reviewer,
            notes=args.notes,
        )
    elif args.command == "eval" and args.eval_command == "mirror":
        report = evaluate_mirror(
            Path(args.mirror),
            repo_path=Path(args.repo) if args.repo is not None else None,
            out_path=Path(args.out) if args.out is not None else None,
        )
        print(json.dumps(_eval_summary(report), indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    elif args.command == "eval" and args.eval_command == "dataset":
        report = evaluate_dataset(
            Path(args.dataset),
            out_path=Path(args.out) if args.out is not None else None,
        )
        print(json.dumps(_eval_summary(report), indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    elif args.command == "eval" and args.eval_command == "compare":
        report = compare_regression_reports(
            Path(args.baseline_report),
            Path(args.current_report),
            max_score_drop=args.max_score_drop,
            out_path=Path(args.out) if args.out is not None else None,
        )
        print(json.dumps(_eval_summary(report), indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    elif args.command == "eval" and args.eval_command == "candidates":
        report = evaluate_model_candidates(
            Path(args.dataset),
            Path(args.candidates),
            model_name=args.model_name,
            out_path=Path(args.out) if args.out is not None else None,
        )
        print(json.dumps(_eval_summary(report), indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    elif args.command == "eval" and args.eval_command == "model-compare":
        report = compare_model_evaluations(
            Path(args.baseline_report),
            Path(args.current_report),
            stage=args.stage,
            min_score_improvement=args.min_score_improvement,
            out_path=Path(args.out) if args.out is not None else None,
        )
        print(json.dumps(_eval_summary(report), indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    elif args.command == "eval" and args.eval_command == "review-pack":
        report = evaluate_review_pack(
            Path(args.review_pack),
            mirror_path=Path(args.mirror) if args.mirror is not None else None,
            out_path=Path(args.out) if args.out is not None else None,
        )
        print(json.dumps(_eval_summary(report), indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    elif args.command == "eval" and args.eval_command == "human-study":
        report = evaluate_human_usefulness_study(
            Path(args.study),
            Path(args.answers),
            min_accuracy=args.min_accuracy,
            min_speedup=args.min_speedup,
            out_path=Path(args.out) if args.out is not None else None,
        )
        print(json.dumps(_eval_summary(report), indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    elif args.command == "eval" and args.eval_command == "human-study-suite":
        report = summarize_human_usefulness_studies(
            [Path(path) for path in args.report],
            out_path=Path(args.out) if args.out is not None else None,
        )
        print(json.dumps(_eval_summary(report), indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    elif args.command == "review" and args.review_command == "pack":
        manifest = create_review_pack(
            Path(args.mirror),
            Path(args.out),
            max_questions=args.max_questions,
            max_change_tasks=args.max_change_tasks,
        )
    elif args.command == "review" and args.review_command == "study":
        manifest = create_human_usefulness_study(
            Path(args.review_pack),
            Path(args.out),
        )
    elif args.command == "review" and args.review_command == "conduct-study":
        manifest = conduct_human_usefulness_study(
            Path(args.study),
            Path(args.out),
            reviewer=args.reviewer,
            task_set=args.task_set,
            max_tasks=args.max_tasks,
            append=args.append,
            overwrite=args.overwrite,
        )
    elif args.command == "review" and args.review_command == "study-status":
        report = summarize_human_study_answer_coverage(
            Path(args.study),
            Path(args.answers) if args.answers is not None else None,
            out_path=Path(args.out) if args.out is not None else None,
        )
        print(json.dumps(_eval_summary(report), indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    elif args.command == "review" and args.review_command == "study-collection-plan":
        manifest = create_human_study_collection_plan(
            _parse_labeled_paths(args.study),
            answers_dir=Path(args.answers_dir),
            reviewer=args.reviewer,
            batch_size=args.batch_size,
            out_path=Path(args.out) if args.out is not None else None,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    elif args.command == "train" and args.train_command == "prepare":
        manifest = prepare_training_data(
            Path(args.dataset),
            Path(args.out),
            base_model=args.base_model,
            max_records=args.max_records,
            include_silver_when_gold_exists=not args.gold_only,
            teacher_results_path=Path(args.teacher_results)
            if args.teacher_results is not None
            else None,
        )
    elif args.command == "train" and args.train_command == "validate":
        report = validate_training_batch(Path(args.training_dir))
        if args.out is not None:
            Path(args.out).write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(_eval_summary(report), indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    elif args.command == "train" and args.train_command == "audit":
        report = audit_training_environment(
            Path(args.training_dir),
            env_file=Path(args.env_file) if args.env_file is not None else None,
            require_gpu=not args.allow_cpu,
            require_hf_token=args.require_hf_token,
            python_executable=args.python_executable,
        )
        if args.out is not None:
            Path(args.out).write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(_training_runtime_summary(report), indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    elif args.command == "train" and args.train_command == "run-sft":
        report = launch_training_job(
            Path(args.training_dir),
            stage="sft",
            output_dir=Path(args.output_dir),
            dry_run=args.dry_run,
            env_file=Path(args.env_file) if args.env_file is not None else None,
            require_gpu=not args.allow_cpu,
            require_hf_token=args.require_hf_token,
            python_executable=args.python_executable,
            max_steps=args.max_steps,
            resume_from_checkpoint=args.resume_from_checkpoint,
            seed=args.seed,
            report_out=Path(args.report_out) if args.report_out is not None else None,
        )
        print(json.dumps(_training_runtime_summary(report), indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    elif args.command == "train" and args.train_command == "run-dpo":
        report = launch_training_job(
            Path(args.training_dir),
            stage="dpo",
            output_dir=Path(args.output_dir),
            dry_run=args.dry_run,
            env_file=Path(args.env_file) if args.env_file is not None else None,
            require_gpu=not args.allow_cpu,
            require_hf_token=args.require_hf_token,
            python_executable=args.python_executable,
            model_name_or_path=args.model_name_or_path,
            beta=args.beta,
            max_steps=args.max_steps,
            resume_from_checkpoint=args.resume_from_checkpoint,
            seed=args.seed,
            report_out=Path(args.report_out) if args.report_out is not None else None,
        )
        print(json.dumps(_training_runtime_summary(report), indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    elif args.command == "train" and args.train_command == "run-rl":
        report = launch_training_job(
            Path(args.training_dir),
            stage="rl",
            output_dir=Path(args.output_dir),
            dry_run=args.dry_run,
            env_file=Path(args.env_file) if args.env_file is not None else None,
            require_gpu=not args.allow_cpu,
            require_hf_token=args.require_hf_token,
            python_executable=args.python_executable,
            model_name_or_path=args.model_name_or_path,
            max_steps=args.max_steps,
            kl_coef=args.kl_coef,
            schema_prefix_mode=args.schema_prefix_mode,
            seed=args.seed,
            report_out=Path(args.report_out) if args.report_out is not None else None,
        )
        print(json.dumps(_training_runtime_summary(report), indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    elif args.command == "train" and args.train_command == "package":
        manifest = package_training_bundle(
            Path(args.training_dir),
            Path(args.out),
            env_file=Path(args.env_file) if args.env_file is not None else None,
            require_gpu=not args.allow_cpu,
            require_hf_token=args.require_hf_token,
            python_executable=args.python_executable,
        )
        print(json.dumps(_summary(manifest), indent=2, sort_keys=True))
        return 0 if manifest["passed"] else 1
    elif args.command == "train" and args.train_command == "report":
        manifest = generate_training_diagnostics(
            Path(args.run_dir),
            out_path=Path(args.out) if args.out is not None else None,
        )
        print(json.dumps(_summary(manifest), indent=2, sort_keys=True))
        return 0
    elif args.command == "train" and args.train_command == "source-freshness":
        manifest = generate_training_package_source_freshness(
            Path(args.package_dir),
            repo_root=Path(args.repo_root) if args.repo_root is not None else None,
            out_path=Path(args.out) if args.out is not None else None,
            markdown_out_path=Path(args.markdown_out) if args.markdown_out is not None else None,
        )
        print(json.dumps(_summary(manifest), indent=2, sort_keys=True))
        return 0 if manifest["all_compared_files_match"] else 1
    elif args.command == "train" and args.train_command == "contract-status":
        manifest = summarize_full_eval_contract_status(
            Path(args.run_dir),
            sft_steps=args.sft_steps,
            dpo_steps=args.dpo_steps,
            rl_steps=args.rl_steps,
            repo_root=Path(args.repo_root) if args.repo_root is not None else None,
            windows_audit_path=Path(args.windows_audit)
            if args.windows_audit is not None
            else None,
            wsl_smoke_manifest_path=Path(args.wsl_smoke_manifest)
            if args.wsl_smoke_manifest is not None
            else None,
            package_source_freshness_path=Path(args.package_source_freshness)
            if args.package_source_freshness is not None
            else None,
            human_study_suite_path=Path(args.human_study_suite)
            if args.human_study_suite is not None
            else None,
            human_study_coverage_paths=[
                Path(path) for path in (args.human_study_coverage or [])
            ],
            out_path=Path(args.out) if args.out is not None else None,
            markdown_out_path=Path(args.markdown_out) if args.markdown_out is not None else None,
        )
        print(json.dumps(_summary(manifest), indent=2, sort_keys=True))
        return 0 if manifest["passed"] else 1
    elif args.command == "train" and args.train_command == "inspect-samples":
        manifest = create_sample_inspection(
            Path(args.dataset),
            raw_candidates_path=Path(args.raw_candidates),
            repaired_candidates_path=Path(args.repaired_candidates),
            out_path=Path(args.out),
            model_name=args.model_name,
            model_or_adapter_path=Path(args.model_or_adapter_path)
            if args.model_or_adapter_path is not None
            else None,
            generation_config=_load_json_arg(args.generation_config_json),
        )
        print(json.dumps(_summary(manifest), indent=2, sort_keys=True))
        return 0
    elif args.command == "teacher" and args.teacher_command == "export":
        manifest = export_teacher_requests(
            Path(args.dataset),
            Path(args.out),
            candidates_per_unit=args.candidates_per_unit,
            models=args.model,
            max_units=args.max_units,
        )
    elif args.command == "teacher" and args.teacher_command == "ingest":
        manifest = ingest_teacher_responses(
            Path(args.dataset),
            Path(args.requests),
            Path(args.responses),
            Path(args.out),
        )
    elif args.command == "teacher" and args.teacher_command == "run-critic":
        manifest = run_critic_requests(
            Path(args.requests),
            Path(args.out),
            provider=args.provider,
            model=args.model,
            env_file=Path(args.env_file) if args.env_file is not None else None,
            max_requests=args.max_requests,
            max_input_chars=None if args.max_input_chars == 0 else args.max_input_chars,
            max_output_tokens=args.max_output_tokens,
        )
    elif args.command == "teacher" and args.teacher_command == "ingest-critic":
        manifest = ingest_critic_responses(
            Path(args.teacher_results),
            Path(args.responses),
            Path(args.out),
        )
    elif args.command == "teacher" and args.teacher_command == "run":
        manifest = run_teacher_requests(
            Path(args.requests),
            Path(args.out),
            provider=args.provider,
            model=args.model,
            env_file=Path(args.env_file) if args.env_file is not None else None,
            max_requests=args.max_requests,
            max_input_chars=None if args.max_input_chars == 0 else args.max_input_chars,
            max_output_tokens=args.max_output_tokens,
        )
    elif args.command == "teacher" and args.teacher_command == "pipeline":
        manifest = run_teacher_pipeline(
            Path(args.dataset),
            Path(args.out),
            providers=args.provider,
            models=args.model,
            candidates_per_provider=args.candidates_per_provider,
            max_units=args.max_units,
            env_file=Path(args.env_file) if args.env_file is not None else None,
            max_input_chars=None if args.max_input_chars == 0 else args.max_input_chars,
            max_output_tokens=args.max_output_tokens,
        )
    else:
        parser.error(f"unknown command {args.command!r}")

    print(json.dumps(_summary(manifest), indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-mirror",
        description="Build evidence-backed semantic IR mirror repositories.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build a whole-repo semantic mirror.")
    build.add_argument("repo", help="Repository path to analyze.")
    build.add_argument("--out", required=True, help="Output mirror repository path.")
    _add_common_options(build)

    diff = subparsers.add_parser("diff", help="Build semantic IR for changed files between refs.")
    diff.add_argument("repo", help="Repository path to analyze.")
    diff.add_argument("--base", required=True, help="Base git ref.")
    diff.add_argument("--head", required=True, help="Head git ref.")
    diff.add_argument("--out", required=True, help="Output diff mirror path.")
    _add_common_options(diff)

    score = subparsers.add_parser("score", help="Score a generated semantic mirror.")
    score.add_argument("mirror", help="Mirror repository path containing manifest.json.")
    score.add_argument(
        "--repo",
        help="Source repository path. Defaults to the repo path stored in manifest.json.",
    )

    corpus = subparsers.add_parser(
        "corpus",
        help="Collect multi-repository Data/ML corpora for training.",
    )
    corpus_subparsers = corpus.add_subparsers(dest="corpus_command", required=True)
    collect = corpus_subparsers.add_parser(
        "collect",
        help="Clone or reference repositories, sample each, and emit an aggregate dataset.",
    )
    collect.add_argument(
        "--repo",
        action="append",
        required=True,
        help="Local path or Git URL. Use name=path_or_url to choose a stable corpus id.",
    )
    collect.add_argument("--out", required=True, help="Output corpus directory.")
    collect.add_argument("--max-units-per-repo", type=int, default=100)
    collect.add_argument("--review-budget-per-repo", type=int, default=25)
    collect.add_argument("--hard-negatives-per-unit", type=int, default=2)
    _add_common_options(collect)

    dataset = subparsers.add_parser(
        "dataset",
        help="Create training and curation datasets from static semantic IR.",
    )
    dataset_subparsers = dataset.add_subparsers(dest="dataset_command", required=True)
    sample = dataset_subparsers.add_parser(
        "sample",
        help="Generate silver examples, hard negatives, and a review queue.",
    )
    sample.add_argument("repo", help="Repository path to sample.")
    sample.add_argument("--out", required=True, help="Output dataset directory.")
    sample.add_argument("--max-units", type=int, default=200, help="Maximum silver units.")
    sample.add_argument(
        "--review-budget",
        type=int,
        default=50,
        help="Maximum units placed in the prioritized review queue.",
    )
    sample.add_argument(
        "--hard-negatives-per-unit",
        type=int,
        default=2,
        help="Maximum deterministic hard negatives generated per silver unit.",
    )
    _add_common_options(sample)

    promote_gold = dataset_subparsers.add_parser(
        "promote-gold",
        help="Promote reviewed silver or review-queue records into gold.jsonl.",
    )
    promote_gold.add_argument("dataset", help="Dataset directory containing manifest.json.")
    promote_gold.add_argument(
        "--record-id",
        action="append",
        required=True,
        help="Silver record id, review queue record id, or silver unit id. May be repeated.",
    )
    promote_gold.add_argument(
        "--label",
        action="append",
        default=[],
        help="Curation label to attach to promoted gold records. May be repeated.",
    )
    promote_gold.add_argument("--reviewer", help="Reviewer identifier.")
    promote_gold.add_argument("--notes", help="Short curation note.")

    evaluation = subparsers.add_parser("eval", help="Evaluate mirrors, datasets, and regressions.")
    eval_subparsers = evaluation.add_subparsers(dest="eval_command", required=True)

    eval_mirror = eval_subparsers.add_parser("mirror", help="Evaluate mirror quality gates.")
    eval_mirror.add_argument("mirror", help="Mirror repository path containing manifest.json.")
    eval_mirror.add_argument(
        "--repo",
        help="Source repository path. Defaults to the repo path stored in manifest.json.",
    )
    eval_mirror.add_argument("--out", help="Optional JSON report output path.")

    eval_dataset = eval_subparsers.add_parser("dataset", help="Evaluate dataset curation gates.")
    eval_dataset.add_argument("dataset", help="Dataset directory containing manifest.json.")
    eval_dataset.add_argument("--out", help="Optional JSON report output path.")

    eval_compare = eval_subparsers.add_parser(
        "compare",
        help="Compare current and baseline evaluation reports for regression gates.",
    )
    eval_compare.add_argument("baseline_report", help="Baseline evaluation report JSON.")
    eval_compare.add_argument("current_report", help="Current evaluation report JSON.")
    eval_compare.add_argument(
        "--max-score-drop",
        type=float,
        default=0.01,
        help="Maximum allowed score drop fraction.",
    )
    eval_compare.add_argument("--out", help="Optional JSON report output path.")

    eval_candidates = eval_subparsers.add_parser(
        "candidates",
        help="Evaluate model-generated SIR candidate JSONL against a held-out dataset.",
    )
    eval_candidates.add_argument("dataset", help="Dataset directory containing manifest.json.")
    eval_candidates.add_argument("--candidates", required=True, help="Candidate JSONL path.")
    eval_candidates.add_argument("--model-name", required=True, help="Name of evaluated model/run.")
    eval_candidates.add_argument("--out", help="Optional JSON report output path.")

    eval_model_compare = eval_subparsers.add_parser(
        "model-compare",
        help="Compare teacher/SFT/RL model evaluation reports against training gates.",
    )
    eval_model_compare.add_argument("baseline_report", help="Baseline model evaluation report JSON.")
    eval_model_compare.add_argument("current_report", help="Current model evaluation report JSON.")
    eval_model_compare.add_argument("--stage", required=True, choices=["sft", "dpo", "rl"])
    eval_model_compare.add_argument(
        "--min-score-improvement",
        type=float,
        default=0.0,
        help="Required average static faithfulness score improvement.",
    )
    eval_model_compare.add_argument("--out", help="Optional JSON report output path.")

    eval_review_pack = eval_subparsers.add_parser(
        "review-pack",
        help="Evaluate reviewer question/change packs against human usefulness gates.",
    )
    eval_review_pack.add_argument("review_pack", help="Review pack directory.")
    eval_review_pack.add_argument("--mirror", help="Optional source mirror path override.")
    eval_review_pack.add_argument("--out", help="Optional JSON report output path.")

    eval_human_study = eval_subparsers.add_parser(
        "human-study",
        help="Evaluate timed source-only versus mirror-first human study answers.",
    )
    eval_human_study.add_argument("study", help="Human usefulness study directory.")
    eval_human_study.add_argument("--answers", required=True, help="Completed answers JSONL path.")
    eval_human_study.add_argument(
        "--min-accuracy",
        type=float,
        default=0.8,
        help="Minimum reviewer-scored mirror answer accuracy.",
    )
    eval_human_study.add_argument(
        "--min-speedup",
        type=float,
        default=1.0,
        help="Minimum source-median/mirror-median speed ratio.",
    )
    eval_human_study.add_argument("--out", help="Optional JSON report output path.")

    eval_human_study_suite = eval_subparsers.add_parser(
        "human-study-suite",
        help="Summarize whole-repo and diff-mode Phase 6 human-study evaluation reports.",
    )
    eval_human_study_suite.add_argument(
        "--report",
        action="append",
        required=True,
        help="A JSON report written by `eval human-study`; repeat for whole-repo and diff-mode.",
    )
    eval_human_study_suite.add_argument("--out", help="Optional JSON summary output path.")

    review = subparsers.add_parser(
        "review",
        help="Create reviewer-facing question and behavior-change packs.",
    )
    review_subparsers = review.add_subparsers(dest="review_command", required=True)
    review_pack = review_subparsers.add_parser(
        "pack",
        help="Generate evidence-backed review questions and diff behavior tasks.",
    )
    review_pack.add_argument("mirror", help="Mirror repository path containing manifest.json.")
    review_pack.add_argument("--out", required=True, help="Output review pack directory.")
    review_pack.add_argument("--max-questions", type=int, default=25)
    review_pack.add_argument("--max-change-tasks", type=int, default=25)

    review_study = review_subparsers.add_parser(
        "study",
        help="Create timed source-only versus mirror-first human usefulness study tasks.",
    )
    review_study.add_argument("review_pack", help="Review pack directory.")
    review_study.add_argument("--out", required=True, help="Output human study directory.")

    review_conduct = review_subparsers.add_parser(
        "conduct-study",
        help="Interactively run timed human study tasks and write answers JSONL.",
    )
    review_conduct.add_argument("study", help="Human usefulness study directory.")
    review_conduct.add_argument("--out", required=True, help="Output completed answers JSONL path.")
    review_conduct.add_argument("--reviewer", required=True, help="Reviewer name or stable handle.")
    review_conduct.add_argument(
        "--task-set",
        choices=("all", "source", "mirror", "visibility"),
        default="all",
        help="Subset of study tasks to conduct.",
    )
    review_conduct.add_argument("--max-tasks", type=int, help="Optional cap for this session.")
    review_conduct.add_argument(
        "--append",
        action="store_true",
        help="Append unanswered tasks when the answers file already exists.",
    )
    review_conduct.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing answers file instead of refusing to run.",
    )
    review_status = review_subparsers.add_parser(
        "study-status",
        help="Report Phase 6 human-study answer coverage before evaluation.",
    )
    review_status.add_argument("study", help="Human usefulness study directory.")
    review_status.add_argument("--answers", help="Completed answers JSONL path.")
    review_status.add_argument("--out", help="Optional JSON coverage report path.")
    review_collection_plan = review_subparsers.add_parser(
        "study-collection-plan",
        help="Write a Phase 6 real-answer collection command plan for one or more studies.",
    )
    review_collection_plan.add_argument(
        "--study",
        action="append",
        required=True,
        help="Labeled study path as label=path, for example whole_repo=path. May be repeated.",
    )
    review_collection_plan.add_argument(
        "--answers-dir",
        required=True,
        help="Directory for real answer, coverage, eval, and suite files.",
    )
    review_collection_plan.add_argument(
        "--reviewer",
        required=True,
        help="Reviewer name or stable handle to embed in conduct-study commands.",
    )
    review_collection_plan.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Suggested --max-tasks value for resumable conduct-study sessions.",
    )
    review_collection_plan.add_argument("--out", help="Optional JSON plan output path.")

    train = subparsers.add_parser("train", help="Prepare SFT and RL training artifacts.")
    train_subparsers = train.add_subparsers(dest="train_command", required=True)
    prepare = train_subparsers.add_parser(
        "prepare",
        help="Build SFT, contrastive, preference, and RL prompt files from a dataset batch.",
    )
    prepare.add_argument("dataset", help="Dataset directory generated by `dataset sample`.")
    prepare.add_argument("--out", required=True, help="Output training batch directory.")
    prepare.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base Qwen3-family model id to place in the Unsloth config.",
    )
    prepare.add_argument("--max-records", type=int, help="Optional cap on positive records.")
    prepare.add_argument(
        "--teacher-results",
        help=(
            "Optional teacher ingest or pipeline directory containing "
            "teacher_preference_pairs.jsonl."
        ),
    )
    prepare.add_argument(
        "--gold-only",
        action="store_true",
        help="When gold records exist, exclude silver records from positive SFT data.",
    )
    validate_train = train_subparsers.add_parser(
        "validate",
        help="Validate prepared SFT/RL training artifacts before launching GPU training.",
    )
    validate_train.add_argument("training_dir", help="Training batch directory.")
    validate_train.add_argument("--out", help="Optional full JSON validation report path.")
    audit_train = train_subparsers.add_parser(
        "audit",
        help="Check local CUDA/dependency readiness for Unsloth/TRL training.",
    )
    audit_train.add_argument("training_dir", help="Training batch directory.")
    audit_train.add_argument("--out", help="Optional full JSON audit report path.")
    _add_training_runtime_options(audit_train)

    run_sft = train_subparsers.add_parser(
        "run-sft",
        help="Launch the generated Unsloth SFT script after readiness gates pass.",
    )
    run_sft.add_argument("training_dir", help="Training batch directory.")
    run_sft.add_argument("--output-dir", required=True, help="Model output directory.")
    run_sft.add_argument("--max-steps", type=int, help="Optional cap on SFT update steps.")
    run_sft.add_argument("--resume-from-checkpoint", help="Optional SFT checkpoint path.")
    run_sft.add_argument("--seed", type=int, help="Deterministic SFT seed.")
    _add_training_launch_options(run_sft)

    run_dpo = train_subparsers.add_parser(
        "run-dpo",
        help="Launch the generated TRL DPO script after readiness gates pass.",
    )
    run_dpo.add_argument("training_dir", help="Training batch directory.")
    run_dpo.add_argument("--output-dir", required=True, help="Model output directory.")
    run_dpo.add_argument("--model-name-or-path", help="SFT model or adapter path for DPO.")
    run_dpo.add_argument("--beta", type=float, default=0.1, help="DPO beta value.")
    run_dpo.add_argument("--max-steps", type=int, help="Optional cap on DPO update steps.")
    run_dpo.add_argument("--resume-from-checkpoint", help="Optional DPO checkpoint path.")
    run_dpo.add_argument("--seed", type=int, help="Deterministic DPO seed.")
    _add_training_launch_options(run_dpo)

    run_rl = train_subparsers.add_parser(
        "run-rl",
        help="Launch deterministic-reward policy optimization after readiness gates pass.",
    )
    run_rl.add_argument("training_dir", help="Training batch directory.")
    run_rl.add_argument("--output-dir", required=True, help="Model output directory.")
    run_rl.add_argument("--model-name-or-path", help="DPO/SFT model or adapter path for RL.")
    run_rl.add_argument("--max-steps", type=int, help="Optional cap on RL update steps.")
    run_rl.add_argument("--kl-coef", type=float, default=0.05, help="Length/KL-style penalty weight.")
    run_rl.add_argument(
        "--schema-prefix-mode",
        choices=["off", "identity", "identity-algorithm", "schema-scaffold"],
        default="schema-scaffold",
        help="Constrain RL generations with an explicit SIR schema prefix.",
    )
    run_rl.add_argument("--seed", type=int, help="Deterministic RL seed.")
    _add_training_launch_options(run_rl)

    package_train = train_subparsers.add_parser(
        "package",
        help="Create a portable Linux/WSL/Colab handoff bundle for prepared training data.",
    )
    package_train.add_argument("training_dir", help="Training batch directory.")
    package_train.add_argument("--out", required=True, help="Output training bundle directory.")
    _add_training_runtime_options(package_train)

    report_train = train_subparsers.add_parser(
        "report",
        help="Generate JSON, Markdown, and PNG diagnostics from a training run directory.",
    )
    report_train.add_argument("run_dir", help="Training run or outputs directory.")
    report_train.add_argument("--out", help="Diagnostics output directory. Defaults to run_dir/diagnostics.")

    source_freshness = train_subparsers.add_parser(
        "source-freshness",
        help="Hash-compare packaged runtime source against the current repository source.",
    )
    source_freshness.add_argument("package_dir", help="Training package directory.")
    source_freshness.add_argument(
        "--repo-root",
        help="Repository root containing src/semantic_mirror. Defaults to the current working directory.",
    )
    source_freshness.add_argument(
        "--out",
        help="Optional JSON output path. Defaults to package_dir/source_freshness.json.",
    )
    source_freshness.add_argument(
        "--markdown-out",
        help="Optional Markdown output path. Defaults to package_dir/source_freshness.md.",
    )

    contract_status = train_subparsers.add_parser(
        "contract-status",
        help="Summarize which full-eval contract gates are proven by an outputs directory.",
    )
    contract_status.add_argument("run_dir", help="Full-eval outputs directory.")
    contract_status.add_argument("--sft-steps", type=int, help="Expected SFT max_steps.")
    contract_status.add_argument("--dpo-steps", type=int, help="Expected DPO max_steps.")
    contract_status.add_argument("--rl-steps", type=int, help="Expected RL max_steps.")
    contract_status.add_argument(
        "--repo-root",
        help="Optional repository root for git hygiene evidence in the contract scorecard.",
    )
    contract_status.add_argument(
        "--windows-audit",
        help="Optional Windows-native training audit JSON for readiness evidence.",
    )
    contract_status.add_argument(
        "--wsl-smoke-manifest",
        help="Optional Windows-hosted WSL smoke-chain manifest JSON for fallback readiness evidence.",
    )
    contract_status.add_argument(
        "--package-source-freshness",
        help="Optional source_freshness JSON proving bundled runtime source matches the repo.",
    )
    contract_status.add_argument(
        "--human-study-suite",
        help="Optional Phase 6 human-study-suite JSON summary for usefulness evidence.",
    )
    contract_status.add_argument(
        "--human-study-coverage",
        action="append",
        help="Optional review study-status JSON coverage report. May be repeated.",
    )
    contract_status.add_argument("--out", help="Optional JSON status report output path.")
    contract_status.add_argument("--markdown-out", help="Optional Markdown status report output path.")

    inspect_samples = train_subparsers.add_parser(
        "inspect-samples",
        help="Build raw-vs-repaired sample eval and inspection artifacts.",
    )
    inspect_samples.add_argument("dataset", help="Held-out dataset directory.")
    inspect_samples.add_argument("--raw-candidates", required=True, help="Raw candidate JSONL.")
    inspect_samples.add_argument(
        "--repaired-candidates",
        required=True,
        help="Repaired candidate JSONL.",
    )
    inspect_samples.add_argument("--out", required=True, help="Sample artifact output directory.")
    inspect_samples.add_argument("--model-name", required=True, help="Model/run name.")
    inspect_samples.add_argument("--model-or-adapter-path", help="Model or adapter path.")
    inspect_samples.add_argument(
        "--generation-config-json",
        help="Inline JSON object recording generation settings.",
    )

    teacher = subparsers.add_parser(
        "teacher",
        help="Export teacher requests and ingest teacher model responses.",
    )
    teacher_subparsers = teacher.add_subparsers(dest="teacher_command", required=True)
    teacher_export = teacher_subparsers.add_parser(
        "export",
        help="Create candidate-generation requests for frontier teacher models.",
    )
    teacher_export.add_argument("dataset", help="Dataset directory generated by `dataset sample`.")
    teacher_export.add_argument("--out", required=True, help="Output teacher request directory.")
    teacher_export.add_argument(
        "--candidates-per-unit",
        type=int,
        default=3,
        help="Number of teacher candidate requests to emit per source unit.",
    )
    teacher_export.add_argument(
        "--model",
        action="append",
        help="Teacher model id. May be passed multiple times.",
    )
    teacher_export.add_argument("--max-units", type=int, help="Optional source unit cap.")

    teacher_ingest = teacher_subparsers.add_parser(
        "ingest",
        help="Validate teacher responses and emit critic requests and preferences.",
    )
    teacher_ingest.add_argument("dataset", help="Dataset directory generated by `dataset sample`.")
    teacher_ingest.add_argument("--requests", required=True, help="candidate_requests.jsonl path.")
    teacher_ingest.add_argument("--responses", required=True, help="Teacher response JSONL path.")
    teacher_ingest.add_argument("--out", required=True, help="Output teacher ingest directory.")

    critic_run = teacher_subparsers.add_parser(
        "run-critic",
        help="Call an API provider for critic_requests.jsonl and write critique JSONL.",
    )
    critic_run.add_argument("--requests", required=True, help="critic_requests.jsonl path.")
    critic_run.add_argument("--out", required=True, help="Output critic response JSONL path.")
    critic_run.add_argument(
        "--provider",
        default="openai",
        choices=SUPPORTED_TEACHER_PROVIDERS,
        help="Teacher API provider.",
    )
    critic_run.add_argument("--model", help="Provider model id. Defaults to .env or package default.")
    critic_run.add_argument(
        "--env-file",
        help="Optional .env path. Defaults to .env in the current working directory.",
    )
    critic_run.add_argument("--max-requests", type=int, help="Optional request cap.")
    critic_run.add_argument(
        "--max-input-chars",
        type=int,
        default=24000,
        help="Maximum combined prompt characters sent per request. Use 0 to disable.",
    )
    critic_run.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Maximum output tokens requested from the provider.",
    )

    critic_ingest = teacher_subparsers.add_parser(
        "ingest-critic",
        help="Ingest critic responses into structured error labels and review queues.",
    )
    critic_ingest.add_argument(
        "teacher_results",
        help="Teacher ingest or pipeline directory containing candidate_results.jsonl.",
    )
    critic_ingest.add_argument("--responses", required=True, help="Critic response JSONL path.")
    critic_ingest.add_argument("--out", required=True, help="Output critic labels directory.")

    teacher_run = teacher_subparsers.add_parser(
        "run",
        help="Call an API provider for exported teacher requests and write response JSONL.",
    )
    teacher_run.add_argument("--requests", required=True, help="candidate_requests.jsonl path.")
    teacher_run.add_argument("--out", required=True, help="Output teacher response JSONL path.")
    teacher_run.add_argument(
        "--provider",
        default="openai",
        choices=SUPPORTED_TEACHER_PROVIDERS,
        help="Teacher API provider.",
    )
    teacher_run.add_argument("--model", help="Provider model id. Defaults to .env or package default.")
    teacher_run.add_argument(
        "--env-file",
        help="Optional .env path. Defaults to .env in the current working directory.",
    )
    teacher_run.add_argument("--max-requests", type=int, help="Optional request cap.")
    teacher_run.add_argument(
        "--max-input-chars",
        type=int,
        default=24000,
        help="Maximum combined prompt characters sent per request. Use 0 to disable.",
    )
    teacher_run.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="Maximum output tokens requested from the provider.",
    )

    teacher_pipeline = teacher_subparsers.add_parser(
        "pipeline",
        help="Export, run, combine, and ingest teacher candidates across providers.",
    )
    teacher_pipeline.add_argument("dataset", help="Dataset directory generated by `dataset sample`.")
    teacher_pipeline.add_argument("--out", required=True, help="Output teacher pipeline directory.")
    teacher_pipeline.add_argument(
        "--provider",
        action="append",
        choices=SUPPORTED_TEACHER_PROVIDERS,
        help="Teacher API provider. May be passed multiple times. Defaults to openai.",
    )
    teacher_pipeline.add_argument(
        "--model",
        action="append",
        help="Provider model override in the same order as --provider. May be passed multiple times.",
    )
    teacher_pipeline.add_argument(
        "--candidates-per-provider",
        type=int,
        default=1,
        help="Candidate requests emitted per source unit for each provider.",
    )
    teacher_pipeline.add_argument("--max-units", type=int, help="Optional source unit cap.")
    teacher_pipeline.add_argument(
        "--env-file",
        help="Optional .env path. Defaults to .env in the current working directory.",
    )
    teacher_pipeline.add_argument(
        "--max-input-chars",
        type=int,
        default=24000,
        help="Maximum combined prompt characters sent per request. Use 0 to disable.",
    )
    teacher_pipeline.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="Maximum output tokens requested from each provider.",
    )
    return parser


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=sorted(SUPPORTED_PROFILES),
        default="data_ml",
        help="Semantic profile to apply.",
    )
    parser.add_argument(
        "--zoom",
        choices=sorted(SUPPORTED_ZOOMS),
        default="L2",
        help="Semantic zoom level.",
    )


def _add_training_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Do not require torch CUDA availability in readiness gates.",
    )
    parser.add_argument(
        "--require-hf-token",
        action="store_true",
        help="Require HF_TOKEN or HUGGINGFACE_HUB_TOKEN to be present.",
    )
    parser.add_argument(
        "--env-file",
        help="Optional .env path. Defaults to .env in the current working directory.",
    )
    parser.add_argument(
        "--python-executable",
        help="Python executable to audit or use for generated training scripts.",
    )


def _add_training_launch_options(parser: argparse.ArgumentParser) -> None:
    _add_training_runtime_options(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the gated launch command without starting training.",
    )
    parser.add_argument("--report-out", help="Optional full launch JSON report path.")


def _summary(manifest: dict[str, object]) -> dict[str, object]:
    if manifest.get("mode") == "dataset_sample":
        return {
            "mode": manifest["mode"],
            "repo": manifest["repo"],
            "profile": manifest["profile"],
            "zoom": manifest["zoom"],
            "counts": manifest["counts"],
            "curation_budget": manifest["curation_budget"],
        }
    if manifest.get("mode") == "dataset_gold_promote":
        return {
            "mode": manifest["mode"],
            "dataset": manifest["dataset"],
            "requested": manifest["requested"],
            "promoted": manifest["promoted"],
            "missing": manifest["missing"],
            "gold_records": manifest["gold_records"],
            "passed": manifest["passed"],
        }
    if manifest.get("mode") == "training_prepare":
        return {
            "mode": manifest["mode"],
            "dataset": manifest["dataset"],
            "base_model": manifest["base_model"],
            "input_counts": manifest["input_counts"],
            "output_counts": manifest["output_counts"],
        }
    if manifest.get("mode") == "training_package":
        return {
            "mode": manifest["mode"],
            "out": manifest["out"],
            "passed": manifest["passed"],
            "current_runtime_ready": manifest.get("current_runtime_ready"),
            "current_runtime_failed_checks": manifest.get("current_runtime_failed_checks", []),
            "output_counts": manifest.get("output_counts", {}),
            "files": manifest["files"],
        }
    if manifest.get("mode") == "training_diagnostics":
        return {
            "mode": manifest["mode"],
            "run_dir": manifest["run_dir"],
            "diagnostics_dir": manifest["diagnostics_dir"],
            "missing_metrics": manifest["missing_metrics"],
            "plots": {
                name: {"path": plot["path"], "points": plot["points"], "missing": plot["missing"]}
                for name, plot in manifest["plots"].items()
            },
        }
    if manifest.get("mode") == "semantic_mirror_package_source_freshness":
        return {
            "mode": manifest["mode"],
            "package_root": manifest["package_root"],
            "repo_root": manifest["repo_root"],
            "git_commit": manifest["git_commit"],
            "compared_file_count": manifest["compared_file_count"],
            "all_compared_files_match": manifest["all_compared_files_match"],
            "mismatched_files": [
                row["relative_path"]
                for row in manifest["comparisons"]
                if not row["match"]
            ],
            "files": manifest["files"],
        }
    if manifest.get("mode") == "sample_inspection":
        return {
            "mode": manifest["mode"],
            "dataset": manifest["dataset"],
            "model_name": manifest["model_name"],
            "raw_parseability_count": manifest["raw_parseability_count"],
            "raw_generation_cap_hits": manifest["raw_generation_cap_hits"],
            "raw_schema_validity_count": manifest["raw_schema_validity_count"],
            "repaired_schema_validity_count": manifest["repaired_schema_validity_count"],
            "files": manifest["files"],
        }
    if manifest.get("mode") == "corpus_collect":
        return {
            "mode": manifest["mode"],
            "out": manifest["out"],
            "repo_count": manifest["repo_count"],
            "successful_repos": manifest["successful_repos"],
            "failed_repos": manifest["failed_repos"],
            "aggregate_dataset": manifest["aggregate_dataset"],
            "aggregate_counts": manifest["aggregate_counts"],
        }
    if manifest.get("mode") == "review_pack":
        return {
            "mode": manifest["mode"],
            "mirror": manifest["mirror"],
            "source_mirror_mode": manifest["source_mirror_mode"],
            "counts": manifest["counts"],
            "files": manifest["files"],
        }
    if manifest.get("mode") == "human_usefulness_study":
        return {
            "mode": manifest["mode"],
            "review_pack": manifest["review_pack"],
            "source_mirror_mode": manifest["source_mirror_mode"],
            "counts": manifest["counts"],
            "files": manifest["files"],
        }
    if manifest.get("mode") == "human_usefulness_study_conduct":
        return {
            "mode": manifest["mode"],
            "study": manifest["study"],
            "answers": manifest["answers"],
            "reviewer": manifest["reviewer"],
            "task_set": manifest["task_set"],
            "requested_tasks": manifest["requested_tasks"],
            "skipped_existing": manifest["skipped_existing"],
            "completed_records": manifest["completed_records"],
            "answer_records": manifest["answer_records"],
        }
    if manifest.get("mode") == "full_eval_contract_status":
        return {
            "mode": manifest["mode"],
            "run_dir": manifest["run_dir"],
            "passed": manifest["passed"],
            "requested_max_steps": manifest["requested_max_steps"],
            "remaining_items": manifest["remaining_items"],
            "gates": manifest["gates"],
        }
    if manifest.get("mode") in {
        "teacher_export",
        "teacher_ingest",
        "teacher_run",
        "teacher_pipeline",
        "critic_run",
        "critic_ingest",
    }:
        return {
            "mode": manifest["mode"],
            "dataset": manifest.get("dataset"),
            "counts": manifest.get("counts", manifest.get("request_counts")),
            "files": manifest["files"],
        }
    return {
        "mode": manifest["mode"],
        "repo": manifest["repo"],
        "profile": manifest["profile"],
        "zoom": manifest["zoom"],
        "coverage": manifest["coverage"],
        "confidence": manifest["confidence"],
    }


def _eval_summary(report: dict[str, object]) -> dict[str, object]:
    if "gates" not in report:
        return {
            "mode": report["mode"],
            "passed": report["passed"],
            "issues": report.get("issues", []),
        }
    summary = {
        "mode": report["mode"],
        "passed": report["passed"],
        "gates": report["gates"],
    }
    if "phase6_gate_summary" in report:
        summary["phase6_gate_summary"] = report["phase6_gate_summary"]
    return summary


def _training_runtime_summary(report: dict[str, object]) -> dict[str, object]:
    if report["mode"] == "training_launch":
        return {
            "mode": report["mode"],
            "stage": report["stage"],
            "passed": report["passed"],
            "dry_run": report["dry_run"],
            "launched": report["launched"],
            "would_launch": report["would_launch"],
            "reason": report.get("reason"),
            "command": report["command"],
            "failed_checks": _failed_training_checks(report["audit"]),
            "logs": report.get("logs"),
        }
    return {
        "mode": report["mode"],
        "passed": report["passed"],
        "ready_to_launch": report["ready_to_launch"],
        "require_gpu": report["require_gpu"],
        "failed_checks": _failed_training_checks(report),
        "blocker": report.get("blocker"),
        "repro": report.get("repro"),
        "command_hints": report["command_hints"],
    }


def _failed_training_checks(report: object) -> list[dict[str, object]]:
    if not isinstance(report, dict):
        return []
    checks = report.get("checks", [])
    if not isinstance(checks, list):
        return []
    return [
        {
            "name": check.get("name"),
            "required": check.get("required"),
            "actual": check.get("actual"),
        }
        for check in checks
        if isinstance(check, dict) and not check.get("passed")
    ]


def _load_json_arg(value: str | None) -> dict[str, object] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--generation-config-json must be a JSON object")
    return parsed


def _parse_labeled_paths(values: list[str]) -> dict[str, Path]:
    labeled: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--study values must use label=path")
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError("--study label must not be empty")
        if label in labeled:
            raise ValueError(f"duplicate --study label: {label}")
        labeled[label] = Path(path)
    return labeled


if __name__ == "__main__":
    raise SystemExit(main())
