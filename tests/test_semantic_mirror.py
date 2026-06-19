from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

from semantic_mirror.builder import build_repository, diff_repository
from semantic_mirror.corpus import collect_corpus
from semantic_mirror.dataset import promote_gold_records, sample_dataset
from semantic_mirror.evaluation import (
    compare_model_evaluations,
    compare_regression_reports,
    evaluate_dataset,
    evaluate_mirror,
    evaluate_model_candidates,
)
from semantic_mirror.rewards import score_document, score_mirror
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
from semantic_mirror.schema import validate_ir_document, validate_manifest, validate_unit
from semantic_mirror.teacher import (
    export_teacher_requests,
    ingest_critic_responses,
    ingest_teacher_responses,
    run_critic_requests,
    run_teacher_pipeline,
    run_teacher_requests,
)
from semantic_mirror.training import (
    DEFAULT_BASE_MODEL,
    REQUIRED_TRAINING_MODULES,
    audit_training_environment,
    create_sample_inspection,
    generate_training_diagnostics,
    generate_training_package_source_freshness,
    inspect_full_training_eval_resume,
    launch_training_job,
    package_training_bundle,
    prepare_training_data,
    summarize_full_eval_contract_status,
    validate_training_batch,
)


def _test_command_manifest(commands: dict[str, str]) -> dict[str, dict[str, object]]:
    training_commands = {
        "wsl_smoke_chain",
        "sft",
        "dpo",
        "rl",
        "full_training_eval",
        "smoke_chain",
    }
    categories = {
        "wsl_smoke_chain": "training",
        "sft": "training",
        "dpo": "training",
        "rl": "training",
        "full_training_eval": "training",
        "smoke_chain": "training",
        "inspect_full_training_eval_resume": "inspection",
        "inspect_resume": "inspection",
        "contract_status": "status",
        "source_freshness": "status",
        "preflight_full_eval_inputs": "validation",
        "preflight_wsl_smoke_inputs": "validation",
        "report": "diagnostics",
        "validate": "validation",
        "audit": "validation",
        "install": "setup",
        "bootstrap_linux_cuda": "setup",
        "bootstrap_wsl_ubuntu": "setup",
        "generate_candidates": "generation",
        "score_candidates": "evaluation",
        "eval_candidates": "evaluation",
        "inspect_samples": "inspection",
        "compare_sft": "evaluation",
        "compare_sft_raw": "evaluation",
        "compare_dpo": "evaluation",
        "compare_dpo_raw": "evaluation",
        "compare_rl": "evaluation",
        "compare_rl_raw": "evaluation",
    }
    required_inputs = {
        "validate": ["training_dir"],
        "audit": ["training_dir"],
        "wsl_smoke_chain": ["held_out_dataset"],
        "preflight_wsl_smoke_inputs": ["held_out_dataset"],
        "sft": ["training_dir", "output_dir"],
        "dpo": ["training_dir", "sft_model_or_adapter", "output_dir"],
        "rl": ["training_dir", "dpo_or_sft_model_or_adapter", "output_dir"],
        "full_training_eval": ["held_out_dataset", "baseline_candidates"],
        "preflight_full_eval_inputs": ["held_out_dataset", "baseline_candidates"],
        "smoke_chain": ["held_out_dataset"],
        "inspect_full_training_eval_resume": ["outputs_dir"],
        "inspect_resume": ["outputs_dir"],
        "generate_candidates": ["model_or_adapter", "prompt_file"],
        "score_candidates": ["candidates_jsonl"],
        "eval_candidates": ["model_or_adapter", "held_out_dataset"],
        "inspect_samples": ["samples_dir"],
        "report": ["outputs_dir"],
        "source_freshness": ["package_root"],
        "contract_status": ["outputs_dir"],
        "compare_sft": ["baseline_candidates", "sft_candidates"],
        "compare_sft_raw": ["baseline_candidates", "sft_raw_candidates"],
        "compare_dpo": ["sft_candidates", "dpo_candidates"],
        "compare_dpo_raw": ["sft_raw_candidates", "dpo_raw_candidates"],
        "compare_rl": ["sft_candidates", "rl_candidates"],
        "compare_rl_raw": ["sft_raw_candidates", "rl_raw_candidates"],
    }
    optional_inputs = {
        "source_freshness": ["repo_root"],
        "contract_status": [
            "repo_root",
            "windows_audit",
            "wsl_smoke_manifest",
            "package_source_freshness",
            "human_study_coverage",
            "human_study_suite",
        ],
        "full_training_eval": [
            "source_freshness_repo_root",
            "windows_audit",
            "wsl_smoke_manifest",
            "package_source_freshness",
            "human_study_coverage",
            "human_study_suite",
        ],
        "inspect_full_training_eval_resume": [
            "sft_resume_from_checkpoint",
            "dpo_resume_from_checkpoint",
        ],
        "inspect_resume": [
            "sft_resume_from_checkpoint",
            "dpo_resume_from_checkpoint",
        ],
    }
    return {
        name: {
            "command": command,
            "category": categories[name],
            "launches_training": name in training_commands,
            "required_inputs": required_inputs.get(name, []),
            "optional_inputs": optional_inputs.get(name, []),
        }
        for name, command in commands.items()
    }


SAMPLE_TRAIN = """\
import torch
from pathlib import Path


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 2)

    def forward(self, batch):
        return self.layer(batch.float())


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch, target in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    torch.save(model.state_dict(), Path("model.pt"))
    return total_loss
"""


COMPLEX_GOLDEN = """\
from pathlib import Path

GLOBAL_STATE = {}


def validate_and_record(path, values, cache):
    if not values:
        raise ValueError("empty values")
    total = 0
    try:
        with open(path, "w", encoding="utf-8") as handle:
            for key, value in values.items():
                if value < 0:
                    raise RuntimeError("negative value")
                total += value
                cache[key] = total
                cache.last_total = total
                handle.write(f"{key}:{value}\\n")
    except OSError as exc:
        GLOBAL_STATE["last_error"] = str(exc)
        return None
    Path(path).write_text(str(total), encoding="utf-8")
    return total


def train_epoch(model, loader, optimizer, scheduler, criterion, metric):
    model.train()
    running = []
    for batch, target in loader:
        optimizer.zero_grad()
        output = model(batch.float())
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running.append(loss.item())
    checkpoint = {"model": model.state_dict(), "metric": metric(output, target)}
    torch.save(checkpoint, "checkpoint.pt")
    return sum(running) / len(running)
"""


def test_build_generates_path_preserving_ir_with_data_ml_details(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    out = tmp_path / "mirror"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "train.py").write_text(SAMPLE_TRAIN, encoding="utf-8")

    manifest = build_repository(repo, out, profile="data_ml", zoom="L4")

    validate_manifest(manifest)
    assert (out / "manifest.json").exists()
    json_path = out / "src" / "train.py.sir.json"
    md_path = out / "src" / "train.py.sir.md"
    assert json_path.exists()
    assert md_path.exists()

    document = json.loads(json_path.read_text(encoding="utf-8"))
    validate_ir_document(document)
    assert document["source_path"] == "src/train.py"
    assert document["static_analysis"]["backend"] == "tree_sitter_python"
    assert document["static_analysis"]["available"]
    assert manifest["static_analysis"]["parser_backends"]["tree_sitter_python"] == 1
    assert manifest["coverage"]["claim_evidence_coverage"] == 1.0
    assert manifest["coverage"]["generated_units"] >= 4

    train_unit = next(unit for unit in document["units"] if unit["qualified_name"] == "train")
    calls = {claim["name"] for claim in train_unit["calls"]}
    assert "optimizer.step" in calls
    assert "optimizer.zero_grad" in calls
    assert "torch.save" in calls
    assert train_unit["returns"]
    assert train_unit["state_mutations"]
    assert train_unit["data_ml_details"]["training_loops"]
    assert train_unit["data_ml_details"]["checkpointing"]
    assert "optimizer_scheduler" in train_unit["data_ml_details"]

    markdown = md_path.read_text(encoding="utf-8")
    assert "Semantic IR: `src/train.py`" in markdown
    assert "Data/ML Details" in markdown

    reward_report = score_mirror(out, repo_path=repo)
    assert reward_report["score"] > 0
    assert reward_report["penalties"] == {}

    call_text_identity = copy.deepcopy(document)
    call_text_train = next(
        unit for unit in call_text_identity["units"] if unit["qualified_name"] == "train"
    )
    for claim in call_text_train["calls"]:
        claim.pop("name", None)
    call_text_report = score_document(call_text_identity, repo_path=repo)
    assert "missing_calls" not in call_text_report["penalties"]
    assert "invented_calls" not in call_text_report["penalties"]

    evaluation_report = evaluate_mirror(out, repo_path=repo, out_path=tmp_path / "mirror_eval.json")
    assert evaluation_report["passed"]
    gate_by_name = {gate["name"]: gate for gate in evaluation_report["gates"]}
    assert gate_by_name["tree_sitter_parse_available"]["actual"] == 1.0
    assert gate_by_name["parsed_symbol_ir_coverage"]["actual"] == 1.0
    assert gate_by_name["claim_evidence_coverage"]["actual"] == 1.0

    corrupted = copy.deepcopy(document)
    corrupted_train = next(unit for unit in corrupted["units"] if unit["qualified_name"] == "train")
    corrupted_train["calls"] = corrupted_train["calls"][1:]
    corrupted_report = score_document(corrupted, repo_path=repo)
    assert corrupted_report["score"] < reward_report["files"][0]["score"]
    assert corrupted_report["penalties"]["missing_calls"] >= 1
    missing_data_ml_category = copy.deepcopy(document)
    missing_category_train = next(
        unit for unit in missing_data_ml_category["units"] if unit["qualified_name"] == "train"
    )
    del missing_category_train["data_ml_details"]["checkpointing"]
    missing_category_report = score_document(missing_data_ml_category, repo_path=repo)
    assert missing_category_report["penalties"]["schema_errors"] == 1
    assert "data_ml_details missing required categories" in missing_category_report["issues"][0][
        "message"
    ]

    review_manifest = create_review_pack(out, tmp_path / "review_pack", max_questions=4)
    assert review_manifest["mode"] == "review_pack"
    assert review_manifest["counts"]["questions"] > 0
    assert review_manifest["counts"]["visibility_items"] > 0
    questions = _read_jsonl(tmp_path / "review_pack" / "questions.jsonl")
    assert all(question["evidence_spans"] for question in questions)
    assert any(question["kind"] == "data_ml_behavior" for question in questions)
    review_eval = evaluate_review_pack(tmp_path / "review_pack", mirror_path=out)
    assert review_eval["passed"]
    review_gates = {gate["name"]: gate for gate in review_eval["gates"]}
    assert review_gates["evidence_backed_questions"]["actual"] == 1.0

    study_manifest = create_human_usefulness_study(
        tmp_path / "review_pack",
        tmp_path / "review_study",
    )
    assert study_manifest["mode"] == "human_usefulness_study"
    assert study_manifest["counts"]["mirror_tasks"] == study_manifest["counts"]["source_tasks"]
    assert study_manifest["phase6_requirements"]["required_answer_source"] == (
        "real_timed_reviewer_logs"
    )
    assert "whole_repo" in study_manifest["phase6_requirements"]["required_task_sets"]
    assert "diff_mode" in study_manifest["phase6_requirements"]["required_task_sets"]
    assert "mirror_accuracy_not_lower_than_source" in study_manifest["phase6_requirements"][
        "pass_gates"
    ]
    study_readme = (tmp_path / "review_study" / "README.md").read_text(encoding="utf-8")
    assert "Phase 6 contract requirements" in study_readme
    assert "real timed reviewer logs" in study_readme
    assert "uv run semantic-mirror eval human-study" in study_readme
    answers_path = tmp_path / "review_study_answers.jsonl"
    _write_jsonl(answers_path, _completed_study_answers(tmp_path / "review_study"))
    study_eval = evaluate_human_usefulness_study(
        tmp_path / "review_study",
        answers_path,
        min_accuracy=1.0,
    )
    assert study_eval["passed"]
    study_gates = {gate["name"]: gate for gate in study_eval["gates"]}
    assert study_gates["real_timed_reviewer_logs"]["actual"] is True
    assert study_gates["reviewer_identity_present"]["actual"] is True
    assert study_gates["answer_text_present"]["actual"] is True
    assert study_gates["mirror_faster_than_source"]["actual"] > 1.0
    assert study_gates["mirror_accuracy_not_lower_than_source"]["actual"] >= 0.0
    assert study_gates["visibility_items_acknowledged"]["actual"] == 1.0
    assert study_eval["phase6_gate_summary"]["real_timed_reviewer_logs"]
    assert study_eval["phase6_gate_summary"]["reviewer_identity_present"]
    assert study_eval["phase6_gate_summary"]["answer_text_present"]
    assert study_eval["phase6_gate_summary"]["mirror_accuracy_not_lower_than_source"]
    assert study_eval["phase6_gate_summary"]["mirror_median_faster_than_source"]
    assert study_eval["metrics"]["source_mirror_answer_records"] == (
        study_manifest["counts"]["mirror_tasks"] + study_manifest["counts"]["source_tasks"]
    )
    cli_report = tmp_path / "review_study_eval.json"
    cli_eval = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_mirror.cli",
            "eval",
            "human-study",
            str(tmp_path / "review_study"),
            "--answers",
            str(answers_path),
            "--min-accuracy",
            "1.0",
            "--out",
            str(cli_report),
        ],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    cli_summary = json.loads(cli_eval.stdout)
    assert cli_summary["phase6_gate_summary"]["mirror_accuracy_not_lower_than_source"]
    assert json.loads(cli_report.read_text(encoding="utf-8"))["phase6_gate_summary"][
        "mirror_median_faster_than_source"
    ]
    diff_report = copy.deepcopy(study_eval)
    diff_report["study"] = str(tmp_path / "diff_review_study")
    diff_report["answers"] = str(tmp_path / "diff_review_study_answers.jsonl")
    diff_report["metrics"]["answered_task_sets"] = ["diff_mode"]
    diff_report["phase6_gate_summary"][
        "changed_behavior_accuracy_at_or_above_threshold"
    ] = True
    whole_report_path = tmp_path / "phase6_whole_report.json"
    diff_report_path = tmp_path / "phase6_diff_report.json"
    whole_report_path.write_text(
        json.dumps(study_eval, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diff_report_path.write_text(
        json.dumps(diff_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    phase6_summary = summarize_human_usefulness_studies(
        [whole_report_path, diff_report_path],
        out_path=tmp_path / "phase6_summary.json",
    )
    assert phase6_summary["passed"]
    assert phase6_summary["phase6_gate_summary"]["required_task_sets_present"]
    assert phase6_summary["phase6_gate_summary"]["all_reports_passed"]
    assert phase6_summary["metrics"]["answered_task_sets"] == ["diff_mode", "whole_repo"]
    assert phase6_summary["metrics"]["report_count"] == 2
    phase6_cli_report = tmp_path / "phase6_cli_summary.json"
    phase6_cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_mirror.cli",
            "eval",
            "human-study-suite",
            "--report",
            str(whole_report_path),
            "--report",
            str(diff_report_path),
            "--out",
            str(phase6_cli_report),
        ],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    phase6_cli_summary = json.loads(phase6_cli.stdout)
    assert phase6_cli_summary["phase6_gate_summary"]["required_task_sets_present"]
    assert json.loads(phase6_cli_report.read_text(encoding="utf-8"))["passed"]

    untimed_answers = _completed_study_answers(tmp_path / "review_study")
    untimed_answers[0]["reviewer"] = ""
    untimed_answers[0]["started_at"] = ""
    untimed_answers[0]["completed_at"] = ""
    untimed_answers[0]["elapsed_seconds"] = 0.0
    untimed_answers[1]["answer"] = ""
    untimed_path = tmp_path / "review_study_untimed_answers.jsonl"
    _write_jsonl(untimed_path, untimed_answers)
    untimed_eval = evaluate_human_usefulness_study(
        tmp_path / "review_study",
        untimed_path,
        min_accuracy=1.0,
    )
    untimed_gates = {gate["name"]: gate for gate in untimed_eval["gates"]}
    assert not untimed_eval["passed"]
    assert not untimed_gates["real_timed_reviewer_logs"]["passed"]
    assert not untimed_gates["reviewer_identity_present"]["passed"]
    assert not untimed_gates["answer_text_present"]["passed"]


def test_human_study_answer_coverage_reports_pending_and_ready_state(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    mirror = tmp_path / "mirror"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "train.py").write_text(SAMPLE_TRAIN, encoding="utf-8")
    build_repository(repo, mirror, profile="data_ml", zoom="L4")
    create_review_pack(mirror, tmp_path / "review_pack", max_questions=2)
    create_human_usefulness_study(tmp_path / "review_pack", tmp_path / "study")

    missing = summarize_human_study_answer_coverage(tmp_path / "study")
    assert not missing["passed"]
    missing_gates = {gate["name"]: gate for gate in missing["gates"]}
    assert not missing_gates["answers_file_present"]["passed"]
    assert not missing_gates["real_timed_reviewer_logs"]["passed"]
    assert missing_gates["real_timed_reviewer_logs"]["actual"] == 0.0
    assert not missing_gates["reviewer_identity_present"]["passed"]
    assert missing_gates["reviewer_identity_present"]["actual"] == 0.0
    assert not missing_gates["answer_text_present"]["passed"]
    assert missing_gates["answer_text_present"]["actual"] == 0.0
    assert missing["counts"]["pending_task_records"] == missing["counts"]["task_records"]

    completed_answers = _completed_study_answers(tmp_path / "study")
    partial_path = tmp_path / "partial_answers.jsonl"
    partial_answers = completed_answers[:1] + [
        {**completed_answers[1], "study_task_id": "unknown-task"},
        completed_answers[0],
    ]
    _write_jsonl(partial_path, partial_answers)
    partial = summarize_human_study_answer_coverage(tmp_path / "study", partial_path)
    partial_gates = {gate["name"]: gate for gate in partial["gates"]}
    assert not partial["passed"]
    assert not partial_gates["answer_task_id_coverage"]["passed"]
    assert not partial_gates["unknown_answer_ids"]["passed"]
    assert not partial_gates["duplicate_answer_ids"]["passed"]
    assert partial["counts"]["known_answer_records"] == 2
    assert partial["unknown_answer_ids"] == ["unknown-task"]
    assert partial["duplicate_answer_ids"] == [completed_answers[0]["study_task_id"]]

    answers_path = tmp_path / "answers.jsonl"
    _write_jsonl(answers_path, completed_answers)
    report_path = tmp_path / "coverage.json"
    ready = summarize_human_study_answer_coverage(
        tmp_path / "study",
        answers_path,
        out_path=report_path,
    )
    assert ready["passed"]
    assert ready["counts"]["pending_task_records"] == 0
    assert ready["counts"]["source_mirror_answer_records"] == (
        ready["counts"]["mirror_tasks"] + ready["counts"]["source_tasks"]
    )
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"]

    cli_report = tmp_path / "cli_coverage.json"
    cli_status = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_mirror.cli",
            "review",
            "study-status",
            str(tmp_path / "study"),
            "--answers",
            str(answers_path),
            "--out",
            str(cli_report),
        ],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    cli_summary = json.loads(cli_status.stdout)
    assert cli_summary["passed"]
    assert json.loads(cli_report.read_text(encoding="utf-8"))["counts"][
        "pending_task_records"
    ] == 0


def test_human_study_collection_plan_generates_reproducible_commands(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    mirror = tmp_path / "mirror"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "train.py").write_text(SAMPLE_TRAIN, encoding="utf-8")
    build_repository(repo, mirror, profile="data_ml", zoom="L4")
    create_review_pack(mirror, tmp_path / "review_pack", max_questions=2)
    create_human_usefulness_study(tmp_path / "review_pack", tmp_path / "study")

    plan_path = tmp_path / "phase6_plan.json"
    plan = create_human_study_collection_plan(
        {"whole_repo": tmp_path / "study"},
        answers_dir=tmp_path / "phase6",
        reviewer="reviewer-a",
        batch_size=7,
        out_path=plan_path,
    )
    assert plan["mode"] == "phase6_real_human_study_collection_plan"
    assert plan["required_total_answer_records"] == plan["studies"]["whole_repo"][
        "answer_template_records"
    ]
    assert "--max-tasks 7" in plan["studies"]["whole_repo"]["conduct_command"]
    assert "reviewer-a" in plan["studies"]["whole_repo"]["conduct_command"]
    assert "review study-status" in plan["studies"]["whole_repo"]["coverage_command"]
    assert "eval human-study-suite" in plan["suite_command"]
    assert json.loads(plan_path.read_text(encoding="utf-8"))["reviewer"] == "reviewer-a"

    cli_plan = tmp_path / "phase6_cli_plan.json"
    cli_status = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_mirror.cli",
            "review",
            "study-collection-plan",
            "--study",
            f"whole_repo={tmp_path / 'study'}",
            "--answers-dir",
            str(tmp_path / "phase6"),
            "--reviewer",
            "reviewer-a",
            "--batch-size",
            "7",
            "--out",
            str(cli_plan),
        ],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    cli_summary = json.loads(cli_status.stdout)
    assert cli_summary["mode"] == "phase6_real_human_study_collection_plan"
    assert json.loads(cli_plan.read_text(encoding="utf-8"))["batch_size"] == 7


def test_semantic_zoom_levels_control_detail_budget(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "train.py").write_text(SAMPLE_TRAIN, encoding="utf-8")
    outputs: dict[str, Path] = {}
    documents: dict[str, dict[str, object]] = {}

    for zoom in ("L1", "L2", "L4"):
        out = tmp_path / f"mirror-{zoom}"
        outputs[zoom] = out
        build_repository(repo, out, profile="data_ml", zoom=zoom)
        documents[zoom] = json.loads((out / "src" / "train.py.sir.json").read_text(encoding="utf-8"))

    train_l1 = next(unit for unit in documents["L1"]["units"] if unit["qualified_name"] == "train")
    train_l2 = next(unit for unit in documents["L2"]["units"] if unit["qualified_name"] == "train")
    train_l4 = next(unit for unit in documents["L4"]["units"] if unit["qualified_name"] == "train")

    assert train_l1["zoom_policy"]["intent"] == "repo_module_intent_and_major_flows"
    assert train_l1["calls"] == []
    assert train_l1["reads"] == []
    assert train_l1["returns"] == []
    assert train_l1["control_flow"][0]["kind"] == "control_flow_summary"
    assert train_l1["data_ml_details"]["training_loops"]
    assert not train_l1["data_ml_details"]["optimizer_scheduler"]

    assert train_l2["zoom_policy"]["intent"] == "function_class_behavior_and_side_effects"
    assert train_l2["calls"]
    assert train_l2["returns"]
    assert train_l2["reads"] == []
    assert train_l2["writes"] == []

    assert train_l4["zoom_policy"]["intent"] == "implementation_sensitive_details_and_data_ml_mechanics"
    assert train_l4["reads"]
    assert train_l4["writes"]
    assert train_l4["data_ml_details"]["optimizer_scheduler"]
    assert all("order" in claim for claim in train_l4["control_flow"])
    assert all("order_scope" in claim for claim in train_l4["state_mutations"])

    l1_score = score_mirror(outputs["L1"], repo_path=repo)
    l4_score = score_mirror(outputs["L4"], repo_path=repo)
    assert l1_score["penalties"] == {}
    assert l4_score["penalties"] == {}
    l1_manifest = json.loads((outputs["L1"] / "manifest.json").read_text(encoding="utf-8"))
    l4_manifest = json.loads((outputs["L4"] / "manifest.json").read_text(encoding="utf-8"))
    assert l1_manifest["coverage"]["generated_claims"] < l4_manifest["coverage"]["generated_claims"]


def test_conduct_human_usefulness_study_records_timed_answers(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    mirror = tmp_path / "mirror"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "train.py").write_text(SAMPLE_TRAIN, encoding="utf-8")
    build_repository(repo, mirror, profile="data_ml", zoom="L4")
    create_review_pack(mirror, tmp_path / "review_pack", max_questions=3)
    create_human_usefulness_study(tmp_path / "review_pack", tmp_path / "study")

    prompts: list[str] = []
    outputs: list[str] = []
    inputs = iter(
        [
            "",
            "first source answer",
            "y",
            "0.8",
            "first note",
            "",
            "second source answer",
            "n",
            "",
            "",
        ]
    )
    timer_values = iter([1.0, 3.5, 10.0, 13.0])
    answers_path = tmp_path / "answers.jsonl"
    summary = conduct_human_usefulness_study(
        tmp_path / "study",
        answers_path,
        reviewer="reviewer-a",
        task_set="source",
        max_tasks=2,
        input_fn=lambda prompt: prompts.append(prompt) or next(inputs),
        output_fn=outputs.append,
        timer=lambda: next(timer_values),
        now_fn=lambda: "2026-06-08T00:00:00+00:00",
    )

    assert summary["mode"] == "human_usefulness_study_conduct"
    assert summary["completed_records"] == 2
    assert any("Expected answer:" in output for output in outputs)
    records = _read_jsonl(answers_path)
    assert [record["elapsed_seconds"] for record in records] == [2.5, 3.0]
    assert [record["correct"] for record in records] == [True, False]
    assert records[0]["confidence"] == 0.8
    assert all(record["reviewer"] == "reviewer-a" for record in records)

    try:
        conduct_human_usefulness_study(
            tmp_path / "study",
            answers_path,
            reviewer="reviewer-a",
            task_set="source",
            max_tasks=1,
            input_fn=lambda prompt: "",
            output_fn=lambda output: None,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("conducting a study should refuse to overwrite existing answers")

    append_inputs = iter(["", "third source answer", "y", "", ""])
    append_timer_values = iter([20.0, 24.0])
    append_summary = conduct_human_usefulness_study(
        tmp_path / "study",
        answers_path,
        reviewer="reviewer-a",
        task_set="source",
        max_tasks=3,
        append=True,
        input_fn=lambda prompt: next(append_inputs),
        output_fn=lambda output: None,
        timer=lambda: next(append_timer_values),
        now_fn=lambda: "2026-06-08T00:00:00+00:00",
    )
    assert append_summary["skipped_existing"] == 2
    assert append_summary["completed_records"] == 1
    assert len(_read_jsonl(answers_path)) == 3

    visibility_inputs = iter(["", "low confidence marker", "y", "1.0", ""])
    visibility_timer_values = iter([30.0, 31.25])
    visibility_path = tmp_path / "visibility_answers.jsonl"
    conduct_human_usefulness_study(
        tmp_path / "study",
        visibility_path,
        reviewer="reviewer-a",
        task_set="visibility",
        max_tasks=1,
        input_fn=lambda prompt: next(visibility_inputs),
        output_fn=lambda output: None,
        timer=lambda: next(visibility_timer_values),
        now_fn=lambda: "2026-06-08T00:00:00+00:00",
    )
    visibility_record = _read_jsonl(visibility_path)[0]
    assert visibility_record["task_type"] == "visibility_marker"
    assert visibility_record["acknowledged"] is True
    assert visibility_record["correct"] is None
    assert visibility_record["elapsed_seconds"] == 1.25


def test_full_eval_contract_status_reports_missing_target_gates(tmp_path: Path) -> None:
    run = tmp_path / "outputs"
    run.mkdir()
    requested = {"sft": 300, "dpo": 120, "rl": 120}
    (run / "training_eval_summary.json").write_text(
        json.dumps(
            {
                "mode": "training_eval_summary",
                "passed": False,
                "all_final_eval_gates_passed": False,
                "eval_run_config": {"requested_max_steps": requested},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8-sig",
    )
    for stage, steps in {"sft": 300, "dpo": 10}.items():
        stage_dir = run / f"semantic-mirror-{stage}"
        stage_dir.mkdir()
        if stage == "dpo":
            (stage_dir / "checkpoint-10").mkdir()
        (stage_dir / "training_stage_manifest.json").write_text(
            json.dumps({"stage": stage, "max_steps": steps}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for name in ("sft_eval", "sft_vs_baseline", "dpo_eval", "dpo_vs_sft"):
        (run / f"{name}.json").write_text(
            json.dumps({"mode": name, "passed": True}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for stage in ("sft", "dpo"):
        sample_dir = run / "samples" / stage
        sample_dir.mkdir(parents=True)
        for name in (
            "sample_manifest.json",
            "raw_candidates.jsonl",
            "repaired_candidates.jsonl",
            "sample_inspection.md",
        ):
            (sample_dir / name).write_text("{}\n", encoding="utf-8")
    diagnostics = run / "diagnostics"
    diagnostics.mkdir()
    for name in (
        "training_summary.json",
        "training_summary.md",
        "sft_loss.png",
        "dpo_loss.png",
        "dpo_reward_accuracy.png",
        "rl_reward.png",
        "rl_parseability.png",
        "generation_lengths.png",
        "eval_metrics.png",
        "schema_coverage.png",
    ):
        (diagnostics / name).write_bytes(b"{}")
    (diagnostics / "training_summary.json").write_text(
        json.dumps(
            {
                "mode": "training_diagnostics",
                "plots": {
                    "dpo_loss": {
                        "source_files": [
                            str(tmp_path / "old-run" / "semantic-mirror-dpo" / "checkpoint-10" / "trainer_state.json")
                        ]
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "full_training_eval_resume_inspection.json").write_text(
        json.dumps(
            {
                "mode": "full_training_eval_resume_inspection",
                "requested_max_steps": requested,
                "reuse_stage_outputs_enabled": True,
                "action_summary": {
                    "action_counts": {"resume": 1, "reuse": 1, "run": 1},
                    "all_stages_reusable": False,
                    "stage_count": 3,
                    "stages_by_action": {
                        "resume": ["dpo"],
                        "reuse": ["sft"],
                        "run": ["rl"],
                    },
                    "training_required": True,
                    "training_stages": ["dpo", "rl"],
                },
                "decisions": {
                    "sft": {
                        "action": "reuse",
                        "requested_max_steps": 300,
                        "manifest_max_steps": 300,
                        "reason": "manifest max_steps matches requested cap",
                        "resume_from_checkpoint": {"path": None, "exists": None},
                    },
                    "dpo": {
                        "action": "resume",
                        "requested_max_steps": 120,
                        "manifest_max_steps": 10,
                        "reason": "resume checkpoint provided",
                        "resume_from_checkpoint": {
                            "path": "outputs/semantic-mirror-dpo/checkpoint-10",
                            "exists": True,
                        },
                    },
                    "rl": {
                        "action": "run",
                        "requested_max_steps": 120,
                        "manifest_max_steps": None,
                        "reason": "no completed stage manifest",
                        "resume_from_checkpoint": {"path": None, "exists": None},
                    },
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    status = summarize_full_eval_contract_status(
        run,
        sft_steps=300,
        dpo_steps=120,
        rl_steps=120,
        out_path=tmp_path / "contract_status.json",
        markdown_out_path=tmp_path / "contract_status.md",
    )
    assert not status["passed"]
    remaining = {item["gate"] for item in status["remaining_items"]}
    assert "dpo_stage_manifest_matches_requested_steps" in remaining
    assert "rl_stage_manifest_matches_requested_steps" in remaining
    assert "dpo_eval_exists_and_passed" in remaining
    assert "dpo_vs_sft_exists_and_passed" in remaining
    assert "rl_eval_exists_and_passed" in remaining
    assert "dpo_sample_inspection_complete" in remaining
    assert "rl_sample_inspection_complete" in remaining
    assert "diagnostic_plots_exist" in remaining
    assert "training_eval_summary_matches_requested_steps" in remaining
    assert not status["training_eval_summary_status"]["stage_manifest_max_steps_match"]
    assert status["stage_status"]["sft"]["manifest_matches_requested_max_steps"]
    assert not status["stage_status"]["dpo"]["manifest_matches_requested_max_steps"]
    assert status["stage_evidence_summary"]["sft"]["manifest_current"]
    assert status["stage_evidence_summary"]["sft"]["eval_current"]
    assert status["stage_evidence_summary"]["sft"]["sample_current"]
    assert not status["stage_evidence_summary"]["dpo"]["manifest_current"]
    assert not status["stage_evidence_summary"]["dpo"]["eval_current"]
    assert not status["stage_evidence_summary"]["dpo"]["sample_current"]
    assert not status["stage_evidence_summary"]["rl"]["manifest_current"]
    assert not status["stage_evidence_summary"]["rl"]["eval_current"]
    assert not status["stage_evidence_summary"]["rl"]["sample_current"]
    assert status["report_status"]["dpo_eval"]["exists"]
    assert status["report_status"]["dpo_eval"]["passed"]
    assert not status["report_status"]["dpo_eval"]["current_for_requested_stage"]
    assert status["sample_status"]["dpo"]["manifest_exists"]
    assert not status["sample_status"]["dpo"]["complete_for_requested_stage"]
    assert status["stage_recovery_status"]["sft"]["action"] == "reuse"
    assert status["stage_recovery_summary"]["sft"]["next_action_command_name"] is None
    assert not status["stage_recovery_summary"]["sft"]["next_action_launches_training"]
    assert status["stage_recovery_status"]["dpo"]["action"] == "resume"
    assert status["stage_recovery_status"]["dpo"]["latest_checkpoint_relative"] == (
        "semantic-mirror-dpo/checkpoint-10"
    )
    assert status["stage_recovery_summary"]["dpo"]["next_action_command_name"] == (
        "full_training_eval"
    )
    assert status["stage_recovery_summary"]["dpo"]["next_action_command_category"] == (
        "training"
    )
    assert status["stage_recovery_summary"]["dpo"]["next_action_launches_training"]
    assert status["stage_recovery_summary"]["dpo"]["next_action_blocked_by_stages"] == [
        "dpo"
    ]
    assert status["stage_recovery_status"]["rl"]["action"] == "run"
    assert status["stage_recovery_summary"]["rl"]["next_action_command_name"] == (
        "full_training_eval"
    )
    assert status["stage_recovery_summary"]["rl"]["next_action_launches_training"]
    assert status["stage_recovery_summary"]["rl"]["next_action_blocked_by_stages"] == [
        "rl"
    ]
    assert "rl_eval" in status["stage_recovery_status"]["rl"]["missing_current_artifacts"]
    assert status["diagnostics_status"]["all_required_plots_exist"]
    assert not status["diagnostics_status"]["sources_current_for_run"]
    assert not status["diagnostics_status"]["stages_current_for_requested_steps"]
    assert status["diagnostics_status"]["stale_or_missing_stages"] == ["dpo", "rl"]
    assert set(status["remaining_by_area"]) == {
        "diagnostics",
        "dpo",
        "final_summary",
        "rl",
    }
    assert "dpo_stage_manifest_matches_requested_steps" in status["remaining_by_area"]["dpo"]
    assert "rl_eval_exists_and_passed" in status["remaining_by_area"]["rl"]
    assert "diagnostic_plots_exist" in status["remaining_by_area"]["diagnostics"]
    assert status["remaining_area_summary"]["dpo"] == {
        "command_category_counts": {"training": 4},
        "command_counts": {"full_training_eval": 4},
        "gate_count": 4,
        "gates": [
            "dpo_stage_manifest_matches_requested_steps",
            "dpo_eval_exists_and_passed",
            "dpo_vs_sft_exists_and_passed",
            "dpo_sample_inspection_complete",
        ],
        "launches_training_count": 4,
        "non_training_count": 0,
    }
    assert status["remaining_area_summary"]["diagnostics"] == {
        "command_category_counts": {"diagnostics": 1},
        "command_counts": {"report": 1},
        "gate_count": 1,
        "gates": ["diagnostic_plots_exist"],
        "launches_training_count": 0,
        "non_training_count": 1,
    }
    recovery_plan = {
        item["gate"]: item for item in status["remaining_recovery_plan"]
    }
    assert status["recovery_plan_summary"] == {
        "action_category_counts": {
            "diagnostics": 1,
            "evaluation": 6,
            "status": 3,
            "training": 2,
        },
        "blocked_item_count": 10,
        "blocked_stage_counts": {"dpo": 7, "rl": 7},
        "blocked_stage_command_matrix": {
            "dpo": {
                "contract_status": {
                    "launches_training_count": 0,
                    "non_training_count": 3,
                    "total_items": 3,
                },
                "full_training_eval": {
                    "launches_training_count": 3,
                    "non_training_count": 0,
                    "total_items": 3,
                },
                "report": {
                    "launches_training_count": 0,
                    "non_training_count": 1,
                    "total_items": 1,
                },
            },
            "rl": {
                "contract_status": {
                    "launches_training_count": 0,
                    "non_training_count": 3,
                    "total_items": 3,
                },
                "full_training_eval": {
                    "launches_training_count": 3,
                    "non_training_count": 0,
                    "total_items": 3,
                },
                "report": {
                    "launches_training_count": 0,
                    "non_training_count": 1,
                    "total_items": 1,
                },
            },
        },
        "command_link_invalid_count": 0,
        "command_link_unchecked_count": 12,
        "command_link_valid_count": 0,
        "command_launches_training_count": 8,
        "command_non_training_count": 4,
        "missing_command_names": [],
        "optional_input_counts": {},
        "optional_input_gate_count": 0,
        "next_action_command_category_counts": {
            "diagnostics": 1,
            "status": 3,
            "training": 8,
        },
        "next_action_command_counts": {
            "contract_status": 3,
            "full_training_eval": 8,
            "report": 1,
        },
        "non_training_action_counts": {},
        "non_training_count": 0,
        "required_input_counts": {},
        "required_input_gate_count": 0,
        "requires_training_count": 12,
        "total_items": 12,
    }
    assert status["training_dependency_summary"] == {
        "ready_non_training_command_counts": {},
        "ready_non_training_command_inputs": {},
        "ready_non_training_count": 0,
        "ready_non_training_optional_input_counts": {},
        "ready_non_training_required_input_counts": {},
        "requires_training_count": 12,
        "total_items": 12,
        "training_launch_command_counts": {"full_training_eval": 8},
        "training_launch_command_inputs": {
            "full_training_eval": {
                "gate_count": 8,
                "optional_inputs": [],
                "required_inputs": [],
            }
        },
        "training_launch_count": 8,
        "training_launch_optional_input_counts": {},
        "training_launch_required_input_counts": {},
        "waiting_non_training_command_counts": {
            "contract_status": 3,
            "report": 1,
        },
        "waiting_non_training_command_inputs": {
            "contract_status": {
                "gate_count": 3,
                "optional_inputs": [],
                "required_inputs": [],
            },
            "report": {
                "gate_count": 1,
                "optional_inputs": [],
                "required_inputs": [],
            },
        },
        "waiting_non_training_count": 4,
        "waiting_non_training_optional_input_counts": {},
        "waiting_non_training_required_input_counts": {},
    }
    assert recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "required_action"
    ] == "resume"
    assert recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "action_category"
    ] == "training"
    assert recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_title"
    ] == "Resume full eval through DPO and RL"
    assert recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_category"
    ] == "training"
    assert recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_command_name"
    ] == "full_training_eval"
    assert recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_launches_training"
    ]
    assert recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_command_link_valid"
    ] is None
    assert recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_command_link_errors"
    ] == ["command_manifest_not_checked"]
    assert recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_command_required_inputs"
    ] == []
    assert recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "requires_training"
    ]
    assert recovery_plan["rl_stage_manifest_matches_requested_steps"][
        "required_action"
    ] == "run"
    assert recovery_plan["rl_stage_manifest_matches_requested_steps"][
        "requires_training"
    ]
    assert recovery_plan["dpo_eval_exists_and_passed"]["required_action"] == (
        "generate_eval_report_after_stage"
    )
    assert recovery_plan["dpo_eval_exists_and_passed"]["action_category"] == (
        "evaluation"
    )
    assert recovery_plan["dpo_eval_exists_and_passed"]["next_action_title"] == (
        "Resume full eval through DPO and RL"
    )
    assert recovery_plan["dpo_eval_exists_and_passed"]["next_action_command_name"] == (
        "full_training_eval"
    )
    assert recovery_plan["dpo_eval_exists_and_passed"]["blocked_by_stages"] == ["dpo"]
    assert recovery_plan["dpo_sample_inspection_complete"]["required_action"] == (
        "generate_sample_inspection_after_stage"
    )
    assert recovery_plan["dpo_sample_inspection_complete"]["blocked_by_stages"] == ["dpo"]
    assert recovery_plan["diagnostic_plots_exist"]["required_action"] == (
        "regenerate_diagnostics"
    )
    assert recovery_plan["diagnostic_plots_exist"]["action_category"] == "diagnostics"
    assert recovery_plan["diagnostic_plots_exist"]["next_action_title"] == (
        "Regenerate target diagnostics"
    )
    assert recovery_plan["diagnostic_plots_exist"]["next_action_command_name"] == (
        "report"
    )
    assert not recovery_plan["diagnostic_plots_exist"][
        "next_action_launches_training"
    ]
    assert recovery_plan["training_eval_summary_matches_requested_steps"][
        "next_action_title"
    ] == "Regenerate contract status"
    assert recovery_plan["training_eval_summary_matches_requested_steps"][
        "next_action_command_name"
    ] == "contract_status"
    assert recovery_plan["diagnostic_plots_exist"]["blocked_by_stages"] == [
        "dpo",
        "rl",
    ]
    assert recovery_plan["training_eval_summary_matches_requested_steps"][
        "blocked_by_stages"
    ] == ["dpo", "rl"]
    assert not status["repo_hygiene_status"]["checked"]
    assert not status["windows_readiness_status"]["checked"]
    assert status["input_preflight_summary"] == {
        "checked": True,
        "failed_report_count": 0,
        "missing_report_count": 2,
        "missing_reports": ["wsl_smoke_inputs", "full_eval_inputs"],
        "passed": False,
        "passed_report_count": 0,
        "passed_reports": [],
        "report_count": 2,
        "summary": "Training input preflight reports are missing or failing.",
        "failed_reports": [],
    }
    assert status["input_preflight_status"]["reports"]["wsl_smoke_inputs"][
        "command_name"
    ] == "preflight_wsl_smoke_inputs"
    assert status["input_preflight_status"]["reports"]["full_eval_inputs"][
        "required_inputs"
    ] == ["held_out_dataset", "baseline_candidates"]
    preflight_dir = run / "preflight"
    preflight_dir.mkdir()
    (preflight_dir / "wsl_smoke_inputs.json").write_text(
        json.dumps(
            {
                "mode": "wsl_smoke_input_preflight",
                "passed": True,
                "issues": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (preflight_dir / "full_eval_inputs.json").write_text(
        json.dumps(
            {
                "mode": "full_eval_input_preflight",
                "passed": True,
                "issues": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert not status["human_usefulness_status"]["checked"]
    scorecard = {row["area"]: row for row in status["contract_scorecard"]}
    assert scorecard["repo_hygiene"]["passed"] is None
    assert scorecard["repo_hygiene"]["max_reward"] == 25
    assert scorecard["repo_hygiene"]["earned_reward"] == 0
    assert scorecard["windows_unsloth_readiness"]["passed"] is None
    assert not scorecard["sft_dpo_rl_implementation"]["passed"]
    assert scorecard["sft_dpo_rl_implementation"]["max_reward"] == 85
    assert not scorecard["diagnostic_plots"]["passed"]
    assert not scorecard["post_training_samples"]["passed"]
    assert not scorecard["real_training_eval_gates"]["passed"]
    assert scorecard["human_usefulness"]["passed"] is None
    assert status["contract_reward_summary"] == {
        "completion_eligible": False,
        "failed_required_areas": [
            "repo_hygiene",
            "windows_unsloth_readiness",
            "sft_dpo_rl_implementation",
            "diagnostic_plots",
            "post_training_samples",
            "real_training_eval_gates",
        ],
        "minimum_acceptable_required_reward": 320,
        "optional_reward_earned": 0,
        "optional_reward_possible": 70,
        "required_reward_earned": 0,
        "required_reward_possible": 420,
        "required_reward_threshold_met": False,
        "zero_failed_required_areas": False,
    }
    diagnostic_item = next(
        item for item in status["remaining_items"] if item["gate"] == "diagnostic_plots_exist"
    )
    assert diagnostic_item["actual"]["foreign_source_file_count"] == 1
    assert diagnostic_item["actual"]["foreign_source_file_examples"]
    assert diagnostic_item["actual"]["stale_or_missing_stages"] == ["dpo", "rl"]
    assert status["resume_inspection_status"]["exists"]
    assert status["resume_inspection_status"]["mode_valid"]
    assert status["resume_inspection_status"]["current_for_requested_steps"]
    assert status["resume_inspection_status"]["requested_step_matches"] == {
        "dpo": True,
        "rl": True,
        "sft": True,
    }
    assert status["resume_inspection_status"]["decision_requested_step_matches"] == {
        "dpo": True,
        "rl": True,
        "sft": True,
    }
    assert status["resume_inspection_status"]["action_summary"] == {
        "action_counts": {"resume": 1, "reuse": 1, "run": 1},
        "all_stages_reusable": False,
        "stage_count": 3,
        "stages_by_action": {
            "resume": ["dpo"],
            "reuse": ["sft"],
            "run": ["rl"],
        },
        "training_required": True,
        "training_stages": ["dpo", "rl"],
    }
    assert status["resume_inspection_status"]["decisions"]["sft"]["action"] == "reuse"
    assert status["resume_inspection_status"]["decisions"]["dpo"]["action"] == "resume"
    assert status["resume_inspection_status"]["decisions"]["rl"]["action"] == "run"
    assert any(
        "DPO_RESUME_FROM_CHECKPOINT=outputs/semantic-mirror-dpo/checkpoint-10"
        in action["command"]
        for action in status["next_actions"]
    )
    assert any(
        action["title"] == "Resume full eval through DPO and RL"
        and "RL/final eval evidence is incomplete" in action["reason"]
        and action["category"] == "training"
        and action["launches_training"] is True
        for action in status["next_actions"]
    )
    assert status["next_action_summary"] == {
        "blocked_stage_command_matrix": {
            "dpo": {
                "contract_status": {
                    "launches_training_count": 0,
                    "non_training_count": 1,
                    "total_items": 1,
                },
                "full_training_eval": {
                    "launches_training_count": 1,
                    "non_training_count": 0,
                    "total_items": 1,
                },
                "report": {
                    "launches_training_count": 0,
                    "non_training_count": 1,
                    "total_items": 1,
                },
            },
            "rl": {
                "contract_status": {
                    "launches_training_count": 0,
                    "non_training_count": 1,
                    "total_items": 1,
                },
                "full_training_eval": {
                    "launches_training_count": 1,
                    "non_training_count": 0,
                    "total_items": 1,
                },
                "report": {
                    "launches_training_count": 0,
                    "non_training_count": 1,
                    "total_items": 1,
                },
            },
        },
        "command_category_counts": {
            "diagnostics": 1,
            "inspection": 1,
            "status": 1,
            "training": 1,
        },
        "command_counts": {
            "contract_status": 1,
            "full_training_eval": 1,
            "inspect_full_training_eval_resume": 1,
            "report": 1,
        },
        "command_inputs": {
            "contract_status": {
                "gate_count": 1,
                "optional_inputs": [],
                "required_inputs": [],
            },
            "full_training_eval": {
                "gate_count": 1,
                "optional_inputs": [],
                "required_inputs": [],
            },
            "inspect_full_training_eval_resume": {
                "gate_count": 1,
                "optional_inputs": [],
                "required_inputs": [],
            },
            "report": {
                "gate_count": 1,
                "optional_inputs": [],
                "required_inputs": [],
            },
        },
        "launches_training_count": 1,
        "missing_command_metadata_count": 0,
        "non_training_count": 3,
        "ready_action_count": 1,
        "ready_command_counts": {"full_training_eval": 1},
        "ready_non_training_action_count": 0,
        "ready_non_training_command_counts": {},
        "ready_training_action_count": 1,
        "ready_training_command_counts": {"full_training_eval": 1},
        "optional_input_counts": {},
        "optional_input_action_count": 0,
        "required_input_counts": {},
        "required_input_action_count": 0,
        "total_items": 4,
    }
    inspect_action = next(
        action for action in status["next_actions"] if action["title"] == "Inspect resume plan"
    )
    assert inspect_action["category"] == "inspection"
    assert inspect_action["command_name"] == "inspect_full_training_eval_resume"
    assert inspect_action["command_category"] == "inspection"
    assert inspect_action["launches_training"] is False
    assert inspect_action["blocked_by_stages"] == []
    assert inspect_action["stage_actions"] == {
        "dpo": "resume",
        "rl": "run",
        "sft": "reuse",
    }
    resume_action = next(
        action
        for action in status["next_actions"]
        if action["title"] == "Resume full eval through DPO and RL"
    )
    assert resume_action["blocked_by_stages"] == ["dpo", "rl"]
    assert resume_action["command_name"] == "full_training_eval"
    assert resume_action["command_category"] == "training"
    assert "required_inputs" not in resume_action
    assert resume_action["stage_actions"]["dpo"] == "resume"
    assert resume_action["stage_actions"]["rl"] == "run"
    assert any(
        action["title"] == "Regenerate target diagnostics"
        and action["command_name"] == "report"
        and action["command_category"] == "diagnostics"
        and action["category"] == "diagnostics"
        and action["launches_training"] is False
        and action["blocked_by_stages"] == ["dpo", "rl"]
        and action["stage_actions"] == {}
        and "stale stages: `dpo`, `rl`" in action["reason"]
        and "train report outputs --out outputs/diagnostics" in action["command"]
        for action in status["next_actions"]
    )
    windows_commands = [
        action["windows_powershell_command"]
        for action in status["next_actions"]
        if action.get("windows_powershell_command")
    ]
    assert windows_commands
    assert any("wsl.exe -d Ubuntu -- bash -lc" in command for command in windows_commands)
    assert any(
        "DPO_RESUME_FROM_CHECKPOINT=outputs/semantic-mirror-dpo/checkpoint-10"
        in command
        for command in windows_commands
    )
    assert any("--dpo-steps 120" in command for command in windows_commands)
    assert json.loads((tmp_path / "contract_status.json").read_text(encoding="utf-8"))[
        "remaining_items"
    ]
    status_markdown = (tmp_path / "contract_status.md").read_text(encoding="utf-8")
    assert "# Semantic Mirror Full-Eval Contract Status" in status_markdown
    assert "`dpo_stage_manifest_matches_requested_steps`" in status_markdown
    assert "`dpo_eval_exists_and_passed`" in status_markdown
    assert "`dpo_sample_inspection_complete`" in status_markdown
    assert "## Stage Evidence" in status_markdown
    assert "| `sft` | `True` | `True` | `True` | `True` |" in status_markdown
    assert "| `dpo` | `False` | `False` | `False` | `False` |" in status_markdown
    assert "| `rl` | `False` | `False` | `False` | `False` |" in status_markdown
    assert "## Stage Recovery" in status_markdown
    assert (
        "| `dpo` | `resume` | `full_training_eval` | `True` | "
        "`semantic-mirror-dpo/checkpoint-10` |"
    ) in status_markdown
    assert "| `rl` | `run` | `full_training_eval` | `True` | `None` |" in status_markdown
    assert "| `dpo` | 4 | `{\"full_training_eval\": 4}` | `4` |" in status_markdown
    assert "| `diagnostics` | 1 | `{\"report\": 1}` | `0` |" in status_markdown
    assert "## Contract Scorecard" in status_markdown
    assert "## Repo Hygiene" in status_markdown
    assert "Repo hygiene not checked" in status_markdown
    assert "## Windows Readiness" in status_markdown
    assert "Windows readiness not checked" in status_markdown
    assert "## Human Usefulness" in status_markdown
    assert "Human usefulness not checked" in status_markdown
    assert "`real_training_eval_gates`" in status_markdown
    assert "## Reward Summary" in status_markdown
    assert "Required reward: `0/420`" in status_markdown
    assert "Completion eligible: `False`" in status_markdown
    assert "## Remaining Items" in status_markdown
    assert "### By Area" in status_markdown
    assert "| `dpo` |" in status_markdown
    assert "| `diagnostics` |" in status_markdown
    assert "### Recovery Plan" in status_markdown
    assert "- Total items: `12`" in status_markdown
    assert "- Requires training: `12`" in status_markdown
    assert "- Command links unchecked: `12`" in status_markdown
    assert "- Commands launching training: `8`" in status_markdown
    assert "- Commands not launching training: `4`" in status_markdown
    assert "- Training launch gates: `8`" in status_markdown
    assert "- Non-training commands waiting on training: `4`" in status_markdown
    assert "- Ready non-training gates: `0`" in status_markdown
    assert (
        '- Training launch command counts: `{"full_training_eval": 8}`'
        in status_markdown
    )
    assert (
        '- Waiting non-training command counts: `{"contract_status": 3, "report": 1}`'
        in status_markdown
    )
    assert "- Ready non-training command counts: `{}`" in status_markdown
    assert "- Training launch command inputs:" in status_markdown
    assert "- Waiting non-training command inputs:" in status_markdown
    assert "- Ready non-training command inputs:" in status_markdown
    assert "- Missing command names: `[]`" in status_markdown
    assert (
        '- Next action command counts: `{"contract_status": 3, "full_training_eval": 8, "report": 1}`'
        in status_markdown
    )
    assert (
        '- Next action command category counts: `{"diagnostics": 1, "status": 3, "training": 8}`'
        in status_markdown
    )
    assert (
        '- Action category counts: `{"diagnostics": 1, "evaluation": 6, "status": 3, "training": 2}`'
        in status_markdown
    )
    assert '- Blocked stage counts: `{"dpo": 7, "rl": 7}`' in status_markdown
    assert (
        "| Gate | Action | Category | Next Action | Command | Inputs | Evidence | Command Link | Requires Training | Blocked By | Artifacts |"
        in status_markdown
    )
    assert "`unchecked`" in status_markdown
    assert (
        "| `dpo_stage_manifest_matches_requested_steps` | `resume` | `training` | `Resume full eval through DPO and RL` (`training`) | `full_training_eval` (training: `True`) | `unchecked` | `True` |"
        not in status_markdown
    )
    assert (
        "| `dpo_stage_manifest_matches_requested_steps` | `resume` | `training` | `Resume full eval through DPO and RL` (`training`) | `full_training_eval` (training: `True`) | required: `None`<br>optional: `None` | current: `10`<br>expected: `120` | `unchecked` | `True` |"
        in status_markdown
    )
    assert (
        "| `dpo_sample_inspection_complete` | `generate_sample_inspection_after_stage` | `evaluation` | `Resume full eval through DPO and RL` (`training`) | `full_training_eval` (training: `True`) | required: `None`<br>optional: `None` | current: `{\"complete\": true, \"stage_current_for_requested_steps\": false}`<br>expected: `true` | `unchecked` | `True` | `dpo` |"
        in status_markdown
    )
    assert (
        "| `diagnostic_plots_exist` | `regenerate_diagnostics` | `diagnostics` | `Regenerate target diagnostics` (`diagnostics`) | `report` (training: `False`) | required: `None`<br>optional: `None` | current: "
        in status_markdown
    )
    assert "## Resume Inspection" in status_markdown
    assert "- Mode valid: `True`" in status_markdown
    assert "- Current for requested steps: `True`" in status_markdown
    assert (
        '- Requested step matches: `{"dpo": true, "rl": true, "sft": true}`'
        in status_markdown
    )
    assert (
        '- Decision requested step matches: `{"dpo": true, "rl": true, "sft": true}`'
        in status_markdown
    )
    assert (
        '- Action summary: `{"action_counts": {"resume": 1, "reuse": 1, "run": 1}, "all_stages_reusable": false, "stage_count": 3, "stages_by_action": {"resume": ["dpo"], "reuse": ["sft"], "run": ["rl"]}, "training_required": true, "training_stages": ["dpo", "rl"]}`'
        in status_markdown
    )
    assert "| `dpo` | `resume` | 120 | 10 |" in status_markdown
    assert "## Next Actions" in status_markdown
    assert "- Ready actions: `1`" in status_markdown
    assert "- Ready training actions: `1`" in status_markdown
    assert "- Ready non-training actions: `0`" in status_markdown
    assert '- Ready command counts: `{"full_training_eval": 1}`' in status_markdown
    assert "Resume full eval through DPO and RL" in status_markdown
    assert "- Category: `training`" in status_markdown
    assert "- Launches training: `True`" in status_markdown
    assert "- Blocked by stages: `dpo, rl`" in status_markdown
    assert '- Stage actions: `{"dpo": "resume", "rl": "run", "sft": "reuse"}`' in status_markdown
    assert "- Missing answer targets: `None`" in status_markdown
    assert "- Remaining answer records: `None`" in status_markdown
    assert "- Category: `inspection`" in status_markdown
    assert "- Launches training: `False`" in status_markdown
    assert "- Blocked by stages: `None`" in status_markdown
    assert "RL/final eval evidence is incomplete" in status_markdown
    assert "Regenerate target diagnostics" in status_markdown
    assert "bash launch/inspect_full_training_eval_resume.sh" in status_markdown
    assert "```powershell" in status_markdown
    assert "wsl.exe -d Ubuntu -- bash -lc" in status_markdown

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Semantic Mirror Test")
    _git(repo, "config", "user.email", "semantic@example.test")
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("review me\n", encoding="utf-8")
    (repo / ".gitignore").write_text(
        ".env\n.semantic-mirror/\nSEMANTIC_MIRROR_GOAL_CONTRACT.md\nignored.tmp\n",
        encoding="utf-8",
    )
    (repo / ".env").write_text("SECRET=local\n", encoding="utf-8")
    (repo / "SEMANTIC_MIRROR_GOAL_CONTRACT.md").write_text(
        "# local execution contract\n",
        encoding="utf-8",
    )
    (repo / ".semantic-mirror").mkdir()
    (repo / ".semantic-mirror" / "local.json").write_text("{}\n", encoding="utf-8")
    (repo / "ignored.tmp").write_text("unexpected\n", encoding="utf-8")
    dirty_status = summarize_full_eval_contract_status(run, repo_root=repo)
    assert dirty_status["repo_hygiene_status"]["checked"]
    assert not dirty_status["repo_hygiene_status"]["passed"]
    assert len(dirty_status["repo_hygiene_status"]["tracked_changes"]) == 1
    assert "untracked.txt" in dirty_status["repo_hygiene_status"]["untracked"]
    assert ".env" in dirty_status["repo_hygiene_status"]["ignored_allowed"]
    assert (
        "SEMANTIC_MIRROR_GOAL_CONTRACT.md"
        in dirty_status["repo_hygiene_status"]["ignored_allowed"]
    )
    assert ".semantic-mirror/" in dirty_status["repo_hygiene_status"]["ignored_allowed"]
    assert "ignored.tmp" in dirty_status["repo_hygiene_status"]["ignored_unexpected"]
    dirty_scorecard = {row["area"]: row for row in dirty_status["contract_scorecard"]}
    assert dirty_scorecard["repo_hygiene"]["passed"] is False
    windows_audit = tmp_path / "windows_audit.json"
    windows_audit.write_text(
        json.dumps(
            {
                "mode": "training_environment_audit",
                "passed": False,
                "ready_to_launch": False,
                "blocker": {
                    "blocked": True,
                    "evidence": {
                        "audit_command": [
                            "uv",
                            "run",
                            "semantic-mirror",
                            "train",
                            "audit",
                            "training",
                        ],
                        "environment": {
                            "platform": "Windows",
                            "python_executable": "C:/repo/.venv/Scripts/python.exe",
                            "python_version": "3.14.0",
                        },
                        "failed_required_checks": ["torch_cuda_available"],
                        "modules": {
                            "imported_required_versions": {"torch": "test-torch"},
                            "missing_required": [],
                        },
                        "nvidia_smi": {
                            "available": True,
                            "devices": [{"name": "Unit Test GPU"}],
                        },
                        "python": {
                            "actual_version": "3.14.0",
                            "supported": False,
                            "supported_range": ">=3.11,<3.14",
                        },
                        "torch": {
                            "cuda_available": False,
                            "cuda_version": None,
                            "error": "CUDA not available",
                            "importable": True,
                        },
                    },
                    "failed_required_checks": ["torch_cuda_available"],
                    "recommended_fallback": "Use WSL CUDA.",
                    "summary": [
                        "PyTorch CUDA is not available for the audited runtime.",
                    ],
                },
                "environment": {
                    "platform": "Windows",
                    "python_executable": "C:/repo/.venv/Scripts/python.exe",
                    "python_version": "3.14.0",
                },
                "repro": {
                    "audit_command": [
                        "uv",
                        "run",
                        "semantic-mirror",
                        "train",
                        "audit",
                        "training",
                    ],
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    wsl_smoke = tmp_path / "smoke_chain_manifest.json"
    wsl_smoke.write_text(
        json.dumps(
            {
                "mode": "smoke_chain",
                "diagnostics_exists": True,
                "smoke_out": "/home/test/smoke",
                "stages": {
                    stage: {"stage_manifest_exists": True}
                    for stage in ("sft", "dpo", "rl")
                },
                "samples": {
                    stage: {"sample_manifest_exists": True}
                    for stage in ("sft", "dpo", "rl")
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    readiness_status = summarize_full_eval_contract_status(
        run,
        windows_audit_path=windows_audit,
        wsl_smoke_manifest_path=wsl_smoke,
    )
    assert readiness_status["windows_readiness_status"]["checked"]
    assert readiness_status["windows_readiness_status"]["native_blocked"]
    assert readiness_status["windows_readiness_status"]["native_python_executable"] == (
        "C:/repo/.venv/Scripts/python.exe"
    )
    assert readiness_status["windows_readiness_status"]["native_python_version"] == "3.14.0"
    assert readiness_status["windows_readiness_status"]["native_platform"] == "Windows"
    assert readiness_status["windows_readiness_status"]["native_audit_command"] == [
        "uv",
        "run",
        "semantic-mirror",
        "train",
        "audit",
        "training",
    ]
    assert readiness_status["windows_readiness_status"]["native_blocker_summary"] == [
        "PyTorch CUDA is not available for the audited runtime.",
    ]
    assert readiness_status["windows_readiness_status"][
        "native_blocker_evidence_summary"
    ] == {
        "audit_command": [
            "uv",
            "run",
            "semantic-mirror",
            "train",
            "audit",
            "training",
        ],
        "failed_required_checks": ["torch_cuda_available"],
        "missing_required_modules": [],
        "nvidia_smi_available": True,
        "nvidia_smi_devices": ["Unit Test GPU"],
        "python_executable": "C:/repo/.venv/Scripts/python.exe",
        "python_supported": False,
        "python_supported_range": ">=3.11,<3.14",
        "python_version": "3.14.0",
        "torch_cuda_available": False,
        "torch_cuda_version": None,
        "torch_error": "CUDA not available",
        "torch_importable": True,
    }
    assert readiness_status["windows_readiness_status"]["wsl_smoke_complete"]
    assert readiness_status["windows_readiness_status"]["wsl_failed_checks"] == []
    assert readiness_status["windows_readiness_status"]["wsl_smoke_manifest_mode"] == (
        "smoke_chain"
    )
    assert readiness_status["windows_readiness_status"][
        "wsl_blocker_evidence_summary"
    ] == {
        "diagnostics_exists": True,
        "failed_checks": [],
        "missing_sample_manifests": [],
        "missing_stage_manifests": [],
        "smoke_complete": True,
        "smoke_manifest_mode": "smoke_chain",
        "smoke_manifest_path": str(wsl_smoke.resolve()),
        "smoke_out": "/home/test/smoke",
    }
    assert readiness_status["windows_readiness_status"]["next_action_command_name"] is None
    assert (
        readiness_status["windows_readiness_status"]["next_action_launches_training"]
        is False
    )
    assert (
        readiness_status["windows_readiness_status"]["next_action_command_exists"]
        is None
    )
    assert (
        readiness_status["windows_readiness_status"][
            "next_action_command_required_inputs"
        ]
        == []
    )
    assert (
        readiness_status["windows_readiness_status"][
            "next_action_command_optional_inputs"
        ]
        == []
    )
    assert (
        readiness_status["windows_readiness_status"][
            "next_action_command_link_valid"
        ]
        is None
    )
    assert (
        readiness_status["windows_readiness_status"][
            "next_action_command_link_errors"
        ]
        == []
    )
    assert readiness_status["windows_readiness_status"]["next_action_failed_checks"] == []
    readiness_scorecard = {
        row["area"]: row for row in readiness_status["contract_scorecard"]
    }
    assert readiness_scorecard["windows_unsloth_readiness"]["passed"] is True
    assert readiness_scorecard["windows_unsloth_readiness"]["earned_reward"] == 65
    wrong_wsl_smoke = tmp_path / "training_validate_report.json"
    wrong_wsl_smoke.write_text(
        json.dumps(
            {
                "mode": "training_validate",
                "passed": True,
                "issues": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    wrong_wsl_status = summarize_full_eval_contract_status(
        run,
        windows_audit_path=windows_audit,
        wsl_smoke_manifest_path=wrong_wsl_smoke,
    )
    wrong_wsl_readiness = wrong_wsl_status["windows_readiness_status"]
    assert not wrong_wsl_readiness["passed"]
    assert not wrong_wsl_readiness["wsl_smoke_complete"]
    assert wrong_wsl_readiness["wsl_smoke_manifest_mode"] == "training_validate"
    assert wrong_wsl_readiness["wsl_failed_checks"] == [
        "smoke_chain_manifest_mode",
        "stage_manifests",
        "sample_manifests",
        "diagnostics",
    ]
    assert wrong_wsl_readiness["wsl_blocker_summary"] == [
        "Expected WSL smoke manifest mode `smoke_chain`, got `training_validate`.",
        "Missing WSL smoke stage manifests for: sft, dpo, rl.",
        "Missing WSL smoke sample manifests for: sft, dpo, rl.",
        "WSL smoke diagnostics directory was not reported.",
    ]
    assert wrong_wsl_readiness["wsl_blocker_evidence_summary"] == {
        "diagnostics_exists": False,
        "failed_checks": [
            "smoke_chain_manifest_mode",
            "stage_manifests",
            "sample_manifests",
            "diagnostics",
        ],
        "missing_sample_manifests": ["sft", "dpo", "rl"],
        "missing_stage_manifests": ["sft", "dpo", "rl"],
        "smoke_complete": False,
        "smoke_manifest_mode": "training_validate",
        "smoke_manifest_path": str(wrong_wsl_smoke.resolve()),
        "smoke_out": None,
    }
    assert wrong_wsl_readiness["wsl_missing_stage_manifests"] == ["sft", "dpo", "rl"]
    assert wrong_wsl_readiness["wsl_missing_sample_manifests"] == ["sft", "dpo", "rl"]
    assert wrong_wsl_readiness["next_action_command_name"] == "wsl_smoke_chain"
    assert wrong_wsl_readiness["next_action_command_category"] == "training"
    assert wrong_wsl_readiness["next_action_launches_training"] is True
    assert wrong_wsl_readiness["next_action_command_exists"] is None
    assert wrong_wsl_readiness["next_action_command_required_inputs"] == []
    assert wrong_wsl_readiness["next_action_command_optional_inputs"] == []
    assert wrong_wsl_readiness["next_action_command_link_valid"] is None
    assert wrong_wsl_readiness["next_action_command_link_errors"] == [
        "command_manifest_not_checked"
    ]
    assert wrong_wsl_readiness["next_action_blocked_by_stages"] == ["sft", "dpo", "rl"]
    assert wrong_wsl_readiness["next_action_failed_checks"] == [
        "smoke_chain_manifest_mode",
        "stage_manifests",
        "sample_manifests",
        "diagnostics",
    ]
    assert "Windows-native readiness is blocked" in wrong_wsl_readiness[
        "next_action_reason"
    ]
    assert wrong_wsl_status["remaining_by_area"]["windows_unsloth_readiness"] == [
        "windows_unsloth_readiness_passed"
    ]
    wrong_wsl_recovery = {
        item["gate"]: item for item in wrong_wsl_status["remaining_recovery_plan"]
    }
    assert wrong_wsl_recovery["windows_unsloth_readiness_passed"][
        "required_action"
    ] == "run_wsl_smoke_chain"
    assert wrong_wsl_recovery["windows_unsloth_readiness_passed"]["requires_training"]
    wsl_smoke_action = next(
        action
        for action in wrong_wsl_status["next_actions"]
        if action["title"] == "Run Windows-hosted WSL smoke chain"
    )
    assert wsl_smoke_action["category"] == "training"
    assert wsl_smoke_action["command_name"] == "wsl_smoke_chain"
    assert wsl_smoke_action["command_category"] == "training"
    assert wsl_smoke_action["launches_training"] is True
    assert "smoke_chain_manifest_mode" in wsl_smoke_action["reason"]
    assert "diagnostics" in wsl_smoke_action["reason"]
    assert "run_wsl_smoke_chain.ps1" in wsl_smoke_action["command"]
    assert "Set-Location $package" in wsl_smoke_action["windows_powershell_command"]
    assert "-HeldOutDataset <windows_dataset_dir>" in wsl_smoke_action[
        "windows_powershell_command"
    ]
    wrong_ordered_plan = wrong_wsl_status["ordered_execution_plan"]
    assert wrong_ordered_plan["smoke_prerequisite_open"] is True
    assert wrong_ordered_plan["first_ready_command"] == "wsl_smoke_chain"
    assert wrong_ordered_plan["first_ready_training_command"] == "wsl_smoke_chain"
    assert wrong_ordered_plan["first_ready_non_training_command"] is None
    wrong_ordered_by_command = {
        item["command_name"]: item for item in wrong_ordered_plan["items"]
    }
    assert wrong_ordered_by_command["inspect_full_training_eval_resume"][
        "execution_state"
    ] == "completed"
    assert wrong_ordered_by_command["wsl_smoke_chain"]["execution_state"] == "ready"
    assert wrong_ordered_by_command["full_training_eval"]["execution_state"] == (
        "blocked_by_preconditions"
    )
    assert wrong_ordered_by_command["full_training_eval"][
        "blocked_by_preconditions"
    ] == ["wsl_smoke_chain"]
    human_suite = tmp_path / "phase6_summary.json"
    human_suite.write_text(
        json.dumps(
            {
                "mode": "human_usefulness_study_suite_summary",
                "passed": True,
                "phase6_gate_summary": {
                    "required_task_sets_present": True,
                    "all_reports_passed": True,
                    "real_timed_reviewer_logs": True,
                    "reviewer_identity_present": True,
                    "answer_text_present": True,
                    "mirror_accuracy_not_lower_than_source": True,
                    "mirror_median_faster_than_source": True,
                    "changed_behavior_accuracy_at_or_above_threshold": True,
                    "visibility_items_acknowledged": True,
                },
                "metrics": {
                    "answered_task_sets": ["diff_mode", "whole_repo"],
                    "report_count": 2,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    usefulness_status = summarize_full_eval_contract_status(
        run,
        human_study_suite_path=human_suite,
    )
    assert usefulness_status["human_usefulness_status"]["checked"]
    assert usefulness_status["human_usefulness_status"]["passed"]
    usefulness_scorecard = {
        row["area"]: row for row in usefulness_status["contract_scorecard"]
    }
    assert usefulness_scorecard["human_usefulness"]["passed"] is True
    assert usefulness_scorecard["human_usefulness"]["earned_reward"] == 70
    coverage_report = tmp_path / "whole_repo_coverage.json"
    coverage_report.write_text(
        json.dumps(
            {
                "mode": "human_usefulness_study_answer_coverage",
                "passed": False,
                "study": str(tmp_path / "study"),
                "answers": str(tmp_path / "answers.jsonl"),
                "counts": {
                    "task_records": 108,
                    "pending_task_records": 108,
                    "real_timed_answer_records": 0,
                },
                "gates": [
                    {"name": "answer_task_id_coverage", "passed": False},
                    {"name": "real_timed_reviewer_logs", "passed": False},
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    coverage_only_status = summarize_full_eval_contract_status(
        run,
        human_study_coverage_paths=[coverage_report],
    )
    assert coverage_only_status["human_usefulness_status"]["checked"]
    assert coverage_only_status["human_usefulness_status"]["passed"] is False
    assert coverage_only_status["human_usefulness_status"]["summary"] == (
        "Phase 6 human-study suite report is missing or invalid, and supplied "
        "coverage reports are failing. Failed coverage gates: "
        "answer_task_id_coverage, real_timed_reviewer_logs."
    )
    coverage_status = summarize_full_eval_contract_status(
        run,
        human_study_suite_path=human_suite,
        human_study_coverage_paths=[coverage_report],
    )
    coverage_reports = coverage_status["human_usefulness_status"]["coverage_reports"]
    assert coverage_reports[0]["passed"] is False
    assert coverage_reports[0]["pending_task_count"] == 108
    assert coverage_reports[0]["real_timed_answer_records"] == 0
    assert coverage_reports[0]["failed_gates"] == [
        "answer_task_id_coverage",
        "real_timed_reviewer_logs",
    ]
    phase6_action = next(
        action
        for action in coverage_status["next_actions"]
        if action["title"] == "Create real Phase 6 answer collection plan"
    )
    assert phase6_action["category"] == "human_study"
    assert phase6_action["command_name"] == "phase6_collection_plan"
    assert phase6_action["command_category"] == "human_study"
    assert phase6_action["launches_training"] is False
    assert "review study-collection-plan" in phase6_action["command"]
    assert "--reviewer 'REPLACE_WITH_REVIEWER'" in phase6_action["command"]
    assert "--study whole_repo=" in phase6_action["command"]
    missing_suite_status = summarize_full_eval_contract_status(
        run,
        human_study_suite_path=tmp_path / "missing_phase6_summary.json",
        human_study_coverage_paths=[coverage_report],
    )
    assert missing_suite_status["human_usefulness_status"]["summary"] == (
        "Phase 6 human-study suite report is missing or invalid, and supplied "
        "coverage reports are failing. Failed coverage gates: "
        "answer_task_id_coverage, real_timed_reviewer_logs."
    )
    collection_plan = {
        "mode": "phase6_real_human_study_collection_plan",
        "batch_size": 20,
        "studies": {
            "whole_repo": {
                "answer_target": str(tmp_path / "whole_answers.jsonl"),
                "answer_template_records": 108,
                "conduct_command": (
                    "uv run semantic-mirror review conduct-study "
                    f"{tmp_path / 'study'} --out {tmp_path / 'whole_answers.jsonl'} "
                    "--reviewer reviewer-a --task-set all --append --max-tasks 20"
                ),
                "coverage_command": (
                    "uv run semantic-mirror review study-status "
                    f"{tmp_path / 'study'} --answers {tmp_path / 'whole_answers.jsonl'} "
                    f"--out {tmp_path / 'whole_coverage.json'}"
                ),
                "coverage_report": str(tmp_path / "whole_real_coverage.json"),
                "eval_command": (
                    "uv run semantic-mirror eval human-study "
                    f"{tmp_path / 'study'} --answers {tmp_path / 'whole_answers.jsonl'} "
                    f"--out {tmp_path / 'whole_eval.json'}"
                ),
            }
        },
        "suite_command": (
            "uv run semantic-mirror eval human-study-suite "
            f"--report {tmp_path / 'whole_eval.json'} --out {tmp_path / 'phase6_suite.json'}"
        ),
        "required_total_answer_records": 108,
    }
    _write_jsonl(tmp_path / "whole_answers.jsonl", [{"task_id": "one"}])
    (tmp_path / "whole_real_coverage.json").write_text(
        json.dumps({"mode": "human_usefulness_study_answer_coverage"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "phase6_collection_manifest.json").write_text(
        json.dumps(collection_plan, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    planned_status = summarize_full_eval_contract_status(
        run,
        human_study_suite_path=human_suite,
        human_study_coverage_paths=[coverage_report],
    )
    conduct_action = next(
        action
        for action in planned_status["next_actions"]
        if action["title"] == "Run real Phase 6 collection and eval sequence"
    )
    collection_status = planned_status["human_usefulness_status"][
        "collection_plan_status"
    ]
    planned_missing_suite_status = summarize_full_eval_contract_status(
        run,
        human_study_suite_path=tmp_path / "missing_phase6_summary.json",
        human_study_coverage_paths=[coverage_report],
    )
    assert planned_missing_suite_status["human_usefulness_status"]["summary"] == (
        "Phase 6 human-study suite report is missing or invalid, and real "
        "answer coverage is incomplete (1/108 records)."
    )
    assert collection_status["checked"]
    assert not collection_status["passed"]
    assert collection_status["answer_record_count"] == 1
    assert collection_status["required_total_answer_records"] == 108
    assert collection_status["studies"]["whole_repo"]["answer_target_exists"]
    assert conduct_action["category"] == "human_study"
    assert conduct_action["command_name"] == "phase6_collection_sequence"
    assert conduct_action["command_category"] == "human_study"
    assert conduct_action["launches_training"] is False
    assert conduct_action["blocked_by_stages"] == []
    assert conduct_action["stage_actions"] == {}
    assert conduct_action["required_inputs"] == []
    assert conduct_action["optional_inputs"] == []
    assert conduct_action["missing_answer_targets"] == []
    assert conduct_action["remaining_answer_records"] == 107
    assert conduct_action["answer_collection_progress"]["whole_repo"] == {
        "answer_target": str(tmp_path / "whole_answers.jsonl"),
        "answer_records": 1,
        "required_answer_records": 108,
        "remaining_answer_records": 107,
        "batch_size": 20,
        "sessions_remaining_at_batch_size": 6,
    }
    assert "phase6_collection_manifest.json" in conduct_action["command"]
    assert "review conduct-study" in conduct_action["command"]
    assert "review study-status" in conduct_action["command"]
    assert "eval human-study" in conduct_action["command"]
    assert "eval human-study-suite" in conduct_action["command"]
    planned_refresh_action = next(
        action
        for action in planned_status["next_actions"]
        if action["title"] == "Regenerate contract status"
    )
    assert "--human-study-coverage 'whole_real_coverage.json'" in planned_refresh_action[
        "command"
    ]
    assert "--human-study-coverage 'whole_coverage.json'" not in planned_refresh_action[
        "command"
    ]
    repo_commit = _git(repo, "rev-parse", "HEAD").strip()
    source_freshness = tmp_path / "source_freshness.json"
    package_root = tmp_path / "package"
    launch_dir = package_root / "launch"
    launch_dir.mkdir(parents=True)
    commands = {
            "wsl_smoke_chain": "powershell -File launch/run_wsl_smoke_chain.ps1",
            "preflight_wsl_smoke_inputs": (
                "powershell -File launch/preflight_wsl_smoke_inputs.ps1"
            ),
            "sft": "bash launch/run_sft.sh",
        "dpo": "bash launch/run_dpo.sh",
        "rl": "bash launch/run_rl.sh",
            "full_training_eval": "bash launch/run_full_training_eval.sh",
            "preflight_full_eval_inputs": (
                "powershell -File launch/preflight_full_eval_inputs.ps1"
            ),
            "smoke_chain": "bash launch/run_smoke_chain.sh",
        "inspect_full_training_eval_resume": "bash launch/inspect_full_training_eval_resume.sh",
        "inspect_resume": "PYTHONPATH=src python -m semantic_mirror.cli train inspect-resume outputs",
        "contract_status": "PYTHONPATH=src python -m semantic_mirror.cli train contract-status outputs",
        "source_freshness": "PYTHONPATH=src python -m semantic_mirror.cli train source-freshness .",
        "report": "PYTHONPATH=src python -m semantic_mirror.cli train report outputs",
        "validate": "PYTHONPATH=src python -m semantic_mirror.cli train validate training",
        "audit": "PYTHONPATH=src python -m semantic_mirror.cli train audit training",
        "install": "python -m pip install -r requirements-training.txt",
        "bootstrap_linux_cuda": "bash setup/bootstrap_linux_cuda.sh",
        "bootstrap_wsl_ubuntu": "powershell -File setup/bootstrap_wsl_ubuntu.ps1",
        "generate_candidates": "bash launch/generate_candidates.sh",
        "score_candidates": "bash launch/score_candidates.sh",
        "eval_candidates": "PYTHONPATH=src python -m semantic_mirror.cli eval candidates",
        "inspect_samples": "PYTHONPATH=src python -m semantic_mirror.cli train inspect-samples",
        "compare_sft": "PYTHONPATH=src python -m semantic_mirror.cli eval model-compare",
        "compare_sft_raw": "PYTHONPATH=src python -m semantic_mirror.cli eval model-compare",
        "compare_dpo": "PYTHONPATH=src python -m semantic_mirror.cli eval model-compare",
        "compare_dpo_raw": "PYTHONPATH=src python -m semantic_mirror.cli eval model-compare",
        "compare_rl": "PYTHONPATH=src python -m semantic_mirror.cli eval model-compare",
        "compare_rl_raw": "PYTHONPATH=src python -m semantic_mirror.cli eval model-compare",
    }
    (launch_dir / "commands_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "commands": _test_command_manifest(commands),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (package_root / "pyproject.toml").write_text(
        '[project]\nname = "semantic-mirror-runtime"\nrequires-python = ">=3.11,<3.14"\n',
        encoding="utf-8",
    )
    (package_root / "README.md").write_text("package runbook\n", encoding="utf-8")
    source_freshness.write_text(
        json.dumps(
            {
                "mode": "semantic_mirror_package_source_freshness",
                "git_commit": repo_commit,
                "repo_root": str(tmp_path),
                "package_root": str(package_root),
                "compared_scope": "src/semantic_mirror runtime source tree",
                "compared_file_count": 2,
                "all_compared_files_match": True,
                "comparisons": [
                    {"relative_path": "src/semantic_mirror/cli.py", "match": True},
                    {"relative_path": "src/semantic_mirror/training.py", "match": True},
                ],
                "package_specific_docs": [
                    {
                        "relative_path": "README.md",
                        "reason": "Generated training-bundle README.",
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8-sig",
    )
    freshness_status = summarize_full_eval_contract_status(
        run,
        repo_root=repo,
        package_source_freshness_path=source_freshness,
    )
    assert freshness_status["package_source_status"]["checked"]
    assert freshness_status["package_source_status"]["passed"]
    assert freshness_status["package_source_status"]["git_commit_matches_repo"] is True
    assert freshness_status["package_source_status"]["compared_file_count"] == 2
    assert freshness_status["package_source_status"]["repo_root"] == str(tmp_path)
    assert freshness_status["package_source_status"]["all_package_specific_docs_present"]
    assert freshness_status["package_source_status"]["package_specific_docs"][0][
        "package_exists"
    ]
    assert freshness_status["package_command_manifest_status"]["checked"]
    assert freshness_status["package_command_manifest_status"]["passed"]
    assert freshness_status["package_command_manifest_status"]["training_command_count"] == 6
    assert "inspect_resume" in freshness_status["package_command_manifest_status"][
        "non_training_commands"
    ]
    assert "full_training_eval" in freshness_status["package_command_manifest_status"][
        "training_commands"
    ]
    assert freshness_status["package_metadata_status"]["checked"]
    assert freshness_status["package_metadata_status"]["passed"]
    assert freshness_status["package_metadata_status"]["requires_python"] == ">=3.11,<3.14"
    assert freshness_status["package_metadata_status"]["excludes_python_3_14"]
    readiness_with_manifest = summarize_full_eval_contract_status(
        run,
        repo_root=repo,
        windows_audit_path=windows_audit,
        wsl_smoke_manifest_path=wrong_wsl_smoke,
        package_source_freshness_path=source_freshness,
    )
    assert readiness_with_manifest["windows_readiness_summary"][
        "next_action_command_name"
    ] == "wsl_smoke_chain"
    assert readiness_with_manifest["windows_readiness_summary"][
        "next_action_command_exists"
    ]
    assert readiness_with_manifest["windows_readiness_summary"][
        "next_action_command_link_valid"
    ]
    assert readiness_with_manifest["windows_readiness_summary"][
        "next_action_command_required_inputs"
    ] == ["held_out_dataset"]
    assert readiness_with_manifest["windows_readiness_summary"][
        "next_action_command_optional_inputs"
    ] == []
    freshness_run_action = next(
        action
        for action in freshness_status["next_actions"]
        if action["title"] == "Resume full eval through DPO and RL"
    )
    assert "SOURCE_FRESHNESS_REPO_ROOT='repo'" in freshness_run_action["command"]

    cli_status = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_mirror.cli",
            "train",
            "contract-status",
            str(run),
            "--sft-steps",
            "300",
            "--dpo-steps",
            "120",
            "--rl-steps",
            "120",
            "--repo-root",
            str(repo),
            "--windows-audit",
            str(windows_audit),
            "--wsl-smoke-manifest",
            str(wsl_smoke),
            "--package-source-freshness",
            str(source_freshness),
            "--human-study-suite",
            str(human_suite),
            "--human-study-coverage",
            str(coverage_report),
            "--out",
            str(tmp_path / "contract_status_cli.json"),
            "--markdown-out",
            str(tmp_path / "contract_status_cli.md"),
        ],
        check=False,
        capture_output=True,
        encoding="utf-8",
    )
    assert cli_status.returncode == 1
    cli_stdout = json.loads(cli_status.stdout)
    assert cli_stdout["passed"] is False
    assert cli_stdout["contract_reward_summary"]["completion_eligible"] is False
    stdout_scorecard = {
        row["area"]: row for row in cli_stdout["contract_scorecard_summary"]
    }
    assert stdout_scorecard["repo_hygiene"]["required"] is True
    assert stdout_scorecard["repo_hygiene"]["passed"] is False
    assert stdout_scorecard["repo_hygiene"]["earned_reward"] == 0
    assert stdout_scorecard["windows_unsloth_readiness"]["passed"] is True
    assert stdout_scorecard["windows_unsloth_readiness"]["earned_reward"] == 65
    assert stdout_scorecard["real_training_eval_gates"]["passed"] is False
    assert stdout_scorecard["human_usefulness"]["required"] is False
    assert cli_stdout["repo_hygiene_summary"]["passed"] is False
    assert cli_stdout["repo_hygiene_summary"]["tracked_change_count"] == 1
    assert cli_stdout["repo_hygiene_summary"]["untracked_count"] == 2
    assert cli_stdout["repo_hygiene_summary"]["unexpected_ignored_count"] == 1
    assert cli_stdout["windows_readiness_summary"]["passed"] is True
    assert cli_stdout["windows_readiness_summary"]["native_blocked"] is True
    assert cli_stdout["windows_readiness_summary"][
        "native_failed_required_checks"
    ] == ["torch_cuda_available"]
    assert cli_stdout["windows_readiness_summary"][
        "native_recommended_fallback"
    ] == "Use WSL CUDA."
    assert cli_stdout["windows_readiness_summary"]["native_python_executable"] == (
        "C:/repo/.venv/Scripts/python.exe"
    )
    assert cli_stdout["windows_readiness_summary"]["native_python_version"] == "3.14.0"
    assert cli_stdout["windows_readiness_summary"]["native_platform"] == "Windows"
    assert cli_stdout["windows_readiness_summary"]["native_audit_command"] == [
        "uv",
        "run",
        "semantic-mirror",
        "train",
        "audit",
        "training",
    ]
    assert cli_stdout["windows_readiness_summary"]["native_blocker_summary"] == [
        "PyTorch CUDA is not available for the audited runtime.",
    ]
    assert cli_stdout["windows_readiness_summary"][
        "native_blocker_evidence_summary"
    ]["nvidia_smi_devices"] == ["Unit Test GPU"]
    assert cli_stdout["windows_readiness_summary"][
        "native_blocker_evidence_summary"
    ]["torch_error"] == "CUDA not available"
    assert cli_stdout["windows_readiness_summary"]["wsl_smoke_complete"] is True
    assert cli_stdout["windows_readiness_summary"]["wsl_failed_checks"] == []
    assert cli_stdout["windows_readiness_summary"]["wsl_blocker_summary"] == []
    assert cli_stdout["windows_readiness_summary"][
        "wsl_blocker_evidence_summary"
    ]["smoke_complete"] is True
    assert cli_stdout["windows_readiness_summary"][
        "wsl_blocker_evidence_summary"
    ]["smoke_out"] == "/home/test/smoke"
    assert cli_stdout["windows_readiness_summary"]["wsl_smoke_manifest_mode"] == (
        "smoke_chain"
    )
    assert cli_stdout["windows_readiness_summary"]["next_action_command_name"] is None
    assert (
        cli_stdout["windows_readiness_summary"]["next_action_launches_training"]
        is False
    )
    assert cli_stdout["windows_readiness_summary"]["next_action_failed_checks"] == []
    assert cli_stdout["package_source_summary"]["passed"] is True
    assert cli_stdout["package_source_summary"]["git_commit_matches_repo"] is True
    assert cli_stdout["package_source_summary"]["compared_file_count"] == 2
    assert cli_stdout["package_source_summary"]["all_package_specific_docs_present"]
    assert cli_stdout["package_source_summary"]["missing_package_specific_doc_count"] == 0
    assert cli_stdout["package_command_manifest_summary"]["passed"] is True
    assert cli_stdout["package_command_manifest_summary"]["training_command_count"] == 6
    assert cli_stdout["package_command_manifest_summary"][
        "required_input_command_count"
    ] == 25
    assert cli_stdout["package_command_manifest_summary"][
        "optional_input_command_count"
    ] == 5
    assert "dpo" in cli_stdout["package_command_manifest_summary"][
        "commands_with_required_inputs"
    ]
    assert "rl" in cli_stdout["package_command_manifest_summary"][
        "commands_with_required_inputs"
    ]
    assert "contract_status" in cli_stdout["package_command_manifest_summary"][
        "commands_with_optional_inputs"
    ]
    assert cli_stdout["package_command_manifest_summary"]["command_category_counts"] == {
        "diagnostics": 1,
        "evaluation": 8,
        "generation": 1,
        "inspection": 3,
        "setup": 3,
        "status": 2,
        "training": 6,
        "validation": 4,
    }
    assert cli_stdout["package_command_manifest_summary"]["commands_by_category"][
        "status"
    ] == ["contract_status", "source_freshness"]
    assert cli_stdout["package_command_manifest_summary"]["commands_by_category"][
        "training"
    ] == ["dpo", "full_training_eval", "rl", "sft", "smoke_chain", "wsl_smoke_chain"]
    assert cli_stdout["package_metadata_summary"]["passed"] is True
    assert cli_stdout["package_metadata_summary"]["requires_python"] == ">=3.11,<3.14"
    assert cli_stdout["package_metadata_summary"]["excludes_python_3_14"] is True
    assert cli_stdout["human_usefulness_summary"]["passed"] is True
    assert cli_stdout["human_usefulness_summary"]["collection_plan"]["passed"] is False
    assert cli_stdout["human_usefulness_summary"]["collection_plan"][
        "answer_record_count"
    ] == 1
    assert cli_stdout["human_usefulness_summary"]["collection_plan"][
        "required_total_answer_records"
    ] == 108
    assert cli_stdout["human_usefulness_summary"]["collection_plan"][
        "remaining_total_answer_records"
    ] == 107
    assert cli_stdout["human_usefulness_summary"]["collection_plan"][
        "complete"
    ] is False
    collection_studies = cli_stdout["human_usefulness_summary"]["collection_plan"][
        "studies"
    ]
    assert collection_studies["whole_repo"]["answer_records"] == 1
    assert collection_studies["whole_repo"]["required_answer_records"] == 108
    assert collection_studies["whole_repo"]["remaining_answer_records"] == 107
    assert collection_studies["whole_repo"]["complete"] is False
    assert collection_studies["whole_repo"]["answer_target_exists"] is True
    assert collection_studies["whole_repo"]["answer_target"].endswith(
        "whole_answers.jsonl"
    )
    assert collection_studies["whole_repo"]["coverage_report"].endswith(
        "whole_real_coverage.json"
    )
    assert collection_studies["whole_repo"]["eval_report"] is None
    assert cli_stdout["human_usefulness_summary"]["failed_phase6_gates"] == []
    assert cli_stdout["human_usefulness_summary"]["coverage_reports"][0][
        "passed"
    ] is False
    assert cli_stdout["human_usefulness_summary"]["coverage_reports"][0][
        "pending_task_count"
    ] == 108
    assert cli_stdout["human_usefulness_summary"]["coverage_reports"][0][
        "real_timed_answer_records"
    ] == 0
    assert cli_stdout["human_usefulness_summary"]["coverage_reports"][0][
        "failed_gates"
    ] == ["answer_task_id_coverage", "real_timed_reviewer_logs"]
    assert cli_stdout["stage_recovery_summary"]["sft"]["action"] == "reuse"
    assert cli_stdout["stage_recovery_summary"]["sft"]["manifest_max_steps"] == 300
    assert cli_stdout["stage_recovery_summary"]["sft"]["next_action_command_name"] is None
    assert not cli_stdout["stage_recovery_summary"]["sft"][
        "next_action_launches_training"
    ]
    assert cli_stdout["stage_recovery_summary"]["dpo"]["action"] == "resume"
    assert cli_stdout["stage_recovery_summary"]["dpo"][
        "latest_checkpoint_relative"
    ] == "semantic-mirror-dpo/checkpoint-10"
    assert cli_stdout["stage_recovery_summary"]["dpo"][
        "next_action_command_name"
    ] == "full_training_eval"
    assert cli_stdout["stage_recovery_summary"]["dpo"][
        "next_action_command_category"
    ] == "training"
    assert cli_stdout["stage_recovery_summary"]["dpo"][
        "next_action_launches_training"
    ]
    assert cli_stdout["stage_recovery_summary"]["dpo"][
        "next_action_blocked_by_stages"
    ] == ["dpo"]
    assert cli_stdout["stage_recovery_summary"]["rl"]["action"] == "run"
    assert cli_stdout["stage_recovery_summary"]["rl"][
        "next_action_command_name"
    ] == "full_training_eval"
    assert cli_stdout["stage_recovery_summary"]["rl"][
        "next_action_launches_training"
    ]
    assert cli_stdout["stage_recovery_summary"]["rl"][
        "next_action_blocked_by_stages"
    ] == ["rl"]
    assert cli_stdout["stage_recovery_summary"]["rl"][
        "missing_current_artifact_count"
    ] >= 1
    assert cli_stdout["remaining_by_area"]["dpo"] == [
        "dpo_stage_manifest_matches_requested_steps",
        "dpo_eval_exists_and_passed",
        "dpo_vs_sft_exists_and_passed",
        "dpo_sample_inspection_complete",
    ]
    stdout_recovery_plan = {
        item["gate"]: item for item in cli_stdout["remaining_recovery_plan"]
    }
    assert cli_stdout["recovery_plan_summary"] == {
        **status["recovery_plan_summary"],
        "command_link_unchecked_count": 0,
        "command_link_valid_count": 12,
        "optional_input_counts": {
            "human_study_coverage": 11,
            "human_study_suite": 11,
            "package_source_freshness": 11,
            "repo_root": 3,
            "source_freshness_repo_root": 8,
            "windows_audit": 11,
            "wsl_smoke_manifest": 11,
        },
        "optional_input_gate_count": 11,
        "required_input_counts": {
            "baseline_candidates": 8,
            "held_out_dataset": 8,
            "outputs_dir": 4,
        },
        "required_input_gate_count": 12,
    }
    assert cli_stdout["training_dependency_summary"][
        "training_launch_required_input_counts"
    ] == {
        "baseline_candidates": 8,
        "held_out_dataset": 8,
    }
    assert cli_stdout["training_dependency_summary"][
        "training_launch_command_inputs"
    ] == {
        "full_training_eval": {
            "gate_count": 8,
            "optional_inputs": [
                "human_study_coverage",
                "human_study_suite",
                "package_source_freshness",
                "source_freshness_repo_root",
                "windows_audit",
                "wsl_smoke_manifest",
            ],
            "required_inputs": ["baseline_candidates", "held_out_dataset"],
        }
    }
    assert cli_stdout["training_dependency_summary"][
        "training_launch_optional_input_counts"
    ] == {
        "package_source_freshness": 8,
        "human_study_coverage": 8,
        "human_study_suite": 8,
        "source_freshness_repo_root": 8,
        "windows_audit": 8,
        "wsl_smoke_manifest": 8,
    }
    assert cli_stdout["training_dependency_summary"][
        "waiting_non_training_required_input_counts"
    ] == {"outputs_dir": 4}
    assert cli_stdout["training_dependency_summary"][
        "waiting_non_training_command_inputs"
    ] == {
        "contract_status": {
            "gate_count": 3,
            "optional_inputs": [
                "human_study_coverage",
                "human_study_suite",
                "package_source_freshness",
                "repo_root",
                "windows_audit",
                "wsl_smoke_manifest",
            ],
            "required_inputs": ["outputs_dir"],
        },
        "report": {
            "gate_count": 1,
            "optional_inputs": [],
            "required_inputs": ["outputs_dir"],
        },
    }
    assert cli_stdout["training_dependency_summary"][
        "waiting_non_training_optional_input_counts"
    ] == {
        "human_study_coverage": 3,
        "human_study_suite": 3,
        "package_source_freshness": 3,
        "repo_root": 3,
        "windows_audit": 3,
        "wsl_smoke_manifest": 3,
    }
    assert cli_stdout["training_dependency_summary"][
        "ready_non_training_required_input_counts"
    ] == {}
    assert cli_stdout["training_dependency_summary"][
        "ready_non_training_command_inputs"
    ] == {}
    assert cli_stdout["training_dependency_summary"][
        "ready_non_training_optional_input_counts"
    ] == {}
    assert cli_stdout["command_input_reference"]["held_out_dataset"] == {
        "purpose": "Held-out Semantic Mirror dataset directory used for smoke or full eval.",
        "example": "outputs/heldout_eval_dataset",
        "required_by": [
            "eval_candidates",
            "full_training_eval",
            "preflight_full_eval_inputs",
            "preflight_wsl_smoke_inputs",
            "smoke_chain",
            "wsl_smoke_chain",
        ],
        "optional_by": [],
    }
    assert cli_stdout["command_input_reference"]["baseline_candidates"] == {
        "purpose": "JSONL candidates for the baseline model over the held-out units.",
        "example": "outputs/baseline_candidates_eval.jsonl",
        "required_by": [
            "compare_sft",
            "compare_sft_raw",
            "full_training_eval",
            "preflight_full_eval_inputs",
        ],
        "optional_by": [],
    }
    assert cli_stdout["command_input_reference"]["outputs_dir"]["example"] == "outputs"
    assert cli_stdout["command_input_reference"]["windows_audit"][
        "optional_by"
    ] == ["contract_status", "full_training_eval"]
    ordered_plan = cli_stdout["ordered_execution_plan"]
    assert ordered_plan["smoke_prerequisite_open"] is False
    assert ordered_plan["first_ready_command"] == "full_training_eval"
    assert ordered_plan["first_ready_training_command"] == "full_training_eval"
    assert (
        ordered_plan["first_ready_non_training_command"]
        == "phase6_collection_sequence"
    )
    assert ordered_plan["state_counts"] == {
        "blocked_by_preconditions": 2,
        "completed": 1,
        "ready": 2,
    }
    ordered_by_command = {
        item["command_name"]: item for item in ordered_plan["items"]
    }
    assert ordered_by_command["inspect_full_training_eval_resume"][
        "execution_state"
    ] == "completed"
    assert ordered_by_command["full_training_eval"]["execution_state"] == "ready"
    assert ordered_by_command["full_training_eval"][
        "required_inputs_preflight_passed"
    ] is True
    assert ordered_by_command["full_training_eval"]["input_preflight_report"] == (
        "full_eval_inputs"
    )
    assert ordered_by_command["full_training_eval"]["blocked_by_preconditions"] == []
    assert ordered_by_command["full_training_eval"]["required_inputs"] == [
        "held_out_dataset",
        "baseline_candidates",
    ]
    assert ordered_by_command["report"]["execution_state"] == (
        "blocked_by_preconditions"
    )
    assert ordered_by_command["report"]["blocked_by_preconditions"] == [
        "stage_current_evidence"
    ]
    assert ordered_by_command["phase6_collection_sequence"][
        "missing_answer_targets"
    ] == []
    assert ordered_by_command["phase6_collection_sequence"][
        "remaining_answer_records"
    ] == 107
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "required_action"
    ] == "resume"
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "action_category"
    ] == "training"
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_title"
    ] == "Resume full eval through DPO and RL"
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_category"
    ] == "training"
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_command_name"
    ] == "full_training_eval"
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_launches_training"
    ]
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_command_exists"
    ]
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_command_category"
    ] == "training"
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_command_launches_training"
    ]
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_command_link_valid"
    ]
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_command_link_errors"
    ] == []
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_command_required_inputs"
    ] == ["held_out_dataset", "baseline_candidates"]
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "next_action_command_optional_inputs"
    ] == [
        "source_freshness_repo_root",
        "windows_audit",
        "wsl_smoke_manifest",
        "package_source_freshness",
        "human_study_coverage",
        "human_study_suite",
    ]
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "area"
    ] == "dpo"
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "stage"
    ] == "dpo"
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "current_evidence"
    ] == 10
    assert stdout_recovery_plan["dpo_stage_manifest_matches_requested_steps"][
        "expected_evidence"
    ] == 120
    assert stdout_recovery_plan["dpo_eval_exists_and_passed"][
        "required_action"
    ] == "generate_eval_report_after_stage"
    assert stdout_recovery_plan["dpo_eval_exists_and_passed"][
        "action_category"
    ] == "evaluation"
    assert stdout_recovery_plan["dpo_eval_exists_and_passed"]["next_action_title"] == (
        "Resume full eval through DPO and RL"
    )
    assert stdout_recovery_plan["dpo_eval_exists_and_passed"][
        "next_action_command_name"
    ] == "full_training_eval"
    assert stdout_recovery_plan["dpo_eval_exists_and_passed"][
        "blocked_by_stages"
    ] == ["dpo"]
    assert stdout_recovery_plan["dpo_sample_inspection_complete"][
        "required_action"
    ] == "generate_sample_inspection_after_stage"
    assert stdout_recovery_plan["dpo_sample_inspection_complete"][
        "blocked_by_stages"
    ] == ["dpo"]
    assert stdout_recovery_plan["rl_sample_inspection_complete"][
        "blocked_by_stages"
    ] == ["rl"]
    assert stdout_recovery_plan["diagnostic_plots_exist"][
        "blocked_by_stages"
    ] == ["dpo", "rl"]
    assert stdout_recovery_plan["diagnostic_plots_exist"]["action_category"] == (
        "diagnostics"
    )
    assert stdout_recovery_plan["diagnostic_plots_exist"]["next_action_title"] == (
        "Regenerate target diagnostics"
    )
    assert stdout_recovery_plan["diagnostic_plots_exist"][
        "next_action_command_name"
    ] == "report"
    assert not stdout_recovery_plan["diagnostic_plots_exist"][
        "next_action_launches_training"
    ]
    assert stdout_recovery_plan["diagnostic_plots_exist"]["area"] == "diagnostics"
    assert stdout_recovery_plan["diagnostic_plots_exist"]["stage"] is None
    assert stdout_recovery_plan["diagnostic_plots_exist"]["current_evidence"][
        "stale_or_missing_stages"
    ] == ["dpo", "rl"]
    assert any(
        action["title"] == "Inspect resume plan"
        and action["command_name"] == "inspect_full_training_eval_resume"
        and action["command_category"] == "inspection"
        and action["launches_training"] is False
        and action["has_command"] is True
        and action["has_windows_powershell_command"] is True
        and action["blocked_by_stages"] == []
        and action["stage_actions"] == {"dpo": "resume", "rl": "run", "sft": "reuse"}
        and "Preview stage reuse" in action["reason"]
        for action in cli_stdout["next_actions"]
    )
    assert any(
        action["title"] == "Run real Phase 6 collection and eval sequence"
        and action["command_name"] == "phase6_collection_sequence"
        and action["command_category"] == "human_study"
        and action["launches_training"] is False
        and action["has_command"] is True
        and action["blocked_by_stages"] == []
        and action["stage_actions"] == {}
        and action["required_inputs"] == []
        and action["optional_inputs"] == []
        and action["missing_answer_targets"] == []
        and action["remaining_answer_records"] == 107
        and "real timed reviewer logs" in action["reason"]
        for action in cli_stdout["next_actions"]
    )
    assert any(
        action["title"] == "Regenerate target diagnostics"
        and action["command_name"] == "report"
        and action["command_category"] == "diagnostics"
        and action["launches_training"] is False
        and action["blocked_by_stages"] == ["dpo", "rl"]
        and action["stage_actions"] == {}
        and "stale stages: `dpo`, `rl`" in action["reason"]
        for action in cli_stdout["next_actions"]
    )
    assert any(
        action["title"] == "Resume full eval through DPO and RL"
        and action["command_name"] == "full_training_eval"
        and action["command_category"] == "training"
        and action["launches_training"] is True
        and action["has_windows_powershell_command"] is True
        and action["blocked_by_stages"] == ["dpo", "rl"]
        and action["stage_actions"]["dpo"] == "resume"
        and action["stage_actions"]["rl"] == "run"
        and action["required_inputs"] == ["held_out_dataset", "baseline_candidates"]
        for action in cli_stdout["next_actions"]
    )
    assert cli_stdout["next_action_summary"] == {
        "blocked_stage_command_matrix": {
            "dpo": {
                "contract_status": {
                    "launches_training_count": 0,
                    "non_training_count": 1,
                    "total_items": 1,
                },
                "full_training_eval": {
                    "launches_training_count": 1,
                    "non_training_count": 0,
                    "total_items": 1,
                },
                "report": {
                    "launches_training_count": 0,
                    "non_training_count": 1,
                    "total_items": 1,
                },
            },
            "rl": {
                "contract_status": {
                    "launches_training_count": 0,
                    "non_training_count": 1,
                    "total_items": 1,
                },
                "full_training_eval": {
                    "launches_training_count": 1,
                    "non_training_count": 0,
                    "total_items": 1,
                },
                "report": {
                    "launches_training_count": 0,
                    "non_training_count": 1,
                    "total_items": 1,
                },
            },
        },
        "command_category_counts": {
            "diagnostics": 1,
            "human_study": 1,
            "inspection": 1,
            "status": 1,
            "training": 1,
        },
        "command_counts": {
            "contract_status": 1,
            "full_training_eval": 1,
            "inspect_full_training_eval_resume": 1,
            "phase6_collection_sequence": 1,
            "report": 1,
        },
        "command_inputs": {
            "contract_status": {
                "gate_count": 1,
                "optional_inputs": [
                    "human_study_coverage",
                    "human_study_suite",
                    "package_source_freshness",
                    "repo_root",
                    "windows_audit",
                    "wsl_smoke_manifest",
                ],
                "required_inputs": ["outputs_dir"],
            },
            "full_training_eval": {
                "gate_count": 1,
                "optional_inputs": [
                    "human_study_coverage",
                    "human_study_suite",
                    "package_source_freshness",
                    "source_freshness_repo_root",
                    "windows_audit",
                    "wsl_smoke_manifest",
                ],
                "required_inputs": ["baseline_candidates", "held_out_dataset"],
            },
            "inspect_full_training_eval_resume": {
                "gate_count": 1,
                "optional_inputs": [
                    "dpo_resume_from_checkpoint",
                    "sft_resume_from_checkpoint",
                ],
                "required_inputs": ["outputs_dir"],
            },
            "phase6_collection_sequence": {
                "gate_count": 1,
                "optional_inputs": [],
                "required_inputs": [],
            },
            "report": {
                "gate_count": 1,
                "optional_inputs": [],
                "required_inputs": ["outputs_dir"],
            },
        },
        "launches_training_count": 1,
        "missing_command_metadata_count": 0,
        "non_training_count": 4,
        "ready_action_count": 2,
        "ready_command_counts": {
            "full_training_eval": 1,
            "phase6_collection_sequence": 1,
        },
        "ready_non_training_action_count": 1,
        "ready_non_training_command_counts": {"phase6_collection_sequence": 1},
        "ready_training_action_count": 1,
        "ready_training_command_counts": {"full_training_eval": 1},
        "optional_input_counts": {
            "dpo_resume_from_checkpoint": 1,
            "human_study_coverage": 2,
            "human_study_suite": 2,
            "package_source_freshness": 2,
            "repo_root": 1,
            "sft_resume_from_checkpoint": 1,
            "source_freshness_repo_root": 1,
            "windows_audit": 2,
            "wsl_smoke_manifest": 2,
        },
        "optional_input_action_count": 3,
        "required_input_counts": {
            "baseline_candidates": 1,
            "held_out_dataset": 1,
            "outputs_dir": 3,
        },
        "required_input_action_count": 4,
        "total_items": 5,
    }
    cli_status_json = json.loads(
        (tmp_path / "contract_status_cli.json").read_text(encoding="utf-8")
    )
    for summary_key in [
        "contract_scorecard_summary",
        "repo_hygiene_summary",
        "windows_readiness_summary",
        "package_source_summary",
        "package_command_manifest_summary",
        "package_metadata_summary",
        "input_preflight_summary",
        "human_usefulness_summary",
        "next_action_summary",
        "ordered_execution_plan",
        "stage_recovery_summary",
        "remaining_area_summary",
        "training_dependency_summary",
        "command_input_reference",
    ]:
        assert cli_stdout[summary_key] == cli_status_json[summary_key]
    saved_scorecard = {
        row["area"]: row for row in cli_status_json["contract_scorecard_summary"]
    }
    assert saved_scorecard["repo_hygiene"]["passed"] is False
    assert saved_scorecard["windows_unsloth_readiness"]["earned_reward"] == 65
    assert cli_status_json["repo_hygiene_summary"]["tracked_change_count"] == 1
    assert cli_status_json["repo_hygiene_summary"]["untracked_count"] == 2
    assert cli_status_json["windows_readiness_summary"]["passed"] is True
    assert cli_status_json["windows_readiness_summary"]["wsl_smoke_manifest_mode"] == (
        "smoke_chain"
    )
    assert cli_status_json["windows_readiness_summary"]["wsl_blocker_summary"] == []
    assert cli_status_json["windows_readiness_status"]["native_python_version"] == "3.14.0"
    assert cli_status_json["windows_readiness_status"]["native_blocker_summary"] == [
        "PyTorch CUDA is not available for the audited runtime.",
    ]
    assert cli_status_json["package_source_summary"]["passed"] is True
    assert cli_status_json["package_source_summary"]["mismatched_file_count"] == 0
    assert cli_status_json["package_source_summary"][
        "missing_package_specific_doc_count"
    ] == 0
    assert cli_status_json["package_command_manifest_status"]["passed"]
    assert cli_status_json["package_command_manifest_summary"]["training_command_count"] == 6
    assert (
        cli_status_json["package_command_manifest_summary"]["required_input_command_count"]
        == 25
    )
    assert (
        cli_status_json["package_command_manifest_summary"]["optional_input_command_count"]
        == 5
    )
    assert cli_status_json["package_command_manifest_summary"]["command_category_counts"][
        "evaluation"
    ] == 8
    assert cli_status_json["package_command_manifest_summary"]["commands_by_category"][
        "diagnostics"
    ] == ["report"]
    assert cli_status_json["package_command_manifest_status"]["training_commands"] == [
        "dpo",
        "full_training_eval",
        "rl",
        "sft",
        "smoke_chain",
        "wsl_smoke_chain",
    ]
    assert cli_status_json["package_command_manifest_status"]["command_lookup"]["dpo"][
        "required_inputs"
    ] == ["training_dir", "sft_model_or_adapter", "output_dir"]
    assert cli_status_json["package_command_manifest_status"]["command_lookup"]["rl"][
        "required_inputs"
    ] == ["training_dir", "dpo_or_sft_model_or_adapter", "output_dir"]
    assert cli_status_json["package_metadata_status"]["passed"]
    assert cli_status_json["package_metadata_summary"]["requires_python"] == ">=3.11,<3.14"
    assert cli_status_json["package_metadata_status"]["requires_python"] == ">=3.11,<3.14"
    assert cli_status_json["human_usefulness_summary"]["collection_plan"][
        "remaining_total_answer_records"
    ] == 107
    assert cli_status_json["human_usefulness_summary"]["coverage_reports"][0][
        "failed_gates"
    ] == ["answer_task_id_coverage", "real_timed_reviewer_logs"]
    assert cli_status_json["stage_recovery_summary"]["dpo"]["action"] == "resume"
    assert cli_status_json["stage_recovery_summary"]["rl"][
        "missing_current_artifact_count"
    ] >= 1
    assert cli_status_json["human_usefulness_status"]["collection_plan_status"][
        "answer_record_count"
    ] == 1
    refresh_action = next(
        action
        for action in cli_status_json["next_actions"]
        if action["title"] == "Regenerate contract status"
    )
    assert refresh_action["category"] == "status"
    assert refresh_action["command_name"] == "contract_status"
    assert refresh_action["command_category"] == "status"
    assert refresh_action["launches_training"] is False
    assert "--repo-root 'repo'" in refresh_action["command"]
    assert "--windows-audit 'windows_audit.json'" in refresh_action["command"]
    assert "--wsl-smoke-manifest 'smoke_chain_manifest.json'" in refresh_action["command"]
    assert "--package-source-freshness 'source_freshness.json'" in refresh_action["command"]
    assert "--human-study-suite 'phase6_summary.json'" in refresh_action["command"]
    assert "--human-study-coverage 'whole_real_coverage.json'" in refresh_action[
        "command"
    ]
    assert "--human-study-coverage 'whole_repo_coverage.json'" not in refresh_action[
        "command"
    ]
    resume_action = next(
        action
        for action in cli_status_json["next_actions"]
        if action["title"] == "Resume full eval through DPO and RL"
    )
    assert resume_action["category"] == "training"
    assert resume_action["command_name"] == "full_training_eval"
    assert resume_action["command_category"] == "training"
    assert resume_action["launches_training"] is True
    assert "SOURCE_FRESHNESS_REPO_ROOT='repo'" in resume_action["command"]
    cli_phase6_action = next(
        action
        for action in cli_status_json["next_actions"]
        if action["title"] == "Run real Phase 6 collection and eval sequence"
    )
    assert cli_phase6_action["category"] == "human_study"
    assert cli_phase6_action["command_name"] == "phase6_collection_sequence"
    assert cli_phase6_action["command_category"] == "human_study"
    assert cli_phase6_action["launches_training"] is False
    assert cli_phase6_action["required_inputs"] == []
    assert cli_phase6_action["optional_inputs"] == []
    assert "review conduct-study" in cli_phase6_action["command"]
    assert "review study-status" in cli_phase6_action["command"]
    assert "eval human-study-suite" in cli_phase6_action["command"]
    contract_status_md = (tmp_path / "contract_status_cli.md").read_text(encoding="utf-8")
    assert "training_eval_summary_matches_requested_steps" in contract_status_md
    assert "- Ready actions: `2`" in contract_status_md
    assert "- Ready training actions: `1`" in contract_status_md
    assert "- Ready non-training actions: `1`" in contract_status_md
    assert (
        '- Ready non-training command counts: `{"phase6_collection_sequence": 1}`'
        in contract_status_md
    )
    assert "Missing Answer Targets | Remaining Answer Records" in contract_status_md
    assert "Inputs Preflight | Launches Training" in contract_status_md
    assert (
        "| 5 | `Run real Phase 6 collection and eval sequence` | "
        "`phase6_collection_sequence` | `ready` | `None` | `None` | `None` | `False` | "
        "`None` | 107 |"
        in contract_status_md
    )
    assert "- Missing answer targets: `None`" in contract_status_md
    assert "- Remaining answer records: `107`" in contract_status_md
    assert (
        "- Answer collection progress: `whole_repo: 1/108 answered, "
        "107 remaining, 6 sessions at batch 20`"
        in contract_status_md
    )
    assert "## Windows Readiness" in contract_status_md
    assert "Native Python executable: `C:/repo/.venv/Scripts/python.exe`" in contract_status_md
    assert "Native Python version: `3.14.0`" in contract_status_md
    assert "Native platform: `Windows`" in contract_status_md
    assert "Native blocker summary: `PyTorch CUDA is not available" in contract_status_md
    assert "Native blocker evidence:" in contract_status_md
    assert '"nvidia_smi_devices": ["Unit Test GPU"]' in contract_status_md
    assert '"torch_error": "CUDA not available"' in contract_status_md
    assert "WSL smoke manifest mode: `smoke_chain`" in contract_status_md
    assert "WSL failed checks: `None`" in contract_status_md
    assert "WSL blocker summary: `None`" in contract_status_md
    assert "WSL blocker evidence:" in contract_status_md
    assert '"smoke_out": "/home/test/smoke"' in contract_status_md
    assert '"smoke_complete": true' in contract_status_md
    assert "WSL missing stage manifests: `None`" in contract_status_md
    assert "WSL missing sample manifests: `None`" in contract_status_md
    assert "## Package Source Freshness" in contract_status_md
    assert (
        "Package runtime source freshness and package-specific docs are proven"
        in contract_status_md
    )
    assert "## Package Command Manifest" in contract_status_md
    assert "Package command manifest classifies training and non-training commands" in contract_status_md
    assert "Training command count: `6`" in contract_status_md
    assert "Commands with optional inputs:" in contract_status_md
    assert "Command category counts:" in contract_status_md
    assert "`status`: `contract_status`, `source_freshness`" in contract_status_md
    assert "Actions with required inputs: `4`" in contract_status_md
    assert "Actions with optional inputs: `3`" in contract_status_md
    assert "Required input counts:" in contract_status_md
    assert "Optional input counts:" in contract_status_md
    assert "Command inputs:" in contract_status_md
    assert "Required inputs: `held_out_dataset, baseline_candidates`" in contract_status_md
    assert "Optional inputs: `source_freshness_repo_root, windows_audit, wsl_smoke_manifest, package_source_freshness, human_study_coverage, human_study_suite`" in contract_status_md
    assert "Gates with required command inputs: `12`" in contract_status_md
    assert "Gates with optional command inputs: `11`" in contract_status_md
    assert "Required command input counts:" in contract_status_md
    assert "Optional command input counts:" in contract_status_md
    assert "| Gate | Action | Category | Next Action | Command | Inputs | Evidence |" in contract_status_md
    assert "required: `held_out_dataset`, `baseline_candidates`<br>optional: `source_freshness_repo_root`, `windows_audit`, `wsl_smoke_manifest`, `package_source_freshness`" in contract_status_md
    assert "## Package Python Metadata" in contract_status_md
    assert "Requires Python: `>=3.11,<3.14`" in contract_status_md
    assert "Run real Phase 6 collection and eval sequence" in contract_status_md
    assert "### Collection Plan" in contract_status_md
    assert "Answer records: `1/108`" in contract_status_md
    assert "whole_repo_coverage.json" in contract_status_md
    assert "real_timed_reviewer_logs" in contract_status_md

    (run / "training_eval_summary.json").write_text(
        json.dumps(
            {
                "mode": "training_eval_summary",
                "passed": True,
                "all_final_eval_gates_passed": True,
                "eval_run_config": {"requested_max_steps": requested},
                "stage_execution_summary": {
                    stage: {"manifest_max_steps": steps}
                    for stage, steps in requested.items()
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for stage, steps in requested.items():
        stage_dir = run / f"semantic-mirror-{stage}"
        stage_dir.mkdir(exist_ok=True)
        (stage_dir / "training_stage_manifest.json").write_text(
            json.dumps({"stage": stage, "max_steps": steps}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for name in ("rl_eval", "rl_vs_sft"):
        (run / f"{name}.json").write_text(
            json.dumps({"mode": name, "passed": True}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    sample_dir = run / "samples" / "rl"
    sample_dir.mkdir(parents=True)
    for name in (
        "sample_manifest.json",
        "raw_candidates.jsonl",
        "repaired_candidates.jsonl",
        "sample_inspection.md",
    ):
        (sample_dir / name).write_text("{}\n", encoding="utf-8")
    (diagnostics / "training_summary.json").write_text(
        json.dumps(
            {
                "mode": "training_diagnostics",
                "plots": {
                    "dpo_loss": {
                        "source_files": [
                            str(run / "semantic-mirror-dpo" / "checkpoint-120" / "trainer_state.json")
                        ]
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert summarize_full_eval_contract_status(run)["passed"]
    good_source_freshness = json.loads(source_freshness.read_text(encoding="utf-8-sig"))
    stale_source_freshness = {**good_source_freshness, "all_compared_files_match": False}
    source_freshness.write_text(
        json.dumps(stale_source_freshness, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stale_source_status = summarize_full_eval_contract_status(
        run,
        repo_root=repo,
        package_source_freshness_path=source_freshness,
    )
    stale_source_gates = {gate["name"]: gate for gate in stale_source_status["gates"]}
    assert not stale_source_status["passed"]
    assert not stale_source_gates["package_source_freshness_valid_when_checked"]["passed"]
    assert stale_source_status["remaining_by_area"]["package"] == [
        "package_source_freshness_valid_when_checked"
    ]
    stale_source_recovery = {
        item["gate"]: item for item in stale_source_status["remaining_recovery_plan"]
    }
    assert stale_source_recovery["package_source_freshness_valid_when_checked"][
        "required_action"
    ] == "regenerate_package_source_freshness"
    assert not stale_source_recovery["package_source_freshness_valid_when_checked"][
        "requires_training"
    ]
    missing_docs_freshness = {
        **good_source_freshness,
        "package_specific_docs": [
            {
                "relative_path": "missing-runbook.md",
                "reason": "Synthetic missing package doc.",
            }
        ],
    }
    source_freshness.write_text(
        json.dumps(missing_docs_freshness, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    missing_docs_status = summarize_full_eval_contract_status(
        run,
        repo_root=repo,
        package_source_freshness_path=source_freshness,
    )
    assert not missing_docs_status["passed"]
    assert not missing_docs_status["package_source_status"][
        "all_package_specific_docs_present"
    ]
    assert missing_docs_status["package_source_status"]["missing_package_specific_docs"] == [
        "missing-runbook.md"
    ]
    source_freshness.write_text(
        json.dumps(good_source_freshness, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command_manifest_path = launch_dir / "commands_manifest.json"
    good_command_manifest = json.loads(command_manifest_path.read_text(encoding="utf-8"))
    command_manifest_path.write_text(
        json.dumps({**good_command_manifest, "schema_version": 2}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bad_command_status = summarize_full_eval_contract_status(
        run,
        repo_root=repo,
        package_source_freshness_path=source_freshness,
    )
    bad_command_gates = {gate["name"]: gate for gate in bad_command_status["gates"]}
    assert not bad_command_status["passed"]
    assert not bad_command_gates["package_command_manifest_valid_when_checked"]["passed"]
    assert bad_command_status["package_command_manifest_status"]["failed_checks"] == [
        "schema_version"
    ]
    assert bad_command_status["remaining_by_area"]["package"] == [
        "package_command_manifest_valid_when_checked"
    ]
    bad_command_recovery = {
        item["gate"]: item for item in bad_command_status["remaining_recovery_plan"]
    }
    assert bad_command_recovery["package_command_manifest_valid_when_checked"][
        "required_action"
    ] == "regenerate_package_command_manifest"
    assert not bad_command_recovery["package_command_manifest_valid_when_checked"][
        "requires_training"
    ]
    command_manifest_path.write_text(
        json.dumps(good_command_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    missing_input_manifest = copy.deepcopy(good_command_manifest)
    missing_input_manifest["commands"]["dpo"].pop("required_inputs")
    command_manifest_path.write_text(
        json.dumps(missing_input_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bad_input_status = summarize_full_eval_contract_status(
        run,
        repo_root=repo,
        package_source_freshness_path=source_freshness,
    )
    assert not bad_input_status["package_command_manifest_status"]["passed"]
    assert bad_input_status["package_command_manifest_status"]["failed_checks"] == [
        "invalid_required_inputs"
    ]
    assert bad_input_status["package_command_manifest_status"][
        "invalid_required_inputs"
    ] == ["dpo"]
    command_manifest_path.write_text(
        json.dumps(good_command_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (package_root / "pyproject.toml").write_text(
        '[project]\nname = "semantic-mirror-runtime"\nrequires-python = ">=3.11"\n',
        encoding="utf-8",
    )
    bad_metadata_status = summarize_full_eval_contract_status(
        run,
        repo_root=repo,
        package_source_freshness_path=source_freshness,
    )
    bad_metadata_gates = {gate["name"]: gate for gate in bad_metadata_status["gates"]}
    assert not bad_metadata_status["passed"]
    assert not bad_metadata_gates["package_python_metadata_valid_when_checked"]["passed"]
    assert bad_metadata_status["package_metadata_status"]["requires_python"] == ">=3.11"
    assert bad_metadata_status["remaining_by_area"]["package"] == [
        "package_python_metadata_valid_when_checked"
    ]
    bad_metadata_recovery = {
        item["gate"]: item for item in bad_metadata_status["remaining_recovery_plan"]
    }
    assert bad_metadata_recovery["package_python_metadata_valid_when_checked"][
        "required_action"
    ] == "fix_package_python_metadata"
    assert not bad_metadata_recovery["package_python_metadata_valid_when_checked"][
        "requires_training"
    ]
    (package_root / "pyproject.toml").write_text(
        '[project]\nname = "semantic-mirror-runtime"\nrequires-python = ">=3.11,<3.14"\n',
        encoding="utf-8",
    )
    assert summarize_full_eval_contract_status(
        run,
        repo_root=repo,
        package_source_freshness_path=source_freshness,
    )["passed"]


def test_inspect_full_eval_resume_cli_reports_safe_stage_decisions(
    tmp_path: Path,
) -> None:
    run = tmp_path / "outputs"
    sft = run / "semantic-mirror-sft"
    dpo = run / "semantic-mirror-dpo"
    rl = run / "semantic-mirror-rl"
    sft.mkdir(parents=True)
    dpo.mkdir(parents=True)
    rl.mkdir(parents=True)
    (sft / "training_stage_manifest.json").write_text(
        json.dumps({"stage": "sft", "max_steps": 300}) + "\n",
        encoding="utf-8",
    )
    (dpo / "training_stage_manifest.json").write_text(
        json.dumps({"stage": "dpo", "max_steps": 10}) + "\n",
        encoding="utf-8",
    )
    (dpo / "checkpoint-10").mkdir()
    report = inspect_full_training_eval_resume(
        run,
        sft_steps=300,
        dpo_steps=120,
        rl_steps=120,
        reuse_stage_outputs=True,
        dpo_resume_from_checkpoint=dpo / "checkpoint-10",
        out_path=tmp_path / "resume.json",
        markdown_out_path=tmp_path / "resume.md",
    )
    assert report["decisions"]["sft"]["action"] == "reuse"
    assert report["decisions"]["dpo"]["action"] == "resume"
    assert report["decisions"]["dpo"]["resume_from_checkpoint"]["exists"] is True
    assert report["decisions"]["rl"]["action"] == "run"
    assert report["action_summary"] == {
        "action_counts": {"resume": 1, "reuse": 1, "run": 1},
        "all_stages_reusable": False,
        "stage_count": 3,
        "stages_by_action": {
            "resume": ["dpo"],
            "reuse": ["sft"],
            "run": ["rl"],
        },
        "training_required": True,
        "training_stages": ["dpo", "rl"],
    }
    resume_markdown = (tmp_path / "resume.md").read_text(encoding="utf-8")
    assert "# Semantic Mirror Full-Eval Resume Inspection" in resume_markdown
    assert "- Training required: `True`" in resume_markdown
    assert "- Training stages: `dpo, rl`" in resume_markdown
    assert '- Action counts: `{"resume": 1, "reuse": 1, "run": 1}`' in resume_markdown
    assert "| `dpo` | `resume` | 120 | 10 |" in resume_markdown

    cli_json = tmp_path / "resume_cli.json"
    cli_markdown = tmp_path / "resume_cli.md"
    cli_status = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_mirror.cli",
            "train",
            "inspect-resume",
            str(run),
            "--sft-steps",
            "300",
            "--dpo-steps",
            "120",
            "--rl-steps",
            "120",
            "--reuse-stage-outputs",
            "--dpo-resume-from-checkpoint",
            str(dpo / "checkpoint-10"),
            "--out",
            str(cli_json),
            "--markdown-out",
            str(cli_markdown),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert cli_status.returncode == 0, cli_status.stderr
    cli_report = json.loads(cli_json.read_text(encoding="utf-8"))
    assert cli_report["decisions"]["sft"]["action"] == "reuse"
    assert cli_report["decisions"]["dpo"]["action"] == "resume"
    assert cli_report["decisions"]["rl"]["action"] == "run"
    assert cli_report["action_summary"]["training_required"] is True
    assert cli_report["action_summary"]["training_stages"] == ["dpo", "rl"]
    assert "full_training_eval_resume_inspection" in cli_status.stdout
    assert cli_markdown.exists()


def test_diff_marks_changed_units_and_preserves_context(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    out = tmp_path / "mirror-diff"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "train.py").write_text(SAMPLE_TRAIN, encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.name", "Semantic Mirror Test")
    _git(repo, "config", "user.email", "semantic@example.test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    base = _git(repo, "rev-parse", "HEAD").strip()

    changed = SAMPLE_TRAIN.replace(
        "        total_loss += loss.item()\n",
        (
            "        total_loss += loss.item()\n"
            "        Path(\"loss.log\").write_text(str(total_loss), encoding=\"utf-8\")\n"
            "        scheduler_step = getattr(optimizer, 'scheduler_step', None)\n"
        ),
    )
    (repo / "src" / "train.py").write_text(changed, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "modify train loop")
    head = _git(repo, "rev-parse", "HEAD").strip()

    manifest = diff_repository(repo, out, base=base, head=head, profile="data_ml", zoom="L4")

    validate_manifest(manifest)
    assert manifest["mode"] == "diff"
    assert manifest["diff"]["base"] == base
    assert manifest["diff"]["head"] == head
    assert manifest["diff"]["changed_paths"][0]["path"] == "src/train.py"
    assert manifest["diff"]["changed_paths"][0]["changed_line_ranges"]

    document = json.loads((out / "src" / "train.py.sir.json").read_text(encoding="utf-8"))
    train_unit = next(unit for unit in document["units"] if unit["qualified_name"] == "train")
    model_unit = next(unit for unit in document["units"] if unit["qualified_name"] == "TinyModel")
    assert train_unit["change_status"] == "changed"
    assert model_unit["change_status"] == "context"
    changed_side_effects = [
        claim
        for claim in train_unit["side_effects"]
        if claim.get("call") == "Path.write_text"
    ]
    assert changed_side_effects
    assert all(claim["change_status"] == "changed" for claim in changed_side_effects)
    assert any(
        claim.get("call") == "torch.save" and claim["change_status"] == "context"
        for claim in train_unit["side_effects"]
    )
    assert any(
        claim.get("kind") == "dynamic_attribute_lookup" and claim["change_status"] == "changed"
        for claim in train_unit["hazards"]
    )
    assert all(claim["change_status"] == "context" for claim in model_unit["calls"])

    markdown = (out / "src" / "train.py.sir.md").read_text(encoding="utf-8")
    assert "[changed] `Path.write_text`" in markdown
    assert "[context] `torch.save`" in markdown

    evaluation_report = evaluate_mirror(out, repo_path=repo)
    assert evaluation_report["passed"]
    diff_gate = next(gate for gate in evaluation_report["gates"] if gate["name"] == "diff_changed_unit_recall")
    assert diff_gate["actual"] == 1.0

    review_manifest = create_review_pack(out, tmp_path / "diff_review_pack")
    assert review_manifest["counts"]["change_tasks"] >= 1
    change_tasks = _read_jsonl(tmp_path / "diff_review_pack" / "change_tasks.jsonl")
    assert any(task["unit_id"] == train_unit["unit_id"] for task in change_tasks)
    assert all(task["evidence_spans"] for task in change_tasks)
    train_task = next(task for task in change_tasks if task["unit_id"] == train_unit["unit_id"])
    assert "Path.write_text" in json.dumps(train_task)
    review_eval = evaluate_review_pack(tmp_path / "diff_review_pack", mirror_path=out)
    assert review_eval["passed"]
    review_gates = {gate["name"]: gate for gate in review_eval["gates"]}
    assert review_gates["changed_behavior_task_coverage"]["actual"] == 1.0

    study_manifest = create_human_usefulness_study(
        tmp_path / "diff_review_pack",
        tmp_path / "diff_review_study",
    )
    assert study_manifest["counts"]["change_task_groups"] >= 1
    assert "diff_mode" in study_manifest["phase6_requirements"]["required_task_sets"]
    assert "changed_behavior_accuracy_at_or_above_threshold" in study_manifest[
        "phase6_requirements"
    ]["pass_gates"]
    diff_answers_path = tmp_path / "diff_review_study_answers.jsonl"
    _write_jsonl(diff_answers_path, _completed_study_answers(tmp_path / "diff_review_study"))
    diff_study_eval = evaluate_human_usefulness_study(
        tmp_path / "diff_review_study",
        diff_answers_path,
        min_accuracy=1.0,
    )
    assert diff_study_eval["passed"]
    diff_study_gates = {gate["name"]: gate for gate in diff_study_eval["gates"]}
    assert diff_study_gates["real_timed_reviewer_logs"]["actual"] is True
    assert diff_study_gates["changed_behavior_accuracy"]["actual"] == 1.0
    assert diff_study_eval["phase6_gate_summary"]["changed_behavior_accuracy_at_or_above_threshold"]
    assert "diff_mode" in diff_study_eval["metrics"]["answered_task_sets"]


def test_syntax_error_file_gets_explicit_unsupported_reason(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    out = tmp_path / "mirror"
    repo.mkdir()
    (repo / "broken.py").write_text("def bad(:\n    pass\n", encoding="utf-8")

    manifest = build_repository(repo, out, profile="data_ml", zoom="L2")
    document = json.loads((out / "broken.py.sir.json").read_text(encoding="utf-8"))

    validate_manifest(manifest)
    validate_ir_document(document)
    assert document["unsupported_reasons"]
    assert document["units"][0]["symbol_type"] == "unsupported_file"
    assert document["units"][0]["uncertainty"][0]["source_spans"]


def test_golden_fixture_covers_control_effects_exceptions_and_ml_training(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    out = tmp_path / "mirror"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "golden.py").write_text(COMPLEX_GOLDEN, encoding="utf-8")

    manifest = build_repository(repo, out, profile="data_ml", zoom="L4")
    document = json.loads((out / "src" / "golden.py.sir.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    validate_ir_document(document)

    validate_unit = next(
        unit for unit in document["units"] if unit["qualified_name"] == "validate_and_record"
    )
    control_kinds = {claim["kind"] for claim in validate_unit["control_flow"]}
    assert {"if", "try", "with", "for"}.issubset(control_kinds)
    failure_kinds = {claim["kind"] for claim in validate_unit["failure_modes"]}
    assert "raise" in failure_kinds
    side_effect_kinds = {claim["kind"] for claim in validate_unit["side_effects"]}
    assert "context_manager" in side_effect_kinds
    assert "call_side_effect" in side_effect_kinds
    mutation_kinds = {claim["kind"] for claim in validate_unit["state_mutations"]}
    assert "subscript_write" in mutation_kinds
    assert "attribute_write" in mutation_kinds
    calls = {claim["name"] for claim in validate_unit["calls"]}
    assert {"open", "handle.write", "Path.write_text"}.issubset(calls)
    assert len(validate_unit["returns"]) == 2
    assert all("order" in claim for claim in validate_unit["control_flow"])

    train_unit = next(unit for unit in document["units"] if unit["qualified_name"] == "train_epoch")
    training_loop_kinds = {
        claim["kind"] for claim in train_unit["data_ml_details"]["training_loops"]
    }
    assert {"data_iteration_loop", "gradient_reset", "backward_pass", "training_step"}.issubset(
        training_loop_kinds
    )
    optimizer_calls = {claim["call"] for claim in train_unit["data_ml_details"]["optimizer_scheduler"]}
    assert {"optimizer.zero_grad", "optimizer.step", "scheduler.step"}.issubset(optimizer_calls)
    checkpoint_calls = {claim["call"] for claim in train_unit["data_ml_details"]["checkpointing"]}
    assert "torch.save" in checkpoint_calls
    metric_calls = {claim["call"] for claim in train_unit["data_ml_details"]["metrics"]}
    assert "metric" in metric_calls
    tensor_calls = {claim["call"] for claim in train_unit["data_ml_details"]["tensor_shapes"]}
    assert "batch.float" in tensor_calls

    reward_report = score_mirror(out, repo_path=repo)
    assert reward_report["penalties"] == {}
    evaluation_report = evaluate_mirror(out, repo_path=repo)
    assert evaluation_report["passed"]


def test_dataset_sample_outputs_curation_sets_and_rejected_negatives(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    out = tmp_path / "dataset"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "train.py").write_text(SAMPLE_TRAIN, encoding="utf-8")

    manifest = sample_dataset(
        repo,
        out,
        profile="data_ml",
        zoom="L4",
        max_units=5,
        review_budget=3,
        hard_negatives_per_unit=3,
    )

    assert manifest["mode"] == "dataset_sample"
    assert manifest["counts"]["silver_records"] == 5
    assert manifest["counts"]["review_queue_records"] == 3
    assert (out / "gold.jsonl").exists()

    silver = _read_jsonl(out / "silver.jsonl")
    review_queue = _read_jsonl(out / "review_queue.jsonl")
    hard_negatives = _read_jsonl(out / "hard_negative.jsonl")
    assert silver
    assert review_queue
    assert hard_negatives
    assert silver[0]["code_slice"]["text"]
    assert "static_facts" in silver[0]
    assert silver[0]["static_analysis"]["backend"] == "tree_sitter_python"
    assert any(record["priority_reasons"] for record in review_queue)
    assert all(record["auto_reject"] for record in hard_negatives)
    assert any(record["verifier_report"]["penalties"] for record in hard_negatives)

    evaluation_report = evaluate_dataset(out, out_path=tmp_path / "dataset_eval.json")
    assert evaluation_report["passed"]
    gate_by_name = {gate["name"]: gate for gate in evaluation_report["gates"]}
    assert gate_by_name["hard_negative_auto_reject_rate"]["actual"] == 1.0
    assert gate_by_name["review_queue_expected_size"]["actual"] == 3

    training_manifest = prepare_training_data(
        out,
        tmp_path / "training",
        base_model="test/base-7b",
        max_records=4,
    )
    assert training_manifest["mode"] == "training_prepare"
    assert training_manifest["base_model"] == "test/base-7b"
    assert training_manifest["output_counts"]["sft_records"] == 4
    assert training_manifest["output_counts"]["rl_prompts"] == 4
    assert training_manifest["output_counts"]["preference_pairs"] > 0

    training_out = tmp_path / "training"
    assert (training_out / "run_unsloth_sft.py").exists()
    assert (training_out / "run_preference_dpo.py").exists()
    assert (training_out / "run_reward_rl.py").exists()
    assert (training_out / "generate_sir_candidates.py").exists()
    assert (training_out / "score_sir_candidates.py").exists()
    sft = _read_jsonl(training_out / "sft.jsonl")
    repairs = _read_jsonl(training_out / "contrastive_repair.jsonl")
    preferences = _read_jsonl(training_out / "preference_pairs.jsonl")
    rl_prompts = _read_jsonl(training_out / "rl_prompts.jsonl")
    sft_config = json.loads((training_out / "unsloth_sft_config.json").read_text(encoding="utf-8"))
    reward_config = json.loads((training_out / "rl_reward_config.json").read_text(encoding="utf-8"))

    assert sft[0]["task"] == "semantic_ir_generation"
    assert len(sft[0]["messages"]) == 3
    assert "static_facts" in sft[0]["messages"][1]["content"]
    assert "static_analysis" in sft[0]["messages"][1]["content"]
    assert "schema_contract" in sft[0]["messages"][1]["content"]
    assert "FINAL_SIR_JSON_START" in sft[0]["messages"][1]["content"]
    assert "FINAL_SIR_JSON_END" in sft[0]["messages"][1]["content"]
    assert "no Markdown fence" in sft[0]["messages"][1]["content"]
    assert "text_excerpt" in sft[0]["messages"][1]["content"]
    prompt_text = sft[0]["messages"][1]["content"]
    prompt_header, final_json_text = prompt_text.split("FINAL_SIR_JSON_START\n", 1)
    final_json_text = final_json_text.split("\nFINAL_SIR_JSON_END", 1)[0]
    prompt_payload = json.loads(prompt_header)
    final_sir_json = json.loads(final_json_text)
    validate_unit(final_sir_json)
    assert final_sir_json["unit_id"] == silver[0]["unit_id"]
    assert (
        prompt_payload["output_rules"][0]
        == "return the final SIR JSON object between FINAL_SIR_JSON_START and FINAL_SIR_JSON_END"
    )
    assert (
        "do not add any top-level key outside schema_contract.required_top_level_keys"
        in prompt_payload["output_rules"]
    )
    assert (
        "copy identity fields exactly from final SIR JSON; do not shorten unit_id or qualified_name"
        in prompt_payload["output_rules"]
    )
    assert (
        'the answer must start with {"unit_id" and must not include template or marker keys'
        in prompt_payload["output_rules"]
    )
    assert final_sir_json["calls"] == prompt_payload["static_facts"]["calls"]
    assert final_sir_json["writes"] == prompt_payload["static_facts"]["writes"]
    assert (
        final_sir_json["data_ml_details"]
        == prompt_payload["static_facts"]["data_ml_details"]
    )
    assert prompt_payload["schema_contract"]["compact_expected_counts"]["calls"] == len(
        final_sir_json["calls"]
    )
    assert (
        prompt_payload["schema_contract"]["compact_expected_counts"]["data_ml_details"][
            "training_loops"
        ]
        == len(final_sir_json["data_ml_details"]["training_loops"])
    )
    assert "source_spans" in sft[0]["messages"][2]["content"]
    assert sft[0]["messages"][2]["content"].startswith('{"unit_id"')
    compact_target = json.loads(sft[0]["messages"][2]["content"])
    validate_unit(compact_target)
    assert "zoom_policy" not in compact_target
    assert len(compact_target["calls"]) <= 2
    assert len(compact_target["writes"]) <= 2
    assert len(compact_target["data_ml_details"]["training_loops"]) <= 1
    assert repairs[0]["task"] == "semantic_ir_repair"
    assert "corrected_sir_unit" in repairs[0]["messages"][2]["content"]
    assert preferences[0]["chosen"] != preferences[0]["rejected"]
    assert preferences[0]["metadata"]["verifier_report"]["penalties"]
    assert rl_prompts[0]["reward_reference"]["static_facts"]
    assert len(rl_prompts[0]["reward_reference"]["static_facts"]["calls"]) >= len(
        compact_target["calls"]
    )
    assert rl_prompts[0]["reward_reference"]["qualified_name"] == silver[0]["qualified_name"]
    assert rl_prompts[0]["reward_reference"]["compact_target"]["unit_id"] == silver[0]["unit_id"]
    assert rl_prompts[0]["reward_reference"]["compact_expected_counts"]["calls"] == len(
        compact_target["calls"]
    )
    assert sft_config["base_model"] == "test/base-7b"
    assert sft_config["method"] == "QLoRA"
    assert reward_config["objective"] == "faithfulness_first_compactness_second"
    assert training_manifest["training_defaults"]["method"] == sft_config["method"]
    assert training_manifest["training_defaults"]["target_model_size"] == sft_config[
        "model_size_target"
    ]
    assert training_manifest["training_defaults"]["target_model_size"] != "7-14B"
    assert training_manifest["files"]["sft_script"] == "run_unsloth_sft.py"
    assert training_manifest["files"]["preference_script"] == "run_preference_dpo.py"
    assert training_manifest["files"]["rl_script"] == "run_reward_rl.py"
    assert training_manifest["files"]["candidate_generation_script"] == "generate_sir_candidates.py"
    assert training_manifest["files"]["reward_script"] == "score_sir_candidates.py"
    validation_report = validate_training_batch(training_out)
    assert validation_report["passed"]
    assert validation_report["issues"] == []
    validation_out = tmp_path / "validation_report.json"
    validation_cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_mirror.cli",
            "train",
            "validate",
            str(training_out),
            "--out",
            str(validation_out),
        ],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    assert json.loads(validation_cli.stdout)["passed"]
    assert json.loads(validation_out.read_text(encoding="utf-8")) == validation_report
    audit_out = tmp_path / "audit_report.json"
    audit_cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_mirror.cli",
            "train",
            "audit",
            str(training_out),
            "--python-executable",
            "definitely-not-semantic-mirror-python",
            "--out",
            str(audit_out),
        ],
        check=False,
        capture_output=True,
        encoding="utf-8",
    )
    assert audit_cli.returncode == 1
    audit_summary = json.loads(audit_cli.stdout)
    audit_report = json.loads(audit_out.read_text(encoding="utf-8"))
    assert audit_summary["blocker"]["blocked"]
    assert audit_summary["repro"]["audit_command"] == audit_report["repro"]["audit_command"]
    assert "--python-executable" in audit_summary["repro"]["audit_command"]

    promotion_report = promote_gold_records(
        out,
        [review_queue[0]["record_id"]],
        labels=["verified_behavior"],
        reviewer="unit-test",
        notes="promote reviewed example",
    )
    assert promotion_report["passed"]
    assert promotion_report["promoted"] == 1
    gold = _read_jsonl(out / "gold.jsonl")
    assert len(gold) == 1
    assert gold[0]["split"] == "gold"
    assert gold[0]["curation"]["source_silver_record_id"] == review_queue[0]["silver_record_id"]
    assert gold[0]["curation"]["reviewer"] == "unit-test"
    assert "verified_behavior" in gold[0]["labels"]
    assert evaluate_dataset(out)["passed"]

    gold_training_manifest = prepare_training_data(
        out,
        tmp_path / "gold_training",
        include_silver_when_gold_exists=False,
    )
    gold_sft_config = json.loads(
        (tmp_path / "gold_training" / "unsloth_sft_config.json").read_text(encoding="utf-8")
    )
    assert gold_training_manifest["base_model"] == DEFAULT_BASE_MODEL
    assert gold_training_manifest["training_defaults"]["method"] == "bf16 LoRA"
    assert gold_sft_config["load_in_16bit"]
    assert gold_training_manifest["input_counts"]["gold"] == 1
    assert gold_training_manifest["output_counts"]["sft_records"] == 1
    gold_sft = _read_jsonl(tmp_path / "gold_training" / "sft.jsonl")
    assert gold_sft[0]["metadata"]["unit_id"] == gold[0]["unit_id"]
    assert validate_training_batch(tmp_path / "gold_training")["passed"]

    def all_modules_available(module: str) -> bool:
        return module in REQUIRED_TRAINING_MODULES

    def cuda_torch_probe() -> dict[str, object]:
        return {
            "importable": True,
            "version": "test-torch",
            "cuda_version": "test-cuda",
            "cuda_available": True,
            "device_count": 1,
            "devices": [{"index": 0, "name": "test-gpu", "total_memory_gb": 24}],
        }

    audit_report = audit_training_environment(
        training_out,
        module_probe=all_modules_available,
        torch_probe=cuda_torch_probe,
        env_values={"HF_TOKEN": "present"},
        platform_name="Linux",
        python_version="3.12.0",
    )
    assert audit_report["passed"]
    assert audit_report["ready_to_launch"]
    assert not [check for check in audit_report["checks"] if check["required"] and not check["passed"]]
    assert audit_report["repro"]["required_python_range"] == ">=3.11,<3.14"
    assert "unsloth" in audit_report["repro"]["required_modules"]
    assert "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" in (
        audit_report["repro"]["training_requirements"]
    )

    dry_run = launch_training_job(
        training_out,
        stage="sft",
        output_dir=tmp_path / "sft-output",
        dry_run=True,
        audit_report=audit_report,
        python_executable="python-test",
        max_steps=5,
        resume_from_checkpoint="sft-checkpoint-path",
        seed=101,
    )
    assert dry_run["passed"]
    assert dry_run["would_launch"]
    assert not dry_run["launched"]
    assert dry_run["command"][0] == "python-test"
    assert "run_unsloth_sft.py" in dry_run["command"][1]
    assert "--max-steps" in dry_run["command"]
    assert "5" in dry_run["command"]
    assert "--resume-from-checkpoint" in dry_run["command"]
    assert "sft-checkpoint-path" in dry_run["command"]
    assert "--seed" in dry_run["command"]
    assert "101" in dry_run["command"]

    dpo_dry_run = launch_training_job(
        training_out,
        stage="dpo",
        output_dir=tmp_path / "dpo-output",
        dry_run=True,
        audit_report=audit_report,
        python_executable="python-test",
        model_name_or_path="sft-adapter-path",
        beta=0.2,
        max_steps=7,
        resume_from_checkpoint="dpo-checkpoint-path",
        seed=102,
    )
    assert dpo_dry_run["passed"]
    assert dpo_dry_run["would_launch"]
    assert not dpo_dry_run["launched"]
    assert "run_preference_dpo.py" in dpo_dry_run["command"][1]
    assert "--model-name-or-path" in dpo_dry_run["command"]
    assert "sft-adapter-path" in dpo_dry_run["command"]
    assert "--beta" in dpo_dry_run["command"]
    assert "0.2" in dpo_dry_run["command"]
    assert "--max-steps" in dpo_dry_run["command"]
    assert "7" in dpo_dry_run["command"]
    assert "--resume-from-checkpoint" in dpo_dry_run["command"]
    assert "dpo-checkpoint-path" in dpo_dry_run["command"]
    assert "--seed" in dpo_dry_run["command"]
    assert "102" in dpo_dry_run["command"]
    dpo_missing_model = launch_training_job(
        training_out,
        stage="dpo",
        output_dir=tmp_path / "dpo-missing-model-output",
        dry_run=True,
        audit_report=audit_report,
    )
    assert not dpo_missing_model["passed"]
    assert not dpo_missing_model["would_launch"]
    assert dpo_missing_model["reason"] == "command_unavailable"
    assert dpo_missing_model["command"] is None
    assert "--model-name-or-path" in dpo_missing_model["command_error"]

    rl_dry_run = launch_training_job(
        training_out,
        stage="rl",
        output_dir=tmp_path / "rl-output",
        dry_run=True,
        audit_report=audit_report,
        python_executable="python-test",
        model_name_or_path="dpo-adapter-path",
        max_steps=3,
        kl_coef=0.02,
        schema_prefix_mode="identity",
        seed=103,
    )
    assert rl_dry_run["passed"]
    assert rl_dry_run["would_launch"]
    assert not rl_dry_run["launched"]
    assert "run_reward_rl.py" in rl_dry_run["command"][1]
    assert "--model-name-or-path" in rl_dry_run["command"]
    assert "dpo-adapter-path" in rl_dry_run["command"]
    assert "--max-steps" in rl_dry_run["command"]
    assert "3" in rl_dry_run["command"]
    assert "--kl-coef" in rl_dry_run["command"]
    assert "--schema-prefix-mode" in rl_dry_run["command"]
    assert "identity" in rl_dry_run["command"]
    assert "--seed" in rl_dry_run["command"]
    assert "103" in rl_dry_run["command"]
    rl_missing_model = launch_training_job(
        training_out,
        stage="rl",
        output_dir=tmp_path / "rl-missing-model-output",
        dry_run=True,
        audit_report=audit_report,
    )
    assert not rl_missing_model["passed"]
    assert not rl_missing_model["would_launch"]
    assert rl_missing_model["reason"] == "command_unavailable"
    assert rl_missing_model["command"] is None
    assert "--model-name-or-path" in rl_missing_model["command_error"]

    package_manifest = package_training_bundle(
        training_out,
        tmp_path / "training_bundle",
        module_probe=all_modules_available,
        torch_probe=cuda_torch_probe,
        env_values={"HF_TOKEN": "secret-hf-token"},
        platform_name="Linux",
        python_version="3.12.0",
    )
    bundle_out = tmp_path / "training_bundle"
    assert package_manifest["mode"] == "training_package"
    assert package_manifest["passed"]
    assert package_manifest["current_runtime_ready"]
    assert package_manifest["output_counts"]["sft_records"] == 4
    assert package_manifest["files"]["smoke_chain_launcher"] == "launch/run_smoke_chain.sh"
    assert package_manifest["files"]["source_freshness"] == "source_freshness.json"
    assert package_manifest["files"]["source_freshness_markdown"] == "source_freshness.md"
    assert package_manifest["source_freshness"]["all_compared_files_match"]
    assert package_manifest["source_freshness"]["compared_file_count"] > 0
    assert validate_training_batch(bundle_out / "training")["passed"]
    assert (bundle_out / "training" / "run_unsloth_sft.py").exists()
    packaged_freshness = json.loads(
        (bundle_out / "source_freshness.json").read_text(encoding="utf-8")
    )
    assert packaged_freshness["all_compared_files_match"]
    assert packaged_freshness["mismatched_files"] == []
    assert packaged_freshness["all_package_specific_docs_present"]
    assert packaged_freshness["missing_package_specific_docs"] == []
    assert {
        doc["relative_path"]: doc["package_exists"]
        for doc in packaged_freshness["package_specific_docs"]
    } == {
        "README.md": True,
        "ENVIRONMENT.md": True,
        ".env.training.example": True,
    }
    assert all(
        isinstance(doc["package_sha256"], str)
        for doc in packaged_freshness["package_specific_docs"]
    )
    assert "Source Freshness Evidence" in (bundle_out / "source_freshness.md").read_text(
        encoding="utf-8"
    )
    freshness = generate_training_package_source_freshness(
        bundle_out,
        repo_root=Path.cwd(),
        out_path=bundle_out / "source_freshness.json",
        markdown_out_path=bundle_out / "source_freshness.md",
    )
    assert freshness["mode"] == "semantic_mirror_package_source_freshness"
    assert freshness["all_compared_files_match"]
    assert freshness["mismatched_files"] == []
    assert freshness["all_package_specific_docs_present"]
    assert freshness["missing_package_specific_docs"] == []
    assert freshness["compared_file_count"] > 0
    assert (bundle_out / "source_freshness.json").exists()
    assert "Source Freshness Evidence" in (bundle_out / "source_freshness.md").read_text(
        encoding="utf-8"
    )
    freshness_cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_mirror.cli",
            "train",
            "source-freshness",
            str(bundle_out),
            "--repo-root",
            str(Path.cwd()),
            "--out",
            str(bundle_out / "source_freshness_cli.json"),
            "--markdown-out",
            str(bundle_out / "source_freshness_cli.md"),
        ],
        check=False,
        capture_output=True,
        encoding="utf-8",
    )
    assert freshness_cli.returncode == 0
    freshness_cli_summary = json.loads(freshness_cli.stdout)
    assert freshness_cli_summary["all_compared_files_match"] is True
    assert freshness_cli_summary["mismatched_files"] == []
    assert (bundle_out / "source_freshness_cli.json").exists()
    sft_script = (bundle_out / "training" / "run_unsloth_sft.py").read_text(encoding="utf-8")
    assert "SFTConfig" in sft_script
    assert "processing_class=tokenizer" in sft_script
    assert "tokenizer=tokenizer" not in sft_script
    assert "_messages_to_text(row[\"messages\"], tokenizer)" in sft_script
    assert "tokenizer.apply_chat_template" in sft_script
    assert 'parser.add_argument("--max-steps", type=int)' in sft_script
    assert 'parser.add_argument("--resume-from-checkpoint")' in sft_script
    assert 'parser.add_argument("--seed", type=int, default=42)' in sft_script
    assert 'parser.add_argument("--save-steps", type=int, default=10)' in sft_script
    assert 'parser.add_argument("--save-total-limit", type=int, default=3)' in sft_script
    assert "max_steps=args.max_steps or -1" in sft_script
    assert 'save_strategy="steps"' in sft_script
    assert "save_steps=args.save_steps" in sft_script
    assert "save_total_limit=args.save_total_limit" in sft_script
    assert "trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)" in sft_script
    assert '"stage": "sft"' in sft_script
    assert '"save_steps": args.save_steps' in sft_script
    assert '"save_total_limit": args.save_total_limit' in sft_script
    assert "training_stage_manifest.json" in sft_script
    assert 'tokenizer.truncation_side = "left"' in sft_script
    assert 'os.environ.setdefault("UNSLOTH_ENABLE_CCE", "0")' in sft_script
    assert 'os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")' in sft_script
    assert 'os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")' in sft_script
    dpo_script = (bundle_out / "training" / "run_preference_dpo.py").read_text(encoding="utf-8")
    assert "TRANSFORMERS_CACHE" in dpo_script
    assert "DPOTrainer" in dpo_script
    assert "model.warnings_issued" in dpo_script
    assert "def _has_peft_adapters" in dpo_script
    assert "if not _has_peft_adapters(model):" in dpo_script
    assert 'dataset = dataset.add_column("images", [None] * len(dataset))' in dpo_script
    assert 'if original_model_type in {"qwen3_5", "qwen3_5_moe"}:' in dpo_script
    assert 'model.config.model_type = "qwen3_text"' in dpo_script
    assert 'dpo_processing_class = getattr(tokenizer, "tokenizer", tokenizer)' in dpo_script
    assert "processing_class=dpo_processing_class" in dpo_script
    assert 'parser.add_argument("--max-steps", type=int)' in dpo_script
    assert 'parser.add_argument("--resume-from-checkpoint")' in dpo_script
    assert 'parser.add_argument("--seed", type=int, default=43)' in dpo_script
    assert 'parser.add_argument("--save-steps", type=int, default=10)' in dpo_script
    assert 'parser.add_argument("--save-total-limit", type=int, default=3)' in dpo_script
    assert "max_steps=args.max_steps or -1" in dpo_script
    assert 'save_strategy="steps"' in dpo_script
    assert "save_steps=args.save_steps" in dpo_script
    assert "save_total_limit=args.save_total_limit" in dpo_script
    assert "trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)" in dpo_script
    assert '"stage": "dpo"' in dpo_script
    assert '"save_steps": args.save_steps' in dpo_script
    assert '"save_total_limit": args.save_total_limit' in dpo_script
    assert "training_stage_manifest.json" in dpo_script
    rl_script = (bundle_out / "training" / "run_reward_rl.py").read_text(encoding="utf-8")
    assert "_preferences_by_prompt(root, reward_config" in rl_script
    assert "_preferences_by_prompt(root / reward_config" not in rl_script
    assert "def _has_peft_adapters" in rl_script
    assert "if not _has_peft_adapters(model):" in rl_script
    assert 'parser.add_argument("--seed", type=int, default=44)' in rl_script
    assert '"--schema-prefix-mode"' in rl_script
    assert 'default="schema-scaffold"' in rl_script
    assert "def _schema_prefix" in rl_script
    assert "def _encode_generation_inputs" in rl_script
    assert "torch.manual_seed(args.seed)" in rl_script
    assert 'text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)' in rl_script
    assert "encoded = text_tokenizer(" in rl_script
    assert "tokenizer.apply_chat_template" in rl_script
    assert "enable_thinking=False" in rl_script
    assert "output_ids = output_ids.detach().clone().to(device)" in rl_script
    assert 'tokenizer.truncation_side = "left"' in rl_script
    assert "formatted_prompt = _format_generation_prompt" in rl_script
    assert "min_new_tokens=8" in rl_script
    assert "StoppingCriteriaList" in rl_script
    assert "def _has_complete_json_object" in rl_script
    assert "stopping_criteria=StoppingCriteriaList" in rl_script
    assert "raw_sir_unit = _extract_json_object(text)" in rl_script
    assert "sir_unit = _repair_sir_unit(" in rl_script
    assert "sir_unit = _apply_faithfulness_repair(" in rl_script
    assert 'elif unit.get("raw_error"):' in rl_script
    assert "reward += _raw_generation_bonus(" in rl_script
    assert 'record["reward_reference"],' in rl_script
    assert "repair_input = json.loads(json.dumps(raw_sir_unit))" in rl_script
    assert 'if raw_sir_unit.get(field) == expected:' in rl_script
    assert "schema_prefix_tokens" in rl_script
    assert "generated_tokens" in rl_script
    assert "schema-scaffold" in rl_script
    assert '"external_dependencies": []' in rl_script
    assert '"confidence": target.get("confidence"' in rl_script
    assert '"schema_prefix_mode": args.schema_prefix_mode' in rl_script
    assert "Copy the final SIR JSON object between" in rl_script
    assert "compact_token_budget" in rl_script
    assert "compact_expected_counts" in rl_script
    assert "return -80.0" in rl_script
    assert "len(missing_keys) * 4.0" in rl_script
    assert "len(extra_keys) * 5.0" in rl_script
    assert "extra_keys = set(raw_sir_unit) - allowed_keys" in rl_script
    assert "must start with" in rl_script
    assert '"raw_parseable"' in rl_script
    assert '"stage": "rl"' in rl_script
    assert "training_stage_manifest.json" in rl_script
    candidate_script = (bundle_out / "training" / "generate_sir_candidates.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument("--max-new-tokens", type=int, default=1536)' in candidate_script
    assert 'parser.add_argument("--max-prompts", type=int)' in candidate_script
    assert 'parser.add_argument("--prompt-file")' in candidate_script
    assert 'parser.add_argument("--no-faithfulness-repair", action="store_true")' in candidate_script
    assert '"--faithfulness-repair-mode"' in candidate_script
    assert '"schema-only"' in candidate_script
    assert '"compact-target"' in candidate_script
    assert '"full-static"' in candidate_script
    assert '"--generation-mode"' in candidate_script
    assert '"field-wise"' in candidate_script
    assert '"--field-max-new-tokens"' in candidate_script
    assert '"--field-target-mode"' in candidate_script
    assert '"static-facts"' in candidate_script
    assert '"static-hints"' in candidate_script
    assert '"--field-target-limit"' in candidate_script
    assert '"--field-target-max-chunks"' in candidate_script
    assert '"--field-target-chunk-fields"' in candidate_script
    assert '"--field-object-prefix-mode"' in candidate_script
    assert '"--schema-prefix-mode"' in candidate_script
    assert "def _generate_field_wise_candidate" in candidate_script
    assert "FIELD_STATIC_HINT_DEFAULT_LIMITS" in candidate_script
    assert '"reads": 2' in candidate_script
    assert '"external_dependencies": 1' in candidate_script
    assert 'or reference.get("source_spans")' in candidate_script
    assert 'or metadata.get("source_spans")' in candidate_script
    assert 'static_algorithm = reference.get("static_facts", {}).get("algorithm", {})' in candidate_script
    assert 'Semantic IR unit for {source_path}.' in candidate_script
    assert "def _compact_static_hint_value" in candidate_script
    assert "def _compact_static_hint_chunk" in candidate_script
    assert "def _merge_field_value" in candidate_script
    assert "def _parse_field_set" in candidate_script
    assert "Output budget for" in candidate_script
    assert "effective_field_target_limit" in candidate_script
    assert "field_target_chunk_index" in candidate_script
    assert "field_target_chunk_count" in candidate_script
    assert "field_target_chunk_enabled" in candidate_script
    assert "field_generation_reports" in candidate_script
    assert "object_prefix_tokens" in candidate_script
    assert 'object_prefix=f\'{{"{field}":\'' in candidate_script
    assert "def _schema_prefix" in candidate_script
    assert "def _encode_generation_inputs" in candidate_script
    assert "prompts_path = Path(args.prompt_file)" in candidate_script
    assert "prompts = prompts[: args.max_prompts]" in candidate_script
    assert 'tokenizer.truncation_side = "left"' in candidate_script
    assert 'text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)' in candidate_script
    assert "tokenizer.apply_chat_template" in candidate_script
    assert "enable_thinking=False" in candidate_script
    assert "formatted_prompt = _format_generation_prompt" in candidate_script
    assert "or use Markdown fences" in candidate_script
    assert "Copy the final SIR JSON object between" in candidate_script
    assert "Do not shorten unit_id" in candidate_script
    assert "safety_report" in candidate_script
    assert "must start with" in candidate_script
    assert "completion_tokens" in candidate_script
    assert "hit_generation_cap" in candidate_script
    assert "raw_parse_error" in candidate_script
    assert "StoppingCriteriaList" in candidate_script
    assert "def _has_complete_json_object" in candidate_script
    assert "stopping_criteria=StoppingCriteriaList" in candidate_script
    assert "def _assistant_completion_region" in candidate_script
    assert "repair_input = json.loads(json.dumps(raw_sir_unit))" in candidate_script
    assert "sir_unit = _repair_sir_unit(" in candidate_script
    assert "sir_unit = _apply_faithfulness_repair(" in candidate_script
    assert 'elif unit.get("raw_error"):' in candidate_script
    assert "DATA_ML_DETAIL_CATEGORIES" in candidate_script
    assert "continue the input JSON" in candidate_script
    assert "generation_tokens = min(" in candidate_script
    assert "max_prompt_tokens = max(128" in candidate_script
    assert "truncation=True" in candidate_script
    assert "max_length=prompt_limit" in candidate_script
    assert "prompt_len = int(input_ids.shape[1])" in candidate_script
    assert "completion_ids = output_ids[:, prompt_len:]" in candidate_script
    assert "text_tokenizer.decode(completion_ids[0]" in candidate_script
    assert '"schema_prefix_mode": args.schema_prefix_mode' in candidate_script
    assert '"faithfulness_repair_mode": args.faithfulness_repair_mode' in candidate_script
    assert '"schema_prefix_applied": bool(schema_prefix)' in candidate_script
    assert '"generated_tokens": generated_tokens' in candidate_script
    assert '"generation_mode": args.generation_mode' in candidate_script
    assert '"field_target_mode": args.field_target_mode' in candidate_script
    assert '"field_target_limit": args.field_target_limit' in candidate_script
    assert '"field_object_prefix_mode": args.field_object_prefix_mode' in candidate_script
    assert "schema-scaffold" in candidate_script
    assert 'default="schema-scaffold"' in candidate_script
    assert '"external_dependencies": []' in candidate_script
    assert '"confidence": target.get("confidence"' in candidate_script
    assert 'parser.add_argument("--raw-out")' in candidate_script
    assert '"raw_parseable"' in candidate_script
    assert '"raw_sir_unit"' in candidate_script

    fake_run = tmp_path / "fake_run"
    fake_run.mkdir()
    (fake_run / "sft_stdout.log").write_text(
        "{'loss': '0.5', 'epoch': '0.1'}\n{'loss': '0.25', 'epoch': '0.2'}\n",
        encoding="utf-8",
    )
    (fake_run / "dpo_stdout.log").write_text(
        "{'loss': '0.4', 'reward_accuracy': '0.75'}\n",
        encoding="utf-8",
    )
    rl_dir = fake_run / "semantic-mirror-rl"
    rl_dir.mkdir()
    (rl_dir / "rl_training_report.json").write_text(
        json.dumps(
            {
                "stage": "rl",
                "history": [
                    {"step": 0, "reward": 1.0, "raw_parseable": False, "loss": 0.3},
                    {"step": 1, "reward": 2.0, "raw_parseable": True, "loss": 0.2},
                ]
            }
        ),
        encoding="utf-8",
    )
    diagnostics = generate_training_diagnostics(fake_run)
    diagnostics_out = fake_run / "diagnostics"
    assert diagnostics["plots"]["sft_loss"]["points"] == 2
    assert diagnostics["plots"]["dpo_reward_accuracy"]["points"] == 1
    assert diagnostics["plots"]["rl_parseability"]["points"] == 2
    assert (diagnostics_out / "training_summary.json").exists()
    assert (diagnostics_out / "training_summary.md").exists()
    assert (diagnostics_out / "sft_loss.png").read_bytes().startswith(b"\x89PNG")

    sample_record = silver[0]
    raw_candidates = tmp_path / "raw_candidates.jsonl"
    repaired_candidates = tmp_path / "repaired_candidates.jsonl"
    raw_candidates.write_text(
        json.dumps(
            {
                "record_id": "raw-1",
                "dataset_record_id": sample_record["record_id"],
                "unit_id": sample_record["unit_id"],
                "source_path": sample_record["source_path"],
                "profile": sample_record["profile"],
                "zoom": sample_record["zoom"],
                "raw_text": "not json",
                "raw_parseable": False,
                "raw_parse_error": "no JSON",
                "hit_generation_cap": True,
                "sir_unit": {"unit_id": "<unparseable>", "raw_error": "no JSON"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    repaired_candidates.write_text(
        json.dumps(
            {
                "record_id": "repaired-1",
                "dataset_record_id": sample_record["record_id"],
                "unit_id": sample_record["unit_id"],
                "source_path": sample_record["source_path"],
                "profile": sample_record["profile"],
                "zoom": sample_record["zoom"],
                "raw_text": "not json",
                "sir_unit": sample_record["target"]["sir_unit"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    sample_manifest = create_sample_inspection(
        out,
        raw_candidates_path=raw_candidates,
        repaired_candidates_path=repaired_candidates,
        out_path=tmp_path / "samples",
        model_name="unit-test-model",
        generation_config={"max_new_tokens": 32, "seed": 42},
    )
    assert sample_manifest["raw_parseability_count"] == 0
    assert sample_manifest["raw_generation_cap_hits"] == 1
    assert sample_manifest["raw_repair_free_contract_count"] == 0
    assert sample_manifest["raw_exact_identity_count"] == 0
    assert sample_manifest["raw_top_level_key_validity_count"] == 0
    assert sample_manifest["raw_compact_shape_count"] == 0
    assert sample_manifest["repaired_schema_validity_count"] == 1
    inspection = (tmp_path / "samples" / "sample_inspection.md").read_text(encoding="utf-8")
    assert "Raw generation cap hits: 1 / 1" in inspection
    assert "Raw repair-free contract valid: 0 / 1" in inspection
    assert "Raw parse error: `no JSON`" in inspection
    assert "Raw repair-free contract valid: `False`" in inspection
    assert (tmp_path / "samples" / "raw_eval.json").exists()
    assert (tmp_path / "samples" / "repaired_eval.json").exists()
    assert (tmp_path / "samples" / "sample_inspection.md").exists()

    contaminated_raw = tmp_path / "contaminated_raw_candidates.jsonl"
    malformed_raw_unit = {
        "unit_id": "short-id",
        "qualified_name": sample_record["qualified_name"],
        "language": "python",
        "source_spans": sample_record["source_spans"],
    }
    contaminated_raw.write_text(
        json.dumps(
            {
                "record_id": "raw-contaminated",
                "dataset_record_id": sample_record["record_id"],
                "unit_id": sample_record["unit_id"],
                "source_path": sample_record["source_path"],
                "profile": sample_record["profile"],
                "zoom": sample_record["zoom"],
                "raw_text": json.dumps(malformed_raw_unit, separators=(",", ":")),
                "raw_parseable": True,
                "repair_applied": False,
                "hit_generation_cap": False,
                "sir_unit": sample_record["target"]["sir_unit"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    contaminated_manifest = create_sample_inspection(
        out,
        raw_candidates_path=contaminated_raw,
        repaired_candidates_path=repaired_candidates,
        out_path=tmp_path / "contaminated_samples",
        model_name="contaminated-raw-model",
    )
    assert contaminated_manifest["raw_parseability_count"] == 1
    assert contaminated_manifest["raw_schema_validity_count"] == 0
    assert contaminated_manifest["raw_repair_free_contract_count"] == 0
    assert contaminated_manifest["raw_contract_reports"][0]["raw_unit_id"] == "short-id"
    assert "unit_id" in contaminated_manifest["raw_contract_reports"][0]["identity_mismatches"]
    assert (bundle_out / "src" / "semantic_mirror" / "training.py").exists()
    assert (bundle_out / "pyproject.toml").exists()
    requirements = (bundle_out / "requirements-training.txt").read_text(encoding="utf-8")
    assert "-e ." in requirements
    assert "unsloth" in requirements
    assert "mergekit" in requirements
    assert "llm-blender" in requirements
    assert "peft" in requirements
    assert "weave" in requirements
    package_pyproject = (bundle_out / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11,<3.14"' in package_pyproject
    commands = json.loads((bundle_out / "launch" / "commands.json").read_text(encoding="utf-8"))
    command_manifest = json.loads(
        (bundle_out / "launch" / "commands_manifest.json").read_text(encoding="utf-8")
    )
    manifest_commands = command_manifest["commands"]
    assert "bootstrap_linux_cuda.sh" in commands["bootstrap_linux_cuda"]
    assert command_manifest["schema_version"] == 1
    assert manifest_commands["full_training_eval"]["command"] == commands["full_training_eval"]
    assert manifest_commands["full_training_eval"]["category"] == "training"
    assert manifest_commands["full_training_eval"]["launches_training"] is True
    assert manifest_commands["full_training_eval"]["required_inputs"] == [
        "held_out_dataset",
        "baseline_candidates",
    ]
    assert manifest_commands["smoke_chain"]["launches_training"] is True
    assert manifest_commands["wsl_smoke_chain"]["launches_training"] is True
    assert manifest_commands["wsl_smoke_chain"]["required_inputs"] == ["held_out_dataset"]
    assert manifest_commands["preflight_wsl_smoke_inputs"]["category"] == "validation"
    assert manifest_commands["preflight_wsl_smoke_inputs"]["launches_training"] is False
    assert manifest_commands["preflight_wsl_smoke_inputs"]["required_inputs"] == [
        "held_out_dataset"
    ]
    assert manifest_commands["preflight_full_eval_inputs"]["category"] == "validation"
    assert manifest_commands["preflight_full_eval_inputs"]["launches_training"] is False
    assert manifest_commands["preflight_full_eval_inputs"]["required_inputs"] == [
        "held_out_dataset",
        "baseline_candidates",
    ]
    assert manifest_commands["inspect_full_training_eval_resume"]["category"] == "inspection"
    assert manifest_commands["inspect_full_training_eval_resume"]["launches_training"] is False
    assert manifest_commands["inspect_resume"]["category"] == "inspection"
    assert manifest_commands["inspect_resume"]["launches_training"] is False
    assert manifest_commands["contract_status"]["category"] == "status"
    assert manifest_commands["contract_status"]["launches_training"] is False
    assert manifest_commands["report"]["category"] == "diagnostics"
    assert manifest_commands["report"]["launches_training"] is False
    assert "run_full_training_eval.sh" in commands["full_training_eval"]
    assert "preflight_full_eval_inputs.ps1" in commands["preflight_full_eval_inputs"]
    assert "-BaselineCandidates <windows_baseline_candidates_jsonl>" in commands[
        "preflight_full_eval_inputs"
    ]
    assert "run_smoke_chain.sh" in commands["smoke_chain"]
    assert "run_wsl_smoke_chain.ps1" in commands["wsl_smoke_chain"]
    assert "preflight_wsl_smoke_inputs.ps1" in commands["preflight_wsl_smoke_inputs"]
    assert "-HeldOutDataset <windows_dataset_dir>" in commands["wsl_smoke_chain"]
    assert "--out outputs/validation_report.json" in commands["validate"]
    assert "--out outputs/audit.json" in commands["audit"]
    assert "train inspect-samples" in commands["inspect_samples"]
    assert "train report outputs" in commands["report"]
    assert "train source-freshness ." in commands["source_freshness"]
    assert "--out source_freshness.json" in commands["source_freshness"]
    assert "train inspect-resume outputs" in commands["inspect_resume"]
    assert "--markdown-out outputs/full_training_eval_resume_inspection.md" in commands[
        "inspect_resume"
    ]
    assert "train contract-status outputs" in commands["contract_status"]
    assert "--sft-steps $SFT_MAX_STEPS" in commands["contract_status"]
    assert "--windows-audit ${WINDOWS_AUDIT:-audit/current_environment.json}" in commands[
        "contract_status"
    ]
    assert (
        "--wsl-smoke-manifest ${WSL_SMOKE_MANIFEST:-outputs/smoke-chain-wsl/smoke_chain_manifest.json}"
        in commands["contract_status"]
    )
    assert "--package-source-freshness source_freshness.json" in commands["contract_status"]
    assert "--markdown-out outputs/contract_status.md" in commands["contract_status"]
    assert "run_unsloth_sft.py" in commands["sft"]
    assert "run_reward_rl.py" in commands["rl"]
    assert manifest_commands["dpo"]["required_inputs"] == [
        "training_dir",
        "sft_model_or_adapter",
        "output_dir",
    ]
    assert manifest_commands["rl"]["required_inputs"] == [
        "training_dir",
        "dpo_or_sft_model_or_adapter",
        "output_dir",
    ]
    assert manifest_commands["contract_status"]["optional_inputs"] == [
        "repo_root",
        "windows_audit",
        "wsl_smoke_manifest",
        "package_source_freshness",
        "human_study_coverage",
        "human_study_suite",
    ]
    assert manifest_commands["full_training_eval"]["optional_inputs"] == [
        "source_freshness_repo_root",
        "windows_audit",
        "wsl_smoke_manifest",
        "package_source_freshness",
        "human_study_coverage",
        "human_study_suite",
    ]
    assert manifest_commands["inspect_full_training_eval_resume"]["optional_inputs"] == [
        "sft_resume_from_checkpoint",
        "dpo_resume_from_checkpoint",
    ]
    assert package_manifest["command_input_reference"]["held_out_dataset"] == {
        "purpose": "Held-out Semantic Mirror dataset directory used for smoke or full eval.",
        "example": "outputs/heldout_eval_dataset",
        "required_by": [
            "eval_candidates",
            "full_training_eval",
            "preflight_full_eval_inputs",
            "preflight_wsl_smoke_inputs",
            "smoke_chain",
            "wsl_smoke_chain",
        ],
        "optional_by": [],
    }
    assert package_manifest["command_input_reference"]["windows_audit"] == {
        "purpose": "Native Windows readiness audit JSON used as contract-status evidence.",
        "example": "audit/current_environment.json",
        "required_by": [],
        "optional_by": ["contract_status", "full_training_eval"],
    }
    assert "--schema-prefix-mode schema-scaffold" in commands["rl"]
    assert "--schema-prefix-mode schema-scaffold" in commands["generate_candidates"]
    assert "outputs/dpo_eval.json" in commands["compare_dpo"]
    assert "--stage dpo" in commands["compare_dpo"]
    assert "outputs/sft_raw_eval.json" in commands["compare_sft_raw"]
    assert "outputs/dpo_raw_vs_sft.json" in commands["compare_dpo_raw"]
    assert "outputs/rl_raw_vs_sft.json" in commands["compare_rl_raw"]
    assert (bundle_out / "setup" / "bootstrap_linux_cuda.sh").exists()
    assert (bundle_out / "setup" / "bootstrap_wsl_ubuntu.ps1").exists()
    bootstrap = (bundle_out / "setup" / "bootstrap_linux_cuda.sh").read_text(encoding="utf-8")
    assert "3.11" in bootstrap
    assert "3.14" in bootstrap
    assert package_manifest["files"]["wsl_smoke_chain_launcher"] == "launch/run_wsl_smoke_chain.ps1"
    assert (
        package_manifest["files"]["wsl_smoke_input_preflight"]
        == "launch/preflight_wsl_smoke_inputs.ps1"
    )
    assert (
        package_manifest["files"]["full_training_eval_input_preflight"]
        == "launch/preflight_full_eval_inputs.sh"
    )
    assert (
        package_manifest["files"]["full_training_eval_input_preflight_windows"]
        == "launch/preflight_full_eval_inputs.ps1"
    )
    assert package_manifest["files"]["launch_command_manifest"] == "launch/commands_manifest.json"
    assert (
        package_manifest["launch_command_manifest"]["commands"]["full_training_eval"][
            "launches_training"
        ]
        is True
    )
    assert package_manifest["launch_command_manifest"]["commands"]["dpo"][
        "required_inputs"
    ] == ["training_dir", "sft_model_or_adapter", "output_dir"]
    assert package_manifest["launch_command_manifest"]["commands"]["rl"][
        "required_inputs"
    ] == ["training_dir", "dpo_or_sft_model_or_adapter", "output_dir"]
    assert (
        package_manifest["launch_command_manifest"]["commands"]["contract_status"][
            "launches_training"
        ]
        is False
    )
    assert (bundle_out / "launch" / "run_sft.sh").exists()
    assert (bundle_out / "launch" / "run_rl.sh").exists()
    assert (bundle_out / "launch" / "run_smoke_chain.sh").exists()
    assert (bundle_out / "launch" / "preflight_wsl_smoke_inputs.ps1").exists()
    assert (bundle_out / "launch" / "run_wsl_smoke_chain.ps1").exists()
    assert (bundle_out / "launch" / "preflight_full_eval_inputs.sh").exists()
    assert (bundle_out / "launch" / "preflight_full_eval_inputs.ps1").exists()
    assert (bundle_out / "launch" / "run_full_training_eval.sh").exists()
    for shell_script in [
        bundle_out / "setup" / "bootstrap_linux_cuda.sh",
        bundle_out / "launch" / "run_sft.sh",
        bundle_out / "launch" / "run_dpo.sh",
        bundle_out / "launch" / "run_rl.sh",
        bundle_out / "launch" / "generate_candidates.sh",
        bundle_out / "launch" / "score_candidates.sh",
        bundle_out / "launch" / "preflight_full_eval_inputs.sh",
        bundle_out / "launch" / "run_smoke_chain.sh",
        bundle_out / "launch" / "run_full_training_eval.sh",
    ]:
        assert b"\r\n" not in shell_script.read_bytes()
    smoke_chain_script = (bundle_out / "launch" / "run_smoke_chain.sh").read_text(
        encoding="utf-8"
    )
    assert "train validate training" in smoke_chain_script
    assert '--out "$SMOKE_OUT/validation_report.json"' in smoke_chain_script
    assert "train audit training" in smoke_chain_script
    assert "train inspect-samples" in smoke_chain_script
    assert "train report" in smoke_chain_script
    assert "smoke_chain_manifest.json" in smoke_chain_script
    assert "sample_rollup" in smoke_chain_script
    assert "raw_parseability_rate" in smoke_chain_script
    assert "raw_schema_validity_gate_passed" in smoke_chain_script
    assert "raw_repair_free_contract_gate_passed" in smoke_chain_script
    assert "repaired_schema_validity_gate_passed" in smoke_chain_script
    assert "all_stage_outputs_exist" in smoke_chain_script
    assert "all_sample_manifests_exist" in smoke_chain_script
    wsl_smoke_script = (bundle_out / "launch" / "run_wsl_smoke_chain.ps1").read_text(
        encoding="utf-8"
    )
    assert "wslpath -a" in wsl_smoke_script
    assert "$windowsPathForWsl = $windowsPath -replace '\\\\', '/'" in wsl_smoke_script
    assert "$datasetPathForWsl = $datasetPath -replace '\\\\', '/'" in wsl_smoke_script
    assert '[string]$VenvPath = ".venv"' in wsl_smoke_script
    assert ".run_wsl_smoke_chain.generated.sh" in wsl_smoke_script
    assert '$bashScript = $bashScript -replace "`r`n", "`n"' in wsl_smoke_script
    assert "System.Text.UTF8Encoding($false)" in wsl_smoke_script
    assert "[System.IO.File]::WriteAllText($scriptPath, $bashScript, $utf8NoBom)" in wsl_smoke_script
    assert 'wsl.exe -d $Distro -- bash "$scriptWslPath"' in wsl_smoke_script
    assert "wsl_smoke_environment.json" in wsl_smoke_script
    assert '"wsl_repo_path"' in wsl_smoke_script
    assert '"python_executable"' in wsl_smoke_script
    assert '"venv_path"' in wsl_smoke_script
    assert '"cuda_device"' in wsl_smoke_script
    assert '"output_path"' in wsl_smoke_script
    assert "HELD_OUT_DATASET='__HELD_OUT_WSL__'" in wsl_smoke_script
    assert "bash launch/run_smoke_chain.sh" in wsl_smoke_script
    wsl_preflight_script = (
        bundle_out / "launch" / "preflight_wsl_smoke_inputs.ps1"
    ).read_text(encoding="utf-8")
    assert "wsl_smoke_input_preflight" in wsl_preflight_script
    assert "manifest.json" in wsl_preflight_script
    assert "outputs/preflight/wsl_smoke_inputs.json" in wsl_preflight_script
    package_readme = (bundle_out / "README.md").read_text(encoding="utf-8")
    assert "preflight_wsl_smoke_inputs.ps1" in package_readme
    assert "run_wsl_smoke_chain.ps1" in package_readme
    assert "wsl_smoke_environment.json" in package_readme
    assert "CUDA device" in package_readme
    wsl_bootstrap = (bundle_out / "setup" / "bootstrap_wsl_ubuntu.ps1").read_text(
        encoding="utf-8"
    )
    assert "$windowsPathForWsl = $windowsPath -replace '\\\\', '/'" in wsl_bootstrap
    assert "all_repaired_samples_schema_valid" in smoke_chain_script
    assert "SFT_SMOKE_STEPS" in smoke_chain_script
    assert "SMOKE_SCHEMA_PREFIX_MODE" in smoke_chain_script
    assert 'SMOKE_SCHEMA_PREFIX_MODE="${SMOKE_SCHEMA_PREFIX_MODE:-schema-scaffold}"' in smoke_chain_script
    assert "SMOKE_GENERATION_MODE" in smoke_chain_script
    assert "SMOKE_FIELD_MAX_NEW_TOKENS" in smoke_chain_script
    assert "SMOKE_FIELD_TARGET_MODE" in smoke_chain_script
    assert "SMOKE_FIELD_TARGET_LIMIT" in smoke_chain_script
    assert "SMOKE_FIELD_TARGET_MAX_CHUNKS" in smoke_chain_script
    assert "SMOKE_FIELD_TARGET_CHUNK_FIELDS" in smoke_chain_script
    assert "SMOKE_FIELD_OBJECT_PREFIX_MODE" in smoke_chain_script
    assert "SMOKE_FAITHFULNESS_REPAIR_MODE" in smoke_chain_script
    assert 'SMOKE_FAITHFULNESS_REPAIR_MODE="${SMOKE_FAITHFULNESS_REPAIR_MODE:-schema-only}"' in smoke_chain_script
    assert '--generation-mode "$SMOKE_GENERATION_MODE"' in smoke_chain_script
    assert '--field-target-mode "$SMOKE_FIELD_TARGET_MODE"' in smoke_chain_script
    assert '--field-target-limit "$SMOKE_FIELD_TARGET_LIMIT"' in smoke_chain_script
    assert '--field-target-max-chunks "$SMOKE_FIELD_TARGET_MAX_CHUNKS"' in smoke_chain_script
    assert '--field-target-chunk-fields "$SMOKE_FIELD_TARGET_CHUNK_FIELDS"' in smoke_chain_script
    assert '--field-object-prefix-mode "$SMOKE_FIELD_OBJECT_PREFIX_MODE"' in smoke_chain_script
    assert '--faithfulness-repair-mode "$SMOKE_FAITHFULNESS_REPAIR_MODE"' in smoke_chain_script
    assert '--schema-prefix-mode "$SMOKE_SCHEMA_PREFIX_MODE"' in smoke_chain_script
    full_eval_script = (bundle_out / "launch" / "run_full_training_eval.sh").read_text(
        encoding="utf-8"
    )
    full_eval_preflight_script = (
        bundle_out / "launch" / "preflight_full_eval_inputs.sh"
    ).read_text(encoding="utf-8")
    full_eval_preflight_ps = (
        bundle_out / "launch" / "preflight_full_eval_inputs.ps1"
    ).read_text(encoding="utf-8")
    resume_inspector = (
        bundle_out / "launch" / "inspect_full_training_eval_resume.sh"
    ).read_text(encoding="utf-8")
    launch_commands = json.loads(
        (bundle_out / "launch" / "commands.json").read_text(encoding="utf-8")
    )
    package_readme = (bundle_out / "README.md").read_text(encoding="utf-8")
    assert "full_eval_input_preflight" in full_eval_preflight_script
    assert "baseline_candidates is empty" in full_eval_preflight_script
    assert "baseline_candidates line {line_number} is invalid JSON" in full_eval_preflight_script
    assert "baseline_candidate_identifier_records" in full_eval_preflight_script
    assert "baseline_candidate_matching_identifier_count" in full_eval_preflight_script
    assert "baseline_candidates identifiers do not match held_out_dataset records" in full_eval_preflight_script
    assert "held_out_dataset manifest missing files.{key}" in full_eval_preflight_script
    assert "outputs/preflight/full_eval_inputs.json" in full_eval_preflight_script
    assert "full_eval_input_preflight" in full_eval_preflight_ps
    assert "baseline_candidates is empty" in full_eval_preflight_ps
    assert "baseline_candidates line $lineNumber is invalid JSON" in full_eval_preflight_ps
    assert "baseline_candidate_identifier_records" in full_eval_preflight_ps
    assert "baseline_candidate_matching_identifier_count" in full_eval_preflight_ps
    assert "baseline_candidates identifiers do not match held_out_dataset records" in full_eval_preflight_ps
    assert "held_out_dataset manifest missing files.$key" in full_eval_preflight_ps
    assert "outputs/preflight/full_eval_inputs.json" in full_eval_preflight_ps
    assert "[string]$BaselineCandidates" in full_eval_preflight_ps
    assert "preflight_full_eval_inputs.ps1" in package_readme
    assert "-BaselineCandidates C:\\path\\to\\teacher_results\\teacher_candidates.jsonl" in package_readme
    assert "HUMAN_STUDY_SUITE=outputs/phase6/phase6_real_suite_summary.json" in package_readme
    assert "HUMAN_STUDY_COVERAGE=outputs/phase6/whole_repo_real_coverage.json:outputs/phase6/diff_mode_real_coverage.json" in package_readme
    assert "--human-study-coverage outputs/phase6/whole_repo_real_coverage.json" in package_readme
    assert "--human-study-suite outputs/phase6/phase6_real_suite_summary.json" in package_readme
    assert "bash launch/preflight_full_eval_inputs.sh" in full_eval_script
    assert "Missing executable launch/preflight_full_eval_inputs.sh" in full_eval_script
    assert full_eval_script.index("bash launch/preflight_full_eval_inputs.sh") < (
        full_eval_script.index("train validate training --out outputs/validation_report.json")
    )
    assert "eval candidates" in full_eval_script
    assert "eval model-compare" in full_eval_script
    assert "train validate training --out outputs/validation_report.json" in full_eval_script
    assert "train audit training --out outputs/audit.json" in full_eval_script
    assert "outputs/heldout_eval_dataset" in full_eval_script
    assert "outputs/heldout_eval_training" in full_eval_script
    assert "outputs/baseline_candidates_eval.jsonl" in full_eval_script
    assert "--prompt-file outputs/heldout_eval_training/rl_prompts.jsonl" in full_eval_script
    assert "SCHEMA_PREFIX_MODE" in full_eval_script
    assert 'SCHEMA_PREFIX_MODE="${SCHEMA_PREFIX_MODE:-schema-scaffold}"' in full_eval_script
    assert 'FAITHFULNESS_REPAIR_MODE="${FAITHFULNESS_REPAIR_MODE:-schema-only}"' in full_eval_script
    assert 'REUSE_STAGE_OUTPUTS="${REUSE_STAGE_OUTPUTS:-0}"' in full_eval_script
    assert 'SFT_RESUME_FROM_CHECKPOINT="${SFT_RESUME_FROM_CHECKPOINT:-}"' in full_eval_script
    assert 'DPO_RESUME_FROM_CHECKPOINT="${DPO_RESUME_FROM_CHECKPOINT:-}"' in full_eval_script
    assert 'HUMAN_STUDY_SUITE="${HUMAN_STUDY_SUITE:-}"' in full_eval_script
    assert 'HUMAN_STUDY_COVERAGE="${HUMAN_STUDY_COVERAGE:-}"' in full_eval_script
    assert 'SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-10}"' in full_eval_script
    assert 'DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-10}"' in full_eval_script
    assert 'SFT_SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-3}"' in full_eval_script
    assert 'DPO_SAVE_TOTAL_LIMIT="${DPO_SAVE_TOTAL_LIMIT:-3}"' in full_eval_script
    assert "stage_ready()" in full_eval_script
    assert 'local requested_steps="$2"' in full_eval_script
    assert 'manifest.get("max_steps") == int(sys.argv[2])' in full_eval_script
    assert 'stage_ready outputs/semantic-mirror-sft "$SFT_MAX_STEPS"' in full_eval_script
    assert 'stage_ready outputs/semantic-mirror-dpo "$DPO_MAX_STEPS"' in full_eval_script
    assert 'stage_ready outputs/semantic-mirror-rl "$RL_MAX_STEPS"' in full_eval_script
    assert "Reusing existing SFT stage output" in full_eval_script
    assert "Reusing existing DPO stage output" in full_eval_script
    assert "Reusing existing RL stage output" in full_eval_script
    assert 'sft_resume_args=(--resume-from-checkpoint "$SFT_RESUME_FROM_CHECKPOINT")' in full_eval_script
    assert 'dpo_resume_args=(--resume-from-checkpoint "$DPO_RESUME_FROM_CHECKPOINT")' in full_eval_script
    assert '--save-steps "$SFT_SAVE_STEPS"' in full_eval_script
    assert '--save-total-limit "$SFT_SAVE_TOTAL_LIMIT"' in full_eval_script
    assert '--save-steps "$DPO_SAVE_STEPS"' in full_eval_script
    assert '--save-total-limit "$DPO_SAVE_TOTAL_LIMIT"' in full_eval_script
    assert '"schema_prefix_mode": sys.argv[5]' in full_eval_script
    assert "GENERATION_MODE" in full_eval_script
    assert '"generation_mode": sys.argv[6]' in full_eval_script
    assert '"field_max_new_tokens": int(sys.argv[7])' in full_eval_script
    assert '"field_target_mode": sys.argv[8]' in full_eval_script
    assert '"field_target_limit": int(sys.argv[9])' in full_eval_script
    assert '"field_target_max_chunks": int(sys.argv[10])' in full_eval_script
    assert '"field_target_chunk_fields": [item for item in sys.argv[11].split(",") if item]' in full_eval_script
    assert '"field_object_prefix_mode": sys.argv[12]' in full_eval_script
    assert '"faithfulness_repair_mode": sys.argv[13]' in full_eval_script
    assert '"checkpoint_policy": {' in full_eval_script
    assert '"sft_save_steps": int(sys.argv[21])' in full_eval_script
    assert '"dpo_save_steps": int(sys.argv[22])' in full_eval_script
    assert '"sft_save_total_limit": int(sys.argv[23])' in full_eval_script
    assert '"dpo_save_total_limit": int(sys.argv[24])' in full_eval_script
    assert '--generation-mode "$GENERATION_MODE"' in full_eval_script
    assert '--field-target-mode "$FIELD_TARGET_MODE"' in full_eval_script
    assert '--field-target-limit "$FIELD_TARGET_LIMIT"' in full_eval_script
    assert '--field-target-max-chunks "$FIELD_TARGET_MAX_CHUNKS"' in full_eval_script
    assert '--field-target-chunk-fields "$FIELD_TARGET_CHUNK_FIELDS"' in full_eval_script
    assert '--field-object-prefix-mode "$FIELD_OBJECT_PREFIX_MODE"' in full_eval_script
    assert '--faithfulness-repair-mode "$FAITHFULNESS_REPAIR_MODE"' in full_eval_script
    assert '--schema-prefix-mode "$SCHEMA_PREFIX_MODE"' in full_eval_script
    assert "generate_and_inspect dpo" in full_eval_script
    assert "train inspect-samples" in full_eval_script
    assert "dpo_vs_sft.json" in full_eval_script
    assert "dpo_raw_eval.json" in full_eval_script
    assert "sft_raw_vs_baseline.json" in full_eval_script
    assert "dpo_raw_vs_sft.json" in full_eval_script
    assert "rl_raw_vs_sft.json" in full_eval_script
    assert "train report outputs" in full_eval_script
    assert '--human-study-suite "$HUMAN_STUDY_SUITE"' in full_eval_script
    assert 'IFS=\':\' read -r -a human_study_coverage_paths' in full_eval_script
    assert '--human-study-coverage "$human_study_coverage_path"' in full_eval_script
    assert "required_reports" in full_eval_script
    assert "diagnostic_reports" in full_eval_script
    assert "eval_run_config" in full_eval_script
    assert "stage_execution_summary" in full_eval_script
    assert "requested_max_steps" in full_eval_script
    assert "manifest_max_steps" in full_eval_script
    assert "reuse_mode_enabled" in full_eval_script
    assert "manifest_matches_requested_max_steps" in full_eval_script
    assert "gate_policy" in full_eval_script
    assert "diagnostic_non_blocking" in full_eval_script
    assert "gate_counts" in full_eval_script
    assert 'summary["required_total"] = summary["gate_counts"]["required_total"]' in full_eval_script
    assert 'summary["required_passed"] = summary["gate_counts"]["required_passed"]' in full_eval_script
    assert 'summary["all_final_eval_gates_passed"] = summary["final_eval_gate_summary"]' in full_eval_script
    assert "raw_gate_summary" in full_eval_script
    assert "raw_gate_stage_summary" in full_eval_script
    assert "raw_compare_deltas" in full_eval_script
    assert "final_eval_gate_summary" in full_eval_script
    assert "all_final_eval_gates_passed" in full_eval_script
    assert "rl_raw_hallucination_not_worse_than_sft" in full_eval_script
    assert "raw_repair_free_contract_stretch_passed" in full_eval_script
    assert "raw_candidate_count" in full_eval_script
    assert "raw_parseability_rate" in full_eval_script
    assert "raw_parseability_gate_passed" in full_eval_script
    assert "raw_schema_validity_gate_passed" in full_eval_script
    assert "raw_repair_free_contract_gate_passed" in full_eval_script
    assert '"validation_report": "outputs/validation_report.json"' in full_eval_script
    assert '"audit_report": "outputs/audit.json"' in full_eval_script
    assert '"sft_raw_vs_baseline": "outputs/sft_raw_vs_baseline.json"' in full_eval_script
    assert '"dpo_raw_vs_sft": "outputs/dpo_raw_vs_sft.json"' in full_eval_script
    assert '"rl_raw_vs_sft": "outputs/rl_raw_vs_sft.json"' in full_eval_script
    assert "raw_parseability_count" in full_eval_script
    assert "raw_generation_cap_hits" in full_eval_script
    assert "raw_repair_free_contract_count" in full_eval_script
    assert "raw_exact_identity_count" in full_eval_script
    assert "raw_top_level_key_validity_count" in full_eval_script
    assert "raw_compact_shape_count" in full_eval_script
    assert "training_eval_summary.json" in full_eval_script
    assert "train contract-status outputs" in full_eval_script
    assert "--sft-steps \"$SFT_MAX_STEPS\"" in full_eval_script
    assert "SOURCE_FRESHNESS_REPO_ROOT" in full_eval_script
    assert "train source-freshness ." in full_eval_script
    assert "--package-source-freshness \"$PACKAGE_SOURCE_FRESHNESS\"" in full_eval_script
    assert 'WINDOWS_AUDIT="${WINDOWS_AUDIT:-audit/current_environment.json}"' in full_eval_script
    assert (
        'WSL_SMOKE_MANIFEST="${WSL_SMOKE_MANIFEST:-outputs/smoke-chain-wsl/smoke_chain_manifest.json}"'
        in full_eval_script
    )
    assert "--windows-audit \"$WINDOWS_AUDIT\"" in full_eval_script
    assert "--wsl-smoke-manifest \"$WSL_SMOKE_MANIFEST\"" in full_eval_script
    assert "--out outputs/contract_status.json" in full_eval_script
    assert "--markdown-out outputs/contract_status.md" in full_eval_script
    assert "source_freshness.json" in package_readme
    assert "train source-freshness" in package_readme
    assert "preflight_full_eval_inputs.sh" in package_readme
    assert "commands_manifest.json" in package_readme
    assert "launches_training" in package_readme
    assert "## Command Input Reference" in package_readme
    assert "| `held_out_dataset` | Held-out Semantic Mirror dataset directory" in package_readme
    assert "`outputs/heldout_eval_dataset`" in package_readme
    assert (
        "`eval_candidates`, `full_training_eval`, `preflight_full_eval_inputs`, "
        "`preflight_wsl_smoke_inputs`, `smoke_chain`, `wsl_smoke_chain`"
        in package_readme
    )
    assert "| `baseline_candidates` | JSONL candidates for the baseline model" in package_readme
    assert "| `outputs_dir` | Full-eval output directory" in package_readme
    assert "| `windows_audit` | Native Windows readiness audit JSON" in package_readme
    assert "| `wsl_smoke_manifest` | WSL smoke-chain manifest" in package_readme
    assert "contract_status.json" in package_readme
    assert "contract_status.md" in package_readme
    assert "`outputs/contract_status.json` and `train contract-status` stdout" in (
        package_readme
    )
    assert "The JSON keeps full `next_actions` commands" in package_readme
    assert "stdout compacts those actions into" in package_readme
    assert "--windows-audit audit/current_environment.json" in package_readme
    assert "--wsl-smoke-manifest outputs/smoke-chain-wsl/smoke_chain_manifest.json" in package_readme
    assert "train inspect-resume outputs" in package_readme
    assert "full_training_eval_resume_inspection.md" in package_readme
    assert "contract_scorecard_summary" in package_readme
    assert "stage_recovery_summary" in package_readme
    assert "remaining_area_summary" in package_readme
    assert "training_dependency_summary" in package_readme
    assert "training-dependency input rollups" in package_readme
    assert "per-command dependency input maps" in package_readme
    assert "reuse decisions can be separated from" in package_readme
    assert "next_action_summary" in package_readme
    assert "current next-action" in package_readme
    assert "set. `stage_recovery_summary`" in package_readme
    assert "input rollups" in package_readme
    assert "per-command input maps" in package_readme
    assert "required-input and optional-input action" in package_readme
    assert "readiness next-command routing fields" in package_readme
    assert "package_metadata_summary" in package_readme
    assert "package-area gates" in package_readme
    assert "Python metadata" in package_readme
    assert "per-action `command_name`, `command_category`" in package_readme
    assert "`blocked_by_stages`" in package_readme
    assert "`required_inputs`" in package_readme
    assert "`optional_inputs`" in package_readme
    assert "`stage_actions`" in package_readme
    assert "blocker summaries" in package_readme
    assert "command-manifest safety checks" in package_readme
    assert "recovery-plan required\nand optional input rollups" in package_readme
    assert "remaining-area command rollups" in package_readme
    assert "training-dependency\nrollups" in package_readme
    assert "`action_category`" in package_readme
    assert "real" in package_readme
    assert "timed-answer counts" in package_readme
    assert "remaining_recovery_plan" in package_readme
    assert "recovery_plan_summary" in package_readme
    assert "blocked-stage command" in package_readme
    assert "matrix" in package_readme
    assert "current and expected evidence" in package_readme
    assert "command inputs" in package_readme
    assert "next_action_title" in package_readme
    assert "next_action_category" in package_readme
    assert "next_action_command_name" in package_readme
    assert "next_action_launches_training" in package_readme
    assert "next_action_command_required_inputs" in package_readme
    assert "next_action_command_optional_inputs" in package_readme
    assert "Recovery Plan" in package_readme
    assert "generate_eval_report_after_stage" in package_readme
    assert "generate_sample_inspection_after_stage" in package_readme
    assert "blocked_by_stages" in package_readme
    root_readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    assert "Contract-status JSON and stdout also include package source" in root_readme
    assert "The saved `contract_status.json` is the durable automation surface" in (
        root_readme
    )
    assert "The JSON keeps full `next_actions` commands" in root_readme
    assert "package_metadata_summary" in root_readme
    assert "reuse decisions can be separated from" in root_readme
    assert "next_action_summary" in root_readme
    assert "current next-action set" in root_readme
    assert "actions with declared required and optional inputs" in root_readme
    assert "input counts and per-command input maps" in root_readme
    assert "next readiness command name" in root_readme
    assert "recovery_plan_summary" in root_readme
    assert "training_dependency_summary" in root_readme
    assert "required and optional input counts" in root_readme
    assert "each bucket" in root_readme
    assert "per-command input maps" in root_readme
    assert "blocked-stage command" in root_readme
    assert "matrix" in root_readme
    assert "current and\nexpected evidence" in root_readme
    assert "command inputs" in root_readme
    assert "next_action_title" in root_readme
    assert "next_action_category" in root_readme
    assert "next_action_command_name" in root_readme
    assert "next_action_launches_training" in root_readme
    assert "next_action_command_required_inputs" in root_readme
    assert "next_action_command_optional_inputs" in root_readme
    assert "package Python" in root_readme
    assert "metadata so" in root_readme
    assert "Next-action rows also expose" in root_readme
    assert "`command_name`, `command_category`, `blocked_by_stages`" in root_readme
    assert "`required_inputs`" in root_readme
    assert "`optional_inputs`" in root_readme
    assert "native and WSL readiness blocker summaries" in root_readme
    assert "recovery-plan required and optional input rollups" in root_readme
    assert "command category rollups" in root_readme
    assert "non-training command is still waiting on training evidence" in root_readme
    assert "`action_category`" in root_readme
    assert "generate_eval_report_after_stage" in root_readme
    assert "generate_sample_inspection_after_stage" in root_readme
    assert (
        package_manifest["files"]["full_training_eval_resume_inspector"]
        == "launch/inspect_full_training_eval_resume.sh"
    )
    assert "inspect_full_training_eval_resume" in launch_commands
    assert "full_training_eval_resume_inspection.json" in package_readme
    assert "full_training_eval_resume_inspection" in resume_inspector
    assert 'REUSE_STAGE_OUTPUTS="${REUSE_STAGE_OUTPUTS:-0}"' in resume_inspector
    assert 'SFT_RESUME_FROM_CHECKPOINT="${SFT_RESUME_FROM_CHECKPOINT:-}"' in resume_inspector
    assert 'DPO_RESUME_FROM_CHECKPOINT="${DPO_RESUME_FROM_CHECKPOINT:-}"' in resume_inspector
    assert 'PYTHON_BIN="${PYTHON_BIN:-python}"' in resume_inspector
    assert "command -v python3" in resume_inspector
    assert "train inspect-resume outputs" in resume_inspector
    assert "--sft-steps \"$SFT_MAX_STEPS\"" in resume_inspector
    assert "--dpo-steps \"$DPO_MAX_STEPS\"" in resume_inspector
    assert "--rl-steps \"$RL_MAX_STEPS\"" in resume_inspector
    assert "--out outputs/full_training_eval_resume_inspection.json" in resume_inspector
    assert "--markdown-out outputs/full_training_eval_resume_inspection.md" in resume_inspector
    assert "inspect_args+=(--reuse-stage-outputs)" in resume_inspector
    assert 'inspect_args+=(--dpo-resume-from-checkpoint "$DPO_RESUME_FROM_CHECKPOINT")' in resume_inspector
    assert 'PYTHONPATH=src "$PYTHON_BIN" -m semantic_mirror.cli "${inspect_args[@]}"' in resume_inspector
    packaged_audit_text = (bundle_out / "audit" / "current_environment.json").read_text(
        encoding="utf-8"
    )
    packaged_audit = json.loads(packaged_audit_text)
    assert packaged_audit["environment"]["secrets"]["hf_token_present"]
    assert "secret-hf-token" not in packaged_audit_text
    assert "secret-hf-token" not in (bundle_out / "ENVIRONMENT.md").read_text(encoding="utf-8")

    unsupported_python_audit = audit_training_environment(
        training_out,
        module_probe=all_modules_available,
        torch_probe=cuda_torch_probe,
        env_values={"HF_TOKEN": "present"},
        platform_name="Linux",
        python_version="3.14.0",
    )
    assert not unsupported_python_audit["passed"]
    assert any(
        check["name"] == "python_version_supported_for_unsloth" and not check["passed"]
        for check in unsupported_python_audit["checks"]
    )
    assert unsupported_python_audit["blocker"]["blocked"]
    assert "python_version_supported_for_unsloth" in unsupported_python_audit["blocker"][
        "failed_required_checks"
    ]
    assert any(
        "outside the supported >=3.11,<3.14 range" in item
        for item in unsupported_python_audit["blocker"]["summary"]
    )
    unsupported_evidence = unsupported_python_audit["blocker"]["evidence"]
    assert unsupported_evidence["python"] == {
        "actual_version": "3.14.0",
        "supported": False,
        "supported_range": ">=3.11,<3.14",
    }
    assert unsupported_evidence["audit_command"][:4] == [
        "uv",
        "run",
        "semantic-mirror",
        "train",
    ]

    failed_runtime_package = package_training_bundle(
        training_out,
        tmp_path / "training_bundle_failed_runtime",
        module_probe=lambda _module: False,
        torch_probe=lambda: {"importable": False, "cuda_available": False, "device_count": 0},
        platform_name="Windows",
    )
    assert failed_runtime_package["passed"]
    assert not failed_runtime_package["current_runtime_ready"]
    assert "required_training_modules" in failed_runtime_package["current_runtime_failed_checks"]

    failed_audit = audit_training_environment(
        training_out,
        module_probe=lambda module: module == "torch",
        torch_probe=lambda: {"importable": True, "cuda_available": False, "device_count": 0},
        platform_name="Windows",
    )
    assert not failed_audit["passed"]
    assert failed_audit["blocker"]["blocked"]
    assert "torch_cuda_available" in failed_audit["blocker"]["failed_required_checks"]
    assert failed_audit["blocker"]["recommended_fallback"]
    failed_evidence = failed_audit["blocker"]["evidence"]
    assert failed_evidence["environment"]["platform"] == "Windows"
    assert failed_evidence["modules"]["missing_required"] == [
        "unsloth",
        "trl",
        "datasets",
        "transformers",
        "bitsandbytes",
        "peft",
    ]
    assert failed_evidence["modules"]["imported_required_versions"] == {"torch": None}
    assert failed_evidence["torch"]["required_gpu"]
    assert failed_evidence["torch"]["importable"]
    assert not failed_evidence["torch"]["cuda_available"]
    assert failed_audit["repro"]["audit_command"][:4] == [
        "uv",
        "run",
        "semantic-mirror",
        "train",
    ]
    failed_launch = launch_training_job(
        training_out,
        stage="dpo",
        output_dir=tmp_path / "dpo-output",
        dry_run=True,
        audit_report=failed_audit,
        model_name_or_path="adapter-path",
    )
    assert not failed_launch["passed"]
    assert failed_launch["reason"] == "environment_not_ready"
    assert not failed_launch["would_launch"]

    invalid_launch = launch_training_job(
        tmp_path / "missing-training-dir",
        stage="sft",
        output_dir=tmp_path / "invalid-output",
        dry_run=True,
    )
    assert not invalid_launch["passed"]
    assert invalid_launch["reason"] == "environment_not_ready"
    assert invalid_launch["command"] is None

    teacher_requests_manifest = export_teacher_requests(
        out,
        tmp_path / "teacher_requests",
        candidates_per_unit=2,
        models=["teacher-a", "teacher-b"],
        max_units=1,
    )
    assert teacher_requests_manifest["mode"] == "teacher_export"
    assert teacher_requests_manifest["request_counts"]["candidate_generation"] == 2

    requests = _read_jsonl(tmp_path / "teacher_requests" / "candidate_requests.jsonl")
    assert "static_analysis" in requests[0]["messages"][1]["content"]
    request_payload = json.loads(requests[0]["messages"][1]["content"])
    field_rules = " ".join(request_payload["response_contract"]["field_name_rules"])
    assert "source_spans, never source_span" in field_rules
    assert "data_ml_details" in request_payload["response_contract"]["sir_unit_skeleton"]
    first_positive = next(record for record in silver if record["record_id"] == requests[0]["dataset_record_id"])
    faithful_candidate = copy.deepcopy(first_positive["target"]["sir_unit"])
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-openai-key",
                "ANTHROPIC_API_KEY=test-anthropic-key",
                "GEMINI_API_KEY=test-gemini-key",
                "SEMANTIC_MIRROR_TEACHER_PROVIDER=openai",
                "SEMANTIC_MIRROR_TEACHER_MODEL=test-openai-model",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_transport(
        _url: str,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> dict[str, object]:
        assert headers["Authorization"] == "Bearer test-openai-key"
        assert payload["model"] == "test-openai-model"
        response_payload = {
            "request_id": requests[0]["request_id"],
            "sir_unit": faithful_candidate,
            "rationale": "fake transport preserves static facts",
        }
        return {
            "id": "resp_test",
            "output_text": json.dumps(response_payload),
        }

    teacher_run_manifest = run_teacher_requests(
        tmp_path / "teacher_requests" / "candidate_requests.jsonl",
        tmp_path / "teacher_api_responses.jsonl",
        env_file=env_file,
        max_requests=1,
        transport=fake_transport,
    )
    assert teacher_run_manifest["mode"] == "teacher_run"
    assert teacher_run_manifest["counts"]["responses"] == 1
    api_responses = _read_jsonl(tmp_path / "teacher_api_responses.jsonl")
    assert api_responses[0]["provider"] == "openai"
    assert api_responses[0]["provider_response_id"] == "resp_test"

    def fake_anthropic_transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> dict[str, object]:
        assert url == "https://api.anthropic.com/v1/messages"
        assert headers["x-api-key"] == "test-anthropic-key"
        assert headers["anthropic-version"] == "2023-06-01"
        assert payload["model"] == "test-anthropic-model"
        assert payload["tool_choice"]["name"] == "emit_semantic_mirror_response"
        response_payload = {
            "request_id": requests[0]["request_id"],
            "sir_unit": faithful_candidate,
            "rationale": "fake anthropic transport preserves static facts",
        }
        return {
            "id": "msg_test",
            "content": [
                {
                    "type": "tool_use",
                    "name": "emit_semantic_mirror_response",
                    "input": response_payload,
                }
            ],
        }

    anthropic_manifest = run_teacher_requests(
        tmp_path / "teacher_requests" / "candidate_requests.jsonl",
        tmp_path / "teacher_anthropic_responses.jsonl",
        provider="anthropic",
        model="test-anthropic-model",
        env_file=env_file,
        max_requests=1,
        transport=fake_anthropic_transport,
    )
    assert anthropic_manifest["provider"] == "anthropic"
    anthropic_responses = _read_jsonl(tmp_path / "teacher_anthropic_responses.jsonl")
    assert anthropic_responses[0]["provider"] == "anthropic"
    assert anthropic_responses[0]["provider_response_id"] == "msg_test"

    def fake_gemini_transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> dict[str, object]:
        assert url.endswith("/models/test-gemini-model:generateContent")
        assert headers["x-goog-api-key"] == "test-gemini-key"
        assert payload["generationConfig"]["responseMimeType"] == "application/json"
        response_payload = {
            "request_id": requests[0]["request_id"],
            "sir_unit": faithful_candidate,
            "rationale": "fake gemini transport preserves static facts",
        }
        return {
            "responseId": "gemini_test",
            "candidates": [
                {"content": {"parts": [{"text": json.dumps(response_payload)}]}},
            ],
        }

    gemini_manifest = run_teacher_requests(
        tmp_path / "teacher_requests" / "candidate_requests.jsonl",
        tmp_path / "teacher_gemini_responses.jsonl",
        provider="gemini",
        model="test-gemini-model",
        env_file=env_file,
        max_requests=1,
        transport=fake_gemini_transport,
    )
    assert gemini_manifest["provider"] == "gemini"
    gemini_responses = _read_jsonl(tmp_path / "teacher_gemini_responses.jsonl")
    assert gemini_responses[0]["provider"] == "gemini"
    assert gemini_responses[0]["provider_response_id"] == "gemini_test"

    corrupted_candidate = copy.deepcopy(first_positive["target"]["sir_unit"])
    if corrupted_candidate["calls"]:
        corrupted_candidate["calls"] = corrupted_candidate["calls"][1:]
    else:
        corrupted_candidate["calls"].append(
            {
                "claim": "Calls `invented.teacher_behavior`.",
                "confidence": 0.1,
                "name": "invented.teacher_behavior",
                "source_spans": corrupted_candidate["source_spans"],
            }
        )

    def fake_pipeline_transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> dict[str, object]:
        if url == "https://api.openai.com/v1/responses":
            assert headers["Authorization"] == "Bearer test-openai-key"
            assert payload["model"] == "pipeline-openai-model"
            return {
                "id": "pipeline_openai",
                "output_text": json.dumps(
                    {
                        "request_id": "provider record will replace this",
                        "sir_unit": faithful_candidate,
                        "rationale": "pipeline openai candidate preserves static facts",
                    }
                ),
            }
        if url == "https://api.anthropic.com/v1/messages":
            assert headers["x-api-key"] == "test-anthropic-key"
            assert payload["model"] == "pipeline-anthropic-model"
            assert payload["tool_choice"]["name"] == "emit_semantic_mirror_response"
            return {
                "id": "pipeline_anthropic",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "emit_semantic_mirror_response",
                        "input": {
                            "request_id": "provider record will replace this",
                            "sir_unit": corrupted_candidate,
                            "rationale": "pipeline anthropic candidate drops a fact",
                        },
                    }
                ],
            }
        raise AssertionError(f"unexpected provider URL: {url}")

    pipeline_manifest = run_teacher_pipeline(
        out,
        tmp_path / "teacher_pipeline",
        providers=["openai", "anthropic"],
        models=["pipeline-openai-model", "pipeline-anthropic-model"],
        candidates_per_provider=1,
        max_units=1,
        env_file=env_file,
        transport=fake_pipeline_transport,
    )
    assert pipeline_manifest["mode"] == "teacher_pipeline"
    assert pipeline_manifest["counts"]["requests"] == 2
    assert pipeline_manifest["counts"]["responses"] == 2
    assert pipeline_manifest["counts"]["errors"] == 0
    assert pipeline_manifest["counts"]["accepted_candidates"] == 1
    assert pipeline_manifest["counts"]["auto_rejected_candidates"] == 1
    assert pipeline_manifest["counts"]["preference_pairs"] == 1
    pipeline_requests = _read_jsonl(tmp_path / "teacher_pipeline" / "candidate_requests.jsonl")
    assert {request["provider"] for request in pipeline_requests} == {"openai", "anthropic"}
    assert all(request["request_id"].startswith(request["provider"]) for request in pipeline_requests)

    teacher_training_manifest = prepare_training_data(
        out,
        tmp_path / "training_with_teacher",
        max_records=2,
        teacher_results_path=tmp_path / "teacher_pipeline",
    )
    assert teacher_training_manifest["input_counts"]["teacher_preference_pairs"] == 1
    assert teacher_training_manifest["output_counts"]["preference_pairs"] > 1
    teacher_training_preferences = _read_jsonl(
        tmp_path / "training_with_teacher" / "preference_pairs.jsonl"
    )
    assert any(
        preference["metadata"].get("source") == "teacher_results"
        for preference in teacher_training_preferences
    )
    assert validate_training_batch(tmp_path / "training_with_teacher")["passed"]

    responses_path = tmp_path / "teacher_responses.jsonl"
    _write_jsonl(
        responses_path,
        [
            {
                "request_id": requests[0]["request_id"],
                "model": "teacher-a",
                "sir_unit": faithful_candidate,
                "rationale": "preserves static facts",
            },
            {
                "request_id": requests[1]["request_id"],
                "model": "teacher-b",
                "sir_unit": corrupted_candidate,
                "rationale": "drops a fact for test coverage",
            },
        ],
    )
    teacher_ingest_manifest = ingest_teacher_responses(
        out,
        tmp_path / "teacher_requests" / "candidate_requests.jsonl",
        responses_path,
        tmp_path / "teacher_results",
    )
    assert teacher_ingest_manifest["mode"] == "teacher_ingest"
    assert teacher_ingest_manifest["counts"]["accepted_candidates"] == 1
    assert teacher_ingest_manifest["counts"]["auto_rejected_candidates"] == 1
    assert teacher_ingest_manifest["counts"]["teacher_candidates"] == 2
    assert teacher_ingest_manifest["counts"]["critic_requests"] == 1
    assert teacher_ingest_manifest["counts"]["preference_pairs"] == 1

    candidate_results = _read_jsonl(tmp_path / "teacher_results" / "candidate_results.jsonl")
    teacher_candidates = _read_jsonl(tmp_path / "teacher_results" / "teacher_candidates.jsonl")
    critic_requests = _read_jsonl(tmp_path / "teacher_results" / "critic_requests.jsonl")
    teacher_preferences = _read_jsonl(
        tmp_path / "teacher_results" / "teacher_preference_pairs.jsonl"
    )
    assert any(result["accepted"] for result in candidate_results)
    assert any(result["auto_reject"] for result in candidate_results)
    assert len(teacher_candidates) == 2
    assert all("dataset_record_id" in candidate for candidate in teacher_candidates)
    assert all("sir_unit" in candidate for candidate in teacher_candidates)
    teacher_baseline_eval = evaluate_model_candidates(
        out,
        tmp_path / "teacher_results" / "teacher_candidates.jsonl",
        model_name="teacher-baseline-from-ingest",
    )
    assert teacher_baseline_eval["metrics"]["candidate_records"] == 2
    assert teacher_baseline_eval["metrics"]["covered_units"] == 1
    assert teacher_baseline_eval["metrics"]["expected_units"] == 5
    assert teacher_baseline_eval["metrics"]["heldout_unit_coverage"] == 0.2
    assert not teacher_baseline_eval["passed"]
    malformed_candidates_path = tmp_path / "malformed_teacher_candidates.jsonl"
    _write_jsonl(
        malformed_candidates_path,
        [
            {
                "dataset_record_id": first_positive["record_id"],
                "unit_id": first_positive["unit_id"],
                "sir_unit": {
                    "name": first_positive["qualified_name"],
                    "source_span": first_positive["source_spans"][0],
                },
            }
        ],
    )
    malformed_eval = evaluate_model_candidates(
        out,
        malformed_candidates_path,
        model_name="malformed-teacher-candidate",
    )
    assert malformed_eval["metrics"]["candidate_records"] == 1
    assert malformed_eval["penalties"]["schema_errors"] == 1
    assert "missing_sir_unit" not in malformed_eval["penalties"]
    assert "verifier_report" in critic_requests[0]["messages"][1]["content"]
    assert teacher_preferences[0]["chosen"] != teacher_preferences[0]["rejected"]

    def fake_critic_transport(
        _url: str,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> dict[str, object]:
        assert headers["Authorization"] == "Bearer test-openai-key"
        assert payload["text"]["format"]["name"] == "semantic_mirror_critic_response"
        response_payload = {
            "request_id": critic_requests[0]["request_id"],
            "candidate_result_id": "model-invented-candidate-id",
            "error_labels": [
                {
                    "label": "missing_calls",
                    "severity": "high",
                    "evidence": "Verifier penalized dropped static calls.",
                }
            ],
            "critique": "Candidate drops source-backed call facts and should be repaired first.",
            "repair_priority": "high",
        }
        return {
            "id": "critic_resp_test",
            "output_text": json.dumps(response_payload),
        }

    critic_run_manifest = run_critic_requests(
        tmp_path / "teacher_results" / "critic_requests.jsonl",
        tmp_path / "critic_responses.jsonl",
        env_file=env_file,
        max_requests=1,
        transport=fake_critic_transport,
    )
    assert critic_run_manifest["mode"] == "critic_run"
    assert critic_run_manifest["counts"]["responses"] == 1
    critic_api_responses = _read_jsonl(tmp_path / "critic_responses.jsonl")
    assert critic_api_responses[0]["provider_response_id"] == "critic_resp_test"
    assert critic_api_responses[0]["candidate_result_id"] == critic_requests[0]["candidate_result_id"]

    critic_ingest_manifest = ingest_critic_responses(
        tmp_path / "teacher_results",
        tmp_path / "critic_responses.jsonl",
        tmp_path / "teacher_results",
    )
    assert critic_ingest_manifest["mode"] == "critic_ingest"
    assert critic_ingest_manifest["counts"]["labeled_candidates"] == 1
    assert critic_ingest_manifest["counts"]["error_labels"] == 1
    critic_labels = _read_jsonl(tmp_path / "teacher_results" / "critic_labels.jsonl")
    assert critic_labels[0]["error_labels"][0]["label"] == "missing_calls"
    critic_review_queue = _read_jsonl(tmp_path / "teacher_results" / "critic_review_queue.jsonl")
    assert critic_review_queue[0]["critic_repair_priority"] == "high"

    critic_training_manifest = prepare_training_data(
        out,
        tmp_path / "training_with_critic_labels",
        max_records=2,
        teacher_results_path=tmp_path / "teacher_results",
    )
    assert critic_training_manifest["input_counts"]["teacher_preference_pairs"] == 1
    critic_training_preferences = _read_jsonl(
        tmp_path / "training_with_critic_labels" / "preference_pairs.jsonl"
    )
    teacher_preference = next(
        preference
        for preference in critic_training_preferences
        if preference["metadata"].get("source") == "teacher_results"
    )
    assert teacher_preference["metadata"]["critic_labels"][0]["label"] == "missing_calls"

    baseline_candidates_path = tmp_path / "teacher_baseline_candidates.jsonl"
    current_candidates_path = tmp_path / "sft_candidates.jsonl"
    hallucinating_candidates_path = tmp_path / "rl_regressed_candidates.jsonl"
    _write_jsonl(
        baseline_candidates_path,
        [
            {
                "dataset_record_id": record["record_id"],
                "sir_unit": _missing_fact_candidate(record["target"]["sir_unit"]),
            }
            for record in silver
        ],
    )
    _write_jsonl(
        current_candidates_path,
        [
            {
                "dataset_record_id": record["record_id"],
                "sir_unit": record["target"]["sir_unit"],
            }
            for record in silver
        ],
    )
    hallucinating = [
        {
            "dataset_record_id": record["record_id"],
            "sir_unit": copy.deepcopy(record["target"]["sir_unit"]),
        }
        for record in silver
    ]
    hallucinating[0]["sir_unit"]["calls"].append(
        {
            "claim": "Calls `invented.model_behavior`.",
            "confidence": 0.1,
            "name": "invented.model_behavior",
            "source_spans": hallucinating[0]["sir_unit"]["source_spans"],
        }
    )
    _write_jsonl(hallucinating_candidates_path, hallucinating)

    baseline_eval = evaluate_model_candidates(
        out,
        baseline_candidates_path,
        model_name="teacher-baseline",
        out_path=tmp_path / "teacher_baseline_eval.json",
    )
    current_eval = evaluate_model_candidates(
        out,
        current_candidates_path,
        model_name="sft-candidate",
        out_path=tmp_path / "sft_eval.json",
    )
    hallucinating_eval = evaluate_model_candidates(
        out,
        hallucinating_candidates_path,
        model_name="rl-regressed",
        out_path=tmp_path / "rl_regressed_eval.json",
    )
    assert baseline_eval["passed"]
    assert current_eval["passed"]
    assert hallucinating_eval["passed"]
    assert current_eval["metrics"]["average_static_faithfulness_score"] > baseline_eval["metrics"][
        "average_static_faithfulness_score"
    ]
    assert hallucinating_eval["metrics"]["hallucination_penalties"] > 0

    sft_comparison = compare_model_evaluations(
        tmp_path / "teacher_baseline_eval.json",
        tmp_path / "sft_eval.json",
        stage="sft",
    )
    rl_comparison = compare_model_evaluations(
        tmp_path / "sft_eval.json",
        tmp_path / "rl_regressed_eval.json",
        stage="rl",
    )
    dpo_comparison = compare_model_evaluations(
        tmp_path / "sft_eval.json",
        tmp_path / "sft_eval.json",
        stage="dpo",
    )
    assert sft_comparison["passed"]
    assert dpo_comparison["passed"]
    assert not rl_comparison["passed"]
    rl_gates = {gate["name"]: gate for gate in rl_comparison["gates"]}
    assert not rl_gates["hallucination_penalties_not_increased"]["passed"]


def test_corpus_collect_aggregates_multiple_repositories_for_training(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    (repo_a / "src").mkdir(parents=True)
    (repo_b / "src").mkdir(parents=True)
    (repo_a / "src" / "train.py").write_text(SAMPLE_TRAIN, encoding="utf-8")
    (repo_b / "src" / "metrics.py").write_text(
        """\
def accuracy(logits, labels):
    predictions = logits.argmax(dim=-1)
    correct = (predictions == labels).float().mean()
    return correct.item()
""",
        encoding="utf-8",
    )

    corpus_manifest = collect_corpus(
        [f"first={repo_a}", f"second={repo_b}"],
        tmp_path / "corpus",
        profile="data_ml",
        zoom="L4",
        max_units_per_repo=3,
        review_budget_per_repo=2,
        hard_negatives_per_unit=1,
    )

    assert corpus_manifest["mode"] == "corpus_collect"
    assert corpus_manifest["repo_count"] == 2
    assert corpus_manifest["successful_repos"] == 2
    aggregate = tmp_path / "corpus" / "aggregate"
    aggregate_manifest = json.loads((aggregate / "manifest.json").read_text(encoding="utf-8"))
    assert aggregate_manifest["mode"] == "corpus_aggregate_dataset"
    assert aggregate_manifest["counts"]["silver_records"] > 0
    aggregate_silver = _read_jsonl(aggregate / "silver.jsonl")
    aggregate_hard_negatives = _read_jsonl(aggregate / "hard_negative.jsonl")
    assert {record["source_repo_id"] for record in aggregate_silver} == {"000-first", "001-second"}
    assert all(":" in record["unit_id"] for record in aggregate_silver)
    assert all(":" in record["positive_unit_id"] for record in aggregate_hard_negatives)

    aggregate_eval = evaluate_dataset(aggregate)
    assert aggregate_eval["passed"]

    training_manifest = prepare_training_data(
        aggregate,
        tmp_path / "corpus_training",
        max_records=4,
    )
    assert training_manifest["input_counts"]["silver"] == len(aggregate_silver)
    assert training_manifest["output_counts"]["sft_records"] > 0
    assert validate_training_batch(tmp_path / "corpus_training")["passed"]
    rl_prompts = _read_jsonl(tmp_path / "corpus_training" / "rl_prompts.jsonl")
    assert rl_prompts[0]["metadata"]["source_repo_path"]

    aggregate_candidates_path = tmp_path / "aggregate_candidates.jsonl"
    _write_jsonl(
        aggregate_candidates_path,
        [
            {
                "dataset_record_id": record["record_id"],
                "unit_id": record["unit_id"],
                "source_repo_path": record["source_repo_path"],
                "sir_unit": record["target"]["sir_unit"],
            }
            for record in aggregate_silver
        ],
    )
    aggregate_model_eval = evaluate_model_candidates(
        aggregate,
        aggregate_candidates_path,
        model_name="aggregate-faithful",
    )
    assert aggregate_model_eval["passed"]
    assert aggregate_model_eval["repo"] is None
    assert set(aggregate_model_eval["source_repos"]) == {"000-first", "001-second"}
    assert aggregate_model_eval["metrics"]["heldout_unit_coverage"] == 1.0
    assert "invented_units" not in aggregate_model_eval["penalties"]
    assert {
        Path(result["source_repo_path"]).name
        for result in aggregate_model_eval["results"]
        if result["reference_found"]
    } == {"repo_a", "repo_b"}

    stale_aggregate = tmp_path / "corpus" / "stale_aggregate"
    shutil.copytree(aggregate, stale_aggregate)
    stale_manifest = json.loads((stale_aggregate / "manifest.json").read_text(encoding="utf-8"))
    for source_repo in stale_manifest["source_repos"]:
        source_repo["path"] = str(tmp_path / "missing-host-path" / source_repo["repo_id"])
    (stale_aggregate / "manifest.json").write_text(
        json.dumps(stale_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stale_silver = _read_jsonl(stale_aggregate / "silver.jsonl")
    for record in stale_silver:
        record["source_repo_path"] = str(tmp_path / "missing-host-path" / record["source_repo_id"])
    _write_jsonl(stale_aggregate / "silver.jsonl", stale_silver)
    stale_candidates_path = tmp_path / "stale_aggregate_candidates.jsonl"
    _write_jsonl(
        stale_candidates_path,
        [
            {
                "dataset_record_id": record["record_id"],
                "unit_id": record["unit_id"],
                "source_repo_path": record["source_repo_path"],
                "sir_unit": record["target"]["sir_unit"],
            }
            for record in stale_silver
        ],
    )
    stale_model_eval = evaluate_model_candidates(
        stale_aggregate,
        stale_candidates_path,
        model_name="aggregate-faithful-stale-paths",
    )
    assert stale_model_eval["passed"]
    assert {
        Path(result["source_repo_path"]).name
        for result in stale_model_eval["results"]
        if result["reference_found"]
    } == {"000-first", "001-second"}

    teacher_request_path = tmp_path / "aggregate_teacher_requests.jsonl"
    teacher_response_path = tmp_path / "aggregate_teacher_responses.jsonl"
    aggregate_record = aggregate_silver[0]
    _write_jsonl(
        teacher_request_path,
        [
            {
                "request_id": "aggregate-teacher-0",
                "dataset_record_id": aggregate_record["record_id"],
                "model": "teacher-test",
            }
        ],
    )
    _write_jsonl(
        teacher_response_path,
        [
            {
                "request_id": "aggregate-teacher-0",
                "model": "teacher-test",
                "sir_unit": aggregate_record["target"]["sir_unit"],
            }
        ],
    )
    aggregate_teacher_manifest = ingest_teacher_responses(
        aggregate,
        teacher_request_path,
        teacher_response_path,
        tmp_path / "aggregate_teacher_results",
    )
    assert aggregate_teacher_manifest["counts"]["accepted_candidates"] == 1
    aggregate_teacher_results = _read_jsonl(
        tmp_path / "aggregate_teacher_results" / "candidate_results.jsonl"
    )
    assert Path(aggregate_teacher_results[0]["source_repo_path"]).name in {"repo_a", "repo_b"}


def test_regression_compare_enforces_score_drop_and_penalty_gates(tmp_path: Path) -> None:
    baseline = {
        "mode": "mirror_evaluation",
        "passed": True,
        "score_report": {"score": 100, "penalties": {}},
    }
    current = {
        "mode": "mirror_evaluation",
        "passed": True,
        "score_report": {"score": 99, "penalties": {}},
    }
    regressed = {
        "mode": "mirror_evaluation",
        "passed": True,
        "score_report": {"score": 80, "penalties": {"missing_calls": 1}},
    }
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    regressed_path = tmp_path / "regressed.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    regressed_path.write_text(json.dumps(regressed), encoding="utf-8")

    passing = compare_regression_reports(baseline_path, current_path, max_score_drop=0.01)
    failing = compare_regression_reports(baseline_path, regressed_path, max_score_drop=0.01)

    assert passing["passed"]
    assert not failing["passed"]
    failing_gates = {gate["name"]: gate for gate in failing["gates"]}
    assert not failing_gates["score_drop"]["passed"]
    assert not failing_gates["verifier_penalty_regression"]["passed"]


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    return result.stdout


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _completed_study_answers(study_dir: Path) -> list[dict[str, object]]:
    answers: list[dict[str, object]] = []
    for template in _read_jsonl(study_dir / "answers_template.jsonl"):
        record = dict(template)
        record["reviewer"] = "unit-test"
        record["started_at"] = "2026-06-08T00:00:00+00:00"
        record["completed_at"] = "2026-06-08T00:00:10+00:00"
        record["answer"] = "Reviewer found the expected source-backed behavior."
        record["correct"] = True
        record["acknowledged"] = record["task_type"] == "visibility_marker"
        if record["task_type"] == "visibility_marker":
            record["elapsed_seconds"] = 3.0
        elif record["condition"] == "mirror":
            record["elapsed_seconds"] = 8.0
        else:
            record["elapsed_seconds"] = 20.0
        answers.append(record)
    return answers


def _missing_fact_candidate(unit: dict[str, object]) -> dict[str, object]:
    corrupted = copy.deepcopy(unit)
    for key in ("calls", "returns", "writes", "control_flow", "state_mutations"):
        if corrupted[key]:
            corrupted[key] = corrupted[key][1:]
            return corrupted
    return corrupted
