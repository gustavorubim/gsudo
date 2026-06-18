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
    launch_training_job,
    package_training_bundle,
    prepare_training_data,
    summarize_full_eval_contract_status,
    validate_training_batch,
)


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
        encoding="utf-8",
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
    assert not status["repo_hygiene_status"]["checked"]
    assert not status["windows_readiness_status"]["checked"]
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
        for action in status["next_actions"]
    )
    assert any(
        action["title"] == "Regenerate target diagnostics"
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
    assert "## Resume Inspection" in status_markdown
    assert "| `dpo` | `resume` | 120 | 10 |" in status_markdown
    assert "## Next Actions" in status_markdown
    assert "Resume full eval through DPO and RL" in status_markdown
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
                    "failed_required_checks": ["torch_cuda_available"],
                    "recommended_fallback": "Use WSL CUDA.",
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
    assert readiness_status["windows_readiness_status"]["wsl_smoke_complete"]
    readiness_scorecard = {
        row["area"]: row for row in readiness_status["contract_scorecard"]
    }
    assert readiness_scorecard["windows_unsloth_readiness"]["passed"] is True
    assert readiness_scorecard["windows_unsloth_readiness"]["earned_reward"] == 65
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
            "--human-study-suite",
            str(human_suite),
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
    assert json.loads(cli_status.stdout)["passed"] is False
    cli_status_json = json.loads(
        (tmp_path / "contract_status_cli.json").read_text(encoding="utf-8")
    )
    refresh_action = next(
        action
        for action in cli_status_json["next_actions"]
        if action["title"] == "Regenerate contract status"
    )
    assert "--repo-root 'repo'" in refresh_action["command"]
    assert "--windows-audit 'windows_audit.json'" in refresh_action["command"]
    assert "--wsl-smoke-manifest 'smoke_chain_manifest.json'" in refresh_action["command"]
    assert "--human-study-suite 'phase6_summary.json'" in refresh_action["command"]
    assert "training_eval_summary_matches_requested_steps" in (
        tmp_path / "contract_status_cli.md"
    ).read_text(encoding="utf-8")

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
    assert validate_training_batch(bundle_out / "training")["passed"]
    assert (bundle_out / "training" / "run_unsloth_sft.py").exists()
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
    commands = json.loads((bundle_out / "launch" / "commands.json").read_text(encoding="utf-8"))
    assert "bootstrap_linux_cuda.sh" in commands["bootstrap_linux_cuda"]
    assert "run_full_training_eval.sh" in commands["full_training_eval"]
    assert "run_smoke_chain.sh" in commands["smoke_chain"]
    assert "run_wsl_smoke_chain.ps1" in commands["wsl_smoke_chain"]
    assert "-HeldOutDataset <windows_dataset_dir>" in commands["wsl_smoke_chain"]
    assert "--out outputs/validation_report.json" in commands["validate"]
    assert "--out outputs/audit.json" in commands["audit"]
    assert "train inspect-samples" in commands["inspect_samples"]
    assert "train report outputs" in commands["report"]
    assert "train contract-status outputs" in commands["contract_status"]
    assert "--sft-steps $SFT_MAX_STEPS" in commands["contract_status"]
    assert "--markdown-out outputs/contract_status.md" in commands["contract_status"]
    assert "run_unsloth_sft.py" in commands["sft"]
    assert "run_reward_rl.py" in commands["rl"]
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
    assert (bundle_out / "launch" / "run_sft.sh").exists()
    assert (bundle_out / "launch" / "run_rl.sh").exists()
    assert (bundle_out / "launch" / "run_smoke_chain.sh").exists()
    assert (bundle_out / "launch" / "run_wsl_smoke_chain.ps1").exists()
    assert (bundle_out / "launch" / "run_full_training_eval.sh").exists()
    for shell_script in [
        bundle_out / "setup" / "bootstrap_linux_cuda.sh",
        bundle_out / "launch" / "run_sft.sh",
        bundle_out / "launch" / "run_dpo.sh",
        bundle_out / "launch" / "run_rl.sh",
        bundle_out / "launch" / "generate_candidates.sh",
        bundle_out / "launch" / "score_candidates.sh",
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
    package_readme = (bundle_out / "README.md").read_text(encoding="utf-8")
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
    resume_inspector = (
        bundle_out / "launch" / "inspect_full_training_eval_resume.sh"
    ).read_text(encoding="utf-8")
    launch_commands = json.loads(
        (bundle_out / "launch" / "commands.json").read_text(encoding="utf-8")
    )
    package_readme = (bundle_out / "README.md").read_text(encoding="utf-8")
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
    assert "--out outputs/contract_status.json" in full_eval_script
    assert "--markdown-out outputs/contract_status.md" in full_eval_script
    assert "contract_status.json" in package_readme
    assert "contract_status.md" in package_readme
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
    assert '"$PYTHON_BIN" - "$SFT_MAX_STEPS"' in resume_inspector
    assert 'json.dumps(summary, indent=2, sort_keys=True) + "\\n"' in resume_inspector
    assert '"sft": Path("outputs/semantic-mirror-sft")' in resume_inspector
    assert 'manifest_max_steps == requested_steps[stage]' in resume_inspector
    assert 'action = "reuse"' in resume_inspector
    assert 'action = "resume"' in resume_inspector
    assert 'action = "rerun"' in resume_inspector
    assert 'action = "run"' in resume_inspector
    assert '"resume_supported": stage in {"sft", "dpo"}' in resume_inspector
    assert "stage action requested manifest checkpoint reason" in resume_inspector
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
