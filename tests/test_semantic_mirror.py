from __future__ import annotations

import copy
import json
import subprocess
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
    REQUIRED_TRAINING_MODULES,
    audit_training_environment,
    launch_training_job,
    package_training_bundle,
    prepare_training_data,
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
    answers_path = tmp_path / "review_study_answers.jsonl"
    _write_jsonl(answers_path, _completed_study_answers(tmp_path / "review_study"))
    study_eval = evaluate_human_usefulness_study(
        tmp_path / "review_study",
        answers_path,
        min_accuracy=1.0,
    )
    assert study_eval["passed"]
    study_gates = {gate["name"]: gate for gate in study_eval["gates"]}
    assert study_gates["mirror_faster_than_source"]["actual"] > 1.0
    assert study_gates["visibility_items_acknowledged"]["actual"] == 1.0


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
    diff_answers_path = tmp_path / "diff_review_study_answers.jsonl"
    _write_jsonl(diff_answers_path, _completed_study_answers(tmp_path / "diff_review_study"))
    diff_study_eval = evaluate_human_usefulness_study(
        tmp_path / "diff_review_study",
        diff_answers_path,
        min_accuracy=1.0,
    )
    assert diff_study_eval["passed"]
    diff_study_gates = {gate["name"]: gate for gate in diff_study_eval["gates"]}
    assert diff_study_gates["changed_behavior_accuracy"]["actual"] == 1.0


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
    assert "text_excerpt" in sft[0]["messages"][1]["content"]
    assert "source_spans" in sft[0]["messages"][2]["content"]
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
    assert sft_config["base_model"] == "test/base-7b"
    assert sft_config["method"] == "QLoRA"
    assert reward_config["objective"] == "faithfulness_first_compactness_second"
    assert training_manifest["files"]["sft_script"] == "run_unsloth_sft.py"
    assert training_manifest["files"]["preference_script"] == "run_preference_dpo.py"
    assert training_manifest["files"]["rl_script"] == "run_reward_rl.py"
    assert training_manifest["files"]["candidate_generation_script"] == "generate_sir_candidates.py"
    assert training_manifest["files"]["reward_script"] == "score_sir_candidates.py"
    validation_report = validate_training_batch(training_out)
    assert validation_report["passed"]
    assert validation_report["issues"] == []

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

    dry_run = launch_training_job(
        training_out,
        stage="sft",
        output_dir=tmp_path / "sft-output",
        dry_run=True,
        audit_report=audit_report,
        python_executable="python-test",
    )
    assert dry_run["passed"]
    assert dry_run["would_launch"]
    assert not dry_run["launched"]
    assert dry_run["command"][0] == "python-test"
    assert "run_unsloth_sft.py" in dry_run["command"][1]

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
    assert validate_training_batch(bundle_out / "training")["passed"]
    assert (bundle_out / "training" / "run_unsloth_sft.py").exists()
    sft_script = (bundle_out / "training" / "run_unsloth_sft.py").read_text(encoding="utf-8")
    assert "SFTConfig" in sft_script
    assert "processing_class=tokenizer" in sft_script
    assert "tokenizer=tokenizer" not in sft_script
    assert 'parser.add_argument("--max-steps", type=int)' in sft_script
    assert "max_steps=args.max_steps or -1" in sft_script
    assert 'tokenizer.truncation_side = "left"' in sft_script
    assert 'os.environ.setdefault("UNSLOTH_ENABLE_CCE", "0")' in sft_script
    assert 'os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")' in sft_script
    assert 'os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")' in sft_script
    dpo_script = (bundle_out / "training" / "run_preference_dpo.py").read_text(encoding="utf-8")
    assert "TRANSFORMERS_CACHE" in dpo_script
    assert "DPOTrainer" in dpo_script
    assert "model.warnings_issued" in dpo_script
    assert 'parser.add_argument("--max-steps", type=int)' in dpo_script
    assert "max_steps=args.max_steps or -1" in dpo_script
    rl_script = (bundle_out / "training" / "run_reward_rl.py").read_text(encoding="utf-8")
    assert "_preferences_by_prompt(root, reward_config" in rl_script
    assert "_preferences_by_prompt(root / reward_config" not in rl_script
    assert "output_ids = output_ids.detach().clone().to(device)" in rl_script
    assert 'tokenizer.truncation_side = "left"' in rl_script
    assert "formatted_prompt = _format_generation_prompt" in rl_script
    assert "min_new_tokens=8" in rl_script
    assert "raw_sir_unit = _extract_json_object(text)" in rl_script
    assert "sir_unit = _repair_sir_unit(" in rl_script
    assert "sir_unit = _apply_faithfulness_repair(" in rl_script
    assert 'elif unit.get("raw_error"):' in rl_script
    assert "reward += _raw_generation_bonus(" in rl_script
    assert '"raw_parseable"' in rl_script
    candidate_script = (bundle_out / "training" / "generate_sir_candidates.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument("--max-new-tokens", type=int, default=1536)' in candidate_script
    assert 'parser.add_argument("--max-prompts", type=int)' in candidate_script
    assert 'parser.add_argument("--no-faithfulness-repair", action="store_true")' in candidate_script
    assert "prompts = prompts[: args.max_prompts]" in candidate_script
    assert 'tokenizer.truncation_side = "left"' in candidate_script
    assert "formatted_prompt = _format_generation_prompt" in candidate_script
    assert "sir_unit = _repair_sir_unit(" in candidate_script
    assert "sir_unit = _apply_faithfulness_repair(" in candidate_script
    assert 'elif unit.get("raw_error"):' in candidate_script
    assert "DATA_ML_DETAIL_CATEGORIES" in candidate_script
    assert "Do not continue the input JSON" in candidate_script
    assert "generation_tokens = min(" in candidate_script
    assert "max_prompt_tokens = max(128" in candidate_script
    assert "truncation=True" in candidate_script
    assert "max_length=max_prompt_tokens" in candidate_script
    assert 'prompt_len = inputs["input_ids"].shape[1]' in candidate_script
    assert "completion_ids = output_ids[:, prompt_len:]" in candidate_script
    assert "tokenizer.decode(completion_ids[0]" in candidate_script
    assert (bundle_out / "src" / "semantic_mirror" / "training.py").exists()
    assert (bundle_out / "pyproject.toml").exists()
    requirements = (bundle_out / "requirements-training.txt").read_text(encoding="utf-8")
    assert "-e ." in requirements
    assert "unsloth" in requirements
    assert "mergekit" in requirements
    assert "llm-blender" in requirements
    assert "weave" in requirements
    commands = json.loads((bundle_out / "launch" / "commands.json").read_text(encoding="utf-8"))
    assert "bootstrap_linux_cuda.sh" in commands["bootstrap_linux_cuda"]
    assert "run_full_training_eval.sh" in commands["full_training_eval"]
    assert "run_unsloth_sft.py" in commands["sft"]
    assert "run_reward_rl.py" in commands["rl"]
    assert (bundle_out / "setup" / "bootstrap_linux_cuda.sh").exists()
    assert (bundle_out / "setup" / "bootstrap_wsl_ubuntu.ps1").exists()
    bootstrap = (bundle_out / "setup" / "bootstrap_linux_cuda.sh").read_text(encoding="utf-8")
    assert "3.11" in bootstrap
    assert "3.14" in bootstrap
    assert (bundle_out / "launch" / "run_sft.sh").exists()
    assert (bundle_out / "launch" / "run_rl.sh").exists()
    assert (bundle_out / "launch" / "run_full_training_eval.sh").exists()
    for shell_script in [
        bundle_out / "setup" / "bootstrap_linux_cuda.sh",
        bundle_out / "launch" / "run_sft.sh",
        bundle_out / "launch" / "run_dpo.sh",
        bundle_out / "launch" / "run_rl.sh",
        bundle_out / "launch" / "generate_candidates.sh",
        bundle_out / "launch" / "score_candidates.sh",
        bundle_out / "launch" / "run_full_training_eval.sh",
    ]:
        assert b"\r\n" not in shell_script.read_bytes()
    full_eval_script = (bundle_out / "launch" / "run_full_training_eval.sh").read_text(
        encoding="utf-8"
    )
    assert "eval candidates" in full_eval_script
    assert "eval model-compare" in full_eval_script
    assert "training_eval_summary.json" in full_eval_script
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
    assert sft_comparison["passed"]
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
