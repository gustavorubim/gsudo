"""Reviewer-facing packs for Semantic Mirror human usefulness gates."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

REVIEW_VERSION = "0.1.0"
QUESTION_KINDS = (
    "data_ml_behavior",
    "side_effects",
    "failure_modes",
    "return_behavior",
    "dependency_calls",
    "uncertainty_and_hazards",
)
PHASE6_HUMAN_STUDY_REQUIREMENTS = {
    "required_task_sets": ["whole_repo", "diff_mode"],
    "required_answer_source": "real_timed_reviewer_logs",
    "evaluation_command": (
        "uv run semantic-mirror eval human-study <study_dir> "
        "--answers <answers.jsonl> --out <report.json>"
    ),
    "pass_gates": [
        "real_timed_reviewer_logs",
        "reviewer_identity_present",
        "answer_text_present",
        "paired_answer_coverage_complete",
        "mirror_accuracy_at_or_above_threshold",
        "mirror_accuracy_not_lower_than_source",
        "mirror_median_faster_than_source",
        "changed_behavior_accuracy_at_or_above_threshold",
        "visibility_items_acknowledged",
    ],
}


def create_review_pack(
    mirror_path: Path | str,
    out_path: Path | str,
    *,
    max_questions: int = 25,
    max_change_tasks: int = 25,
) -> dict[str, Any]:
    """Create evidence-backed questions and diff review tasks from a mirror."""

    mirror = Path(mirror_path).resolve()
    out = Path(out_path).resolve()
    manifest = _read_json(mirror / "manifest.json")
    documents = _load_documents(mirror, manifest)
    questions = _review_questions(
        mirror=mirror,
        documents=documents,
        max_questions=max_questions,
    )
    change_tasks = _change_tasks(
        mirror=mirror,
        manifest=manifest,
        documents=documents,
        max_change_tasks=max_change_tasks,
    )
    visibility_items = _visibility_items(manifest, documents)

    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "questions.jsonl", questions)
    _write_jsonl(out / "change_tasks.jsonl", change_tasks)
    _write_jsonl(out / "visibility_items.jsonl", visibility_items)
    (out / "REVIEW.md").write_text(
        _review_markdown(
            manifest=manifest,
            questions=questions,
            change_tasks=change_tasks,
            visibility_items=visibility_items,
        ),
        encoding="utf-8",
    )
    pack_manifest = {
        "mode": "review_pack",
        "review_version": REVIEW_VERSION,
        "mirror": str(mirror),
        "repo": manifest["repo"],
        "source_mirror_mode": manifest["mode"],
        "profile": manifest["profile"],
        "zoom": manifest["zoom"],
        "generated_at": _now(),
        "counts": {
            "questions": len(questions),
            "change_tasks": len(change_tasks),
            "visibility_items": len(visibility_items),
            "changed_units": _changed_unit_count(documents),
            "unsupported_files": len(manifest["files"].get("unsupported", [])),
        },
        "limits": {
            "max_questions": max_questions,
            "max_change_tasks": max_change_tasks,
        },
        "files": {
            "questions": "questions.jsonl",
            "change_tasks": "change_tasks.jsonl",
            "visibility_items": "visibility_items.jsonl",
            "review_markdown": "REVIEW.md",
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(pack_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pack_manifest


def evaluate_review_pack(
    review_pack_path: Path | str,
    *,
    mirror_path: Path | str | None = None,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Validate that a review pack supports the human usefulness gates."""

    pack = Path(review_pack_path).resolve()
    pack_manifest = _read_json(pack / "manifest.json")
    mirror = (
        Path(mirror_path).resolve()
        if mirror_path is not None
        else Path(pack_manifest["mirror"]).resolve()
    )
    mirror_manifest = _read_json(mirror / "manifest.json")
    documents = _load_documents(mirror, mirror_manifest)
    questions = _read_jsonl(pack / pack_manifest["files"]["questions"])
    change_tasks = _read_jsonl(pack / pack_manifest["files"]["change_tasks"])
    visibility_items = _read_jsonl(pack / pack_manifest["files"]["visibility_items"])

    expected_changed_units = _changed_unit_ids(documents)
    covered_changed_units = {
        task["unit_id"] for task in change_tasks if task.get("unit_id") in expected_changed_units
    }
    expected_visibility = _expected_visibility_ids(mirror_manifest, documents)
    observed_visibility = {item.get("visibility_id") for item in visibility_items}
    evidence_backed_questions = [
        question for question in questions if question.get("evidence_spans")
    ]
    gates = [
        _gate("review_pack_manifest_mode", pack_manifest.get("mode"), expected="review_pack"),
        _gate("review_questions_present", len(questions), minimum=1),
        _gate(
            "evidence_backed_questions",
            _ratio(len(evidence_backed_questions), len(questions)),
            minimum=1.0,
        ),
        _gate(
            "visibility_items_complete",
            _ratio(len(expected_visibility & observed_visibility), len(expected_visibility)),
            minimum=1.0,
        ),
    ]
    if mirror_manifest["mode"] == "diff":
        gates.append(
            _gate(
                "changed_behavior_task_coverage",
                _ratio(len(covered_changed_units), len(expected_changed_units)),
                minimum=1.0,
            )
        )
    report = {
        "mode": "review_pack_evaluation",
        "review_pack": str(pack),
        "mirror": str(mirror),
        "generated_at": _now(),
        "passed": all(gate["passed"] for gate in gates),
        "gates": gates,
        "counts": {
            "questions": len(questions),
            "change_tasks": len(change_tasks),
            "visibility_items": len(visibility_items),
            "expected_changed_units": len(expected_changed_units),
            "covered_changed_units": len(covered_changed_units),
            "expected_visibility_items": len(expected_visibility),
        },
    }
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def create_human_usefulness_study(
    review_pack_path: Path | str,
    out_path: Path | str,
) -> dict[str, Any]:
    """Create source-only versus mirror-first tasks for timed human review."""

    pack = Path(review_pack_path).resolve()
    out = Path(out_path).resolve()
    pack_manifest = _read_json(pack / "manifest.json")
    questions = _read_jsonl(pack / pack_manifest["files"]["questions"])
    change_tasks = _read_jsonl(pack / pack_manifest["files"]["change_tasks"])
    visibility_items = _read_jsonl(pack / pack_manifest["files"]["visibility_items"])

    paired_source_tasks: list[dict[str, Any]] = []
    paired_mirror_tasks: list[dict[str, Any]] = []
    for index, question in enumerate(questions):
        group_id = f"question-{index}-{question['question_id']}"
        paired_mirror_tasks.append(_study_question_task(question, group_id, condition="mirror"))
        paired_source_tasks.append(_study_question_task(question, group_id, condition="source"))
    for index, task in enumerate(change_tasks):
        group_id = f"change-{index}-{task['task_id']}"
        paired_mirror_tasks.append(_study_change_task(task, group_id, condition="mirror"))
        paired_source_tasks.append(_study_change_task(task, group_id, condition="source"))

    mirror = Path(pack_manifest["mirror"]).resolve()
    visibility_tasks = [
        _study_visibility_task(item, index, mirror=mirror)
        for index, item in enumerate(visibility_items)
    ]
    answer_template = _answer_template(
        paired_mirror_tasks + paired_source_tasks + visibility_tasks
    )

    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "mirror_tasks.jsonl", paired_mirror_tasks)
    _write_jsonl(out / "source_tasks.jsonl", paired_source_tasks)
    _write_jsonl(out / "visibility_tasks.jsonl", visibility_tasks)
    _write_jsonl(out / "answers_template.jsonl", answer_template)
    (out / "README.md").write_text(
        _human_study_readme(
            review_pack=pack,
            mirror_count=len(paired_mirror_tasks),
            source_count=len(paired_source_tasks),
            visibility_count=len(visibility_tasks),
        ),
        encoding="utf-8",
    )
    manifest = {
        "mode": "human_usefulness_study",
        "review_version": REVIEW_VERSION,
        "review_pack": str(pack),
        "mirror": pack_manifest["mirror"],
        "repo": pack_manifest["repo"],
        "source_mirror_mode": pack_manifest["source_mirror_mode"],
        "generated_at": _now(),
        "counts": {
            "paired_task_groups": len(paired_mirror_tasks),
            "mirror_tasks": len(paired_mirror_tasks),
            "source_tasks": len(paired_source_tasks),
            "visibility_tasks": len(visibility_tasks),
            "answer_template_records": len(answer_template),
            "change_task_groups": len(change_tasks),
        },
        "phase6_requirements": PHASE6_HUMAN_STUDY_REQUIREMENTS,
        "files": {
            "mirror_tasks": "mirror_tasks.jsonl",
            "source_tasks": "source_tasks.jsonl",
            "visibility_tasks": "visibility_tasks.jsonl",
            "answers_template": "answers_template.jsonl",
            "study_readme": "README.md",
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def evaluate_human_usefulness_study(
    study_path: Path | str,
    answers_path: Path | str,
    *,
    min_accuracy: float = 0.8,
    min_speedup: float = 1.0,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate timed human answers for mirror usefulness gates."""

    study = Path(study_path).resolve()
    manifest = _read_json(study / "manifest.json")
    mirror_tasks = _read_jsonl(study / manifest["files"]["mirror_tasks"])
    source_tasks = _read_jsonl(study / manifest["files"]["source_tasks"])
    visibility_tasks = _read_jsonl(study / manifest["files"]["visibility_tasks"])
    tasks_by_id = {
        task["study_task_id"]: task
        for task in [*mirror_tasks, *source_tasks, *visibility_tasks]
    }
    answers = _read_jsonl(Path(answers_path))
    valid_answers = [
        _normalise_answer(answer, tasks_by_id[answer["study_task_id"]])
        for answer in answers
        if answer.get("study_task_id") in tasks_by_id
    ]
    answers_by_task = {answer["study_task_id"]: answer for answer in valid_answers}
    paired_groups = sorted({task["task_group_id"] for task in [*mirror_tasks, *source_tasks]})
    answered_groups = [
        group_id
        for group_id in paired_groups
        if _group_answer(group_id, "mirror", mirror_tasks, answers_by_task) is not None
        and _group_answer(group_id, "source", source_tasks, answers_by_task) is not None
    ]
    mirror_answers = [
        _group_answer(group_id, "mirror", mirror_tasks, answers_by_task)
        for group_id in answered_groups
    ]
    source_answers = [
        _group_answer(group_id, "source", source_tasks, answers_by_task)
        for group_id in answered_groups
    ]
    mirror_answers = [answer for answer in mirror_answers if answer is not None]
    source_answers = [answer for answer in source_answers if answer is not None]
    mirror_seconds = [
        answer["elapsed_seconds"] for answer in mirror_answers if answer["elapsed_seconds"] is not None
    ]
    source_seconds = [
        answer["elapsed_seconds"] for answer in source_answers if answer["elapsed_seconds"] is not None
    ]
    mirror_median = _median(mirror_seconds)
    source_median = _median(source_seconds)
    speedup = round(source_median / mirror_median, 6) if mirror_median > 0 else 0.0
    mirror_accuracy = _accuracy(mirror_answers)
    source_accuracy = _accuracy(source_answers)
    mirror_change_answers = [
        answer for answer in mirror_answers if answer.get("task_type") == "changed_behavior"
    ]
    visibility_answers = [
        answers_by_task[task["study_task_id"]]
        for task in visibility_tasks
        if task["study_task_id"] in answers_by_task
    ]
    visibility_acknowledged = [
        answer
        for answer in visibility_answers
        if bool(answer.get("acknowledged")) or bool(answer.get("correct"))
    ]
    real_timed_answers = [answer for answer in valid_answers if _has_real_timed_log(answer)]
    reviewer_answers = [answer for answer in valid_answers if _has_reviewer_identity(answer)]
    source_mirror_answers = [
        answer
        for answer in valid_answers
        if answer.get("condition") in {"mirror", "source"}
        and answer.get("task_type") != "visibility_marker"
    ]
    text_answers = [
        answer for answer in source_mirror_answers if _has_answer_text(answer)
    ]
    answered_task_sets = sorted(
        {
            "diff_mode" if answer.get("task_type") == "changed_behavior" else "whole_repo"
            for answer in source_mirror_answers
        }
    )
    real_timed_logs_present = bool(valid_answers) and len(real_timed_answers) == len(valid_answers)
    reviewer_identity_present = bool(valid_answers) and len(reviewer_answers) == len(valid_answers)
    answer_text_present = bool(source_mirror_answers) and (
        len(text_answers) == len(source_mirror_answers)
    )

    gates = [
        _gate("study_manifest_mode", manifest.get("mode"), expected="human_usefulness_study"),
        _gate("real_timed_reviewer_logs", real_timed_logs_present, expected=True),
        _gate("reviewer_identity_present", reviewer_identity_present, expected=True),
        _gate("answer_text_present", answer_text_present, expected=True),
        _gate("paired_task_groups_present", len(paired_groups), minimum=1),
        _gate(
            "paired_answer_coverage",
            _ratio(len(answered_groups), len(paired_groups)),
            minimum=1.0,
        ),
        _gate("mirror_answer_accuracy", mirror_accuracy, minimum=min_accuracy),
        _gate(
            "mirror_accuracy_not_lower_than_source",
            round(mirror_accuracy - source_accuracy, 6),
            minimum=0.0,
        ),
        _gate("mirror_faster_than_source", speedup, minimum=min_speedup),
        _gate(
            "changed_behavior_accuracy",
            _accuracy(mirror_change_answers),
            minimum=min_accuracy,
        ),
        _gate(
            "visibility_items_acknowledged",
            _ratio(len(visibility_acknowledged), len(visibility_tasks)),
            minimum=1.0,
        ),
    ]
    report = {
        "mode": "human_usefulness_study_evaluation",
        "study": str(study),
        "answers": str(Path(answers_path).resolve()),
        "generated_at": _now(),
        "passed": all(gate["passed"] for gate in gates),
        "gates": gates,
        "phase6_gate_summary": {
            "real_timed_reviewer_logs": real_timed_logs_present,
            "reviewer_identity_present": reviewer_identity_present,
            "answer_text_present": answer_text_present,
            "paired_answer_coverage_complete": _ratio(len(answered_groups), len(paired_groups)) == 1.0,
            "mirror_accuracy_at_or_above_threshold": mirror_accuracy >= min_accuracy,
            "mirror_accuracy_not_lower_than_source": mirror_accuracy >= source_accuracy,
            "mirror_median_faster_than_source": speedup >= min_speedup,
            "changed_behavior_accuracy_at_or_above_threshold": (
                _accuracy(mirror_change_answers) >= min_accuracy
            ),
            "visibility_items_acknowledged": (
                _ratio(len(visibility_acknowledged), len(visibility_tasks)) == 1.0
            ),
        },
        "metrics": {
            "paired_task_groups": len(paired_groups),
            "answered_paired_task_groups": len(answered_groups),
            "mirror_median_seconds": mirror_median,
            "source_median_seconds": source_median,
            "mirror_speedup_over_source": speedup,
            "mirror_answer_accuracy": mirror_accuracy,
            "source_answer_accuracy": source_accuracy,
            "mirror_change_answers": len(mirror_change_answers),
            "visibility_tasks": len(visibility_tasks),
            "visibility_acknowledged": len(visibility_acknowledged),
            "valid_answer_records": len(valid_answers),
            "real_timed_answer_records": len(real_timed_answers),
            "source_mirror_answer_records": len(source_mirror_answers),
            "source_mirror_answer_text_records": len(text_answers),
            "answered_task_sets": answered_task_sets,
        },
    }
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def summarize_human_study_answer_coverage(
    study_path: Path | str,
    answers_path: Path | str | None = None,
    *,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Report Phase 6 answer collection coverage before running full evaluation."""

    study = Path(study_path).resolve()
    manifest = _read_json(study / "manifest.json")
    mirror_tasks = _read_jsonl(study / manifest["files"]["mirror_tasks"])
    source_tasks = _read_jsonl(study / manifest["files"]["source_tasks"])
    visibility_tasks = _read_jsonl(study / manifest["files"]["visibility_tasks"])
    tasks = [*mirror_tasks, *source_tasks, *visibility_tasks]
    tasks_by_id = {task["study_task_id"]: task for task in tasks}

    answers_file = Path(answers_path).resolve() if answers_path is not None else None
    answers = _read_jsonl(answers_file) if answers_file is not None else []
    answer_ids = [answer.get("study_task_id") for answer in answers]
    known_answers = [
        _normalise_answer(answer, tasks_by_id[answer["study_task_id"]])
        for answer in answers
        if answer.get("study_task_id") in tasks_by_id
    ]
    known_answer_ids = {answer["study_task_id"] for answer in known_answers}
    duplicate_answer_ids = sorted(
        answer_id
        for answer_id in set(answer_ids)
        if answer_id is not None and answer_ids.count(answer_id) > 1
    )
    unknown_answer_ids = sorted(
        str(answer_id)
        for answer_id in answer_ids
        if answer_id is not None and answer_id not in tasks_by_id
    )
    pending_tasks = [task for task in tasks if task["study_task_id"] not in known_answer_ids]
    source_mirror_answers = [
        answer
        for answer in known_answers
        if answer.get("condition") in {"mirror", "source"}
        and answer.get("task_type") != "visibility_marker"
    ]
    visibility_answers = [
        answer for answer in known_answers if answer.get("task_type") == "visibility_marker"
    ]
    real_timed_answers = [answer for answer in known_answers if _has_real_timed_log(answer)]
    reviewer_answers = [answer for answer in known_answers if _has_reviewer_identity(answer)]
    text_answers = [
        answer for answer in source_mirror_answers if _has_answer_text(answer)
    ]
    visibility_acknowledged = [
        answer
        for answer in visibility_answers
        if bool(answer.get("acknowledged")) or bool(answer.get("correct"))
    ]
    paired_groups = sorted({task["task_group_id"] for task in [*mirror_tasks, *source_tasks]})
    answered_groups = [
        group_id
        for group_id in paired_groups
        if _group_answer(group_id, "mirror", mirror_tasks, {a["study_task_id"]: a for a in known_answers})
        is not None
        and _group_answer(
            group_id,
            "source",
            source_tasks,
            {a["study_task_id"]: a for a in known_answers},
        )
        is not None
    ]

    gates = [
        _gate("study_manifest_mode", manifest.get("mode"), expected="human_usefulness_study"),
        _gate("answers_file_present", answers_file is not None and answers_file.exists(), expected=True),
        _gate("answer_task_id_coverage", _ratio(len(known_answer_ids), len(tasks)), minimum=1.0),
        _gate("paired_answer_coverage", _ratio(len(answered_groups), len(paired_groups)), minimum=1.0),
        _gate("unknown_answer_ids", len(unknown_answer_ids), expected=0),
        _gate("duplicate_answer_ids", len(duplicate_answer_ids), expected=0),
        _gate(
            "real_timed_reviewer_logs",
            _ratio(len(real_timed_answers), len(known_answers)),
            minimum=1.0,
        ),
        _gate(
            "reviewer_identity_present",
            _ratio(len(reviewer_answers), len(known_answers)),
            minimum=1.0,
        ),
        _gate(
            "answer_text_present",
            _ratio(len(text_answers), len(source_mirror_answers)),
            minimum=1.0,
        ),
        _gate(
            "visibility_items_acknowledged",
            _ratio(len(visibility_acknowledged), len(visibility_tasks)),
            minimum=1.0,
        ),
    ]
    report = {
        "mode": "human_usefulness_study_answer_coverage",
        "study": str(study),
        "answers": str(answers_file) if answers_file is not None else None,
        "generated_at": _now(),
        "passed": all(gate["passed"] for gate in gates),
        "gates": gates,
        "counts": {
            "task_records": len(tasks),
            "answer_records": len(answers),
            "known_answer_records": len(known_answers),
            "pending_task_records": len(pending_tasks),
            "mirror_tasks": len(mirror_tasks),
            "source_tasks": len(source_tasks),
            "visibility_tasks": len(visibility_tasks),
            "paired_task_groups": len(paired_groups),
            "answered_paired_task_groups": len(answered_groups),
            "source_mirror_answer_records": len(source_mirror_answers),
            "source_mirror_answer_text_records": len(text_answers),
            "visibility_answer_records": len(visibility_answers),
            "visibility_acknowledged": len(visibility_acknowledged),
            "real_timed_answer_records": len(real_timed_answers),
            "reviewer_identity_records": len(reviewer_answers),
        },
        "pending_task_ids": [task["study_task_id"] for task in pending_tasks],
        "unknown_answer_ids": unknown_answer_ids,
        "duplicate_answer_ids": duplicate_answer_ids,
    }
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def summarize_human_usefulness_studies(
    report_paths: Iterable[Path | str],
    *,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Summarize whole-repo and diff-mode Phase 6 human-study evaluations."""

    reports = [_read_json(Path(path)) for path in report_paths]
    report_summaries: list[dict[str, Any]] = []
    answered_task_sets: set[str] = set()
    phase6_keys = {
        "real_timed_reviewer_logs",
        "reviewer_identity_present",
        "answer_text_present",
        "paired_answer_coverage_complete",
        "mirror_accuracy_not_lower_than_source",
        "mirror_median_faster_than_source",
        "changed_behavior_accuracy_at_or_above_threshold",
        "visibility_items_acknowledged",
    }
    aggregate_phase6 = {key: bool(reports) for key in phase6_keys}
    for report in reports:
        metrics = report.get("metrics", {})
        task_sets = metrics.get("answered_task_sets", [])
        answered_task_sets.update(task_sets)
        phase6 = report.get("phase6_gate_summary", {})
        for key in phase6_keys:
            if key == "changed_behavior_accuracy_at_or_above_threshold" and "diff_mode" not in task_sets:
                continue
            if key in phase6:
                aggregate_phase6[key] = aggregate_phase6[key] and bool(phase6[key])
        report_summaries.append(
            {
                "path": report.get("answers") or report.get("study"),
                "study": report.get("study"),
                "answers": report.get("answers"),
                "passed": bool(report.get("passed")),
                "answered_task_sets": task_sets,
                "phase6_gate_summary": phase6,
                "metrics": {
                    "mirror_answer_accuracy": metrics.get("mirror_answer_accuracy"),
                    "source_answer_accuracy": metrics.get("source_answer_accuracy"),
                    "mirror_median_seconds": metrics.get("mirror_median_seconds"),
                    "source_median_seconds": metrics.get("source_median_seconds"),
                    "mirror_speedup_over_source": metrics.get("mirror_speedup_over_source"),
                    "valid_answer_records": metrics.get("valid_answer_records"),
                    "real_timed_answer_records": metrics.get("real_timed_answer_records"),
                },
            }
        )
    required_task_sets = {"whole_repo", "diff_mode"}
    gates = [
        _gate("human_study_reports_present", len(reports), minimum=2),
        _gate("whole_repo_answers_present", "whole_repo" in answered_task_sets, expected=True),
        _gate("diff_mode_answers_present", "diff_mode" in answered_task_sets, expected=True),
        _gate("all_reports_passed", all(bool(report.get("passed")) for report in reports), expected=True),
        _gate(
            "real_timed_reviewer_logs_all_reports",
            aggregate_phase6["real_timed_reviewer_logs"],
            expected=True,
        ),
        _gate(
            "reviewer_identity_present_all_reports",
            aggregate_phase6["reviewer_identity_present"],
            expected=True,
        ),
        _gate(
            "answer_text_present_all_reports",
            aggregate_phase6["answer_text_present"],
            expected=True,
        ),
        _gate(
            "mirror_accuracy_not_lower_than_source_all_reports",
            aggregate_phase6["mirror_accuracy_not_lower_than_source"],
            expected=True,
        ),
        _gate(
            "mirror_median_faster_than_source_all_reports",
            aggregate_phase6["mirror_median_faster_than_source"],
            expected=True,
        ),
        _gate(
            "changed_behavior_accuracy_at_or_above_threshold",
            aggregate_phase6["changed_behavior_accuracy_at_or_above_threshold"],
            expected=True,
        ),
        _gate(
            "visibility_items_acknowledged_all_reports",
            aggregate_phase6["visibility_items_acknowledged"],
            expected=True,
        ),
    ]
    summary = {
        "mode": "human_usefulness_study_suite_summary",
        "generated_at": _now(),
        "passed": all(gate["passed"] for gate in gates),
        "reports": report_summaries,
        "gates": gates,
        "phase6_gate_summary": {
            **aggregate_phase6,
            "whole_repo_answers_present": "whole_repo" in answered_task_sets,
            "diff_mode_answers_present": "diff_mode" in answered_task_sets,
            "required_task_sets_present": required_task_sets.issubset(answered_task_sets),
            "all_reports_passed": all(bool(report.get("passed")) for report in reports),
        },
        "metrics": {
            "report_count": len(reports),
            "answered_task_sets": sorted(answered_task_sets),
            "required_task_sets": sorted(required_task_sets),
            "total_valid_answer_records": sum(
                int(report.get("metrics", {}).get("valid_answer_records", 0))
                for report in reports
            ),
            "total_real_timed_answer_records": sum(
                int(report.get("metrics", {}).get("real_timed_answer_records", 0))
                for report in reports
            ),
        },
    }
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def conduct_human_usefulness_study(
    study_path: Path | str,
    out_path: Path | str,
    *,
    reviewer: str,
    task_set: str = "all",
    max_tasks: int | None = None,
    append: bool = False,
    overwrite: bool = False,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    timer: Callable[[], float] = time.perf_counter,
    now_fn: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Run a timed reviewer session and write evaluator-ready answer JSONL."""

    if task_set not in {"all", "source", "mirror", "visibility"}:
        raise ValueError("task_set must be one of: all, source, mirror, visibility")
    if now_fn is None:
        now_fn = _now
    study = Path(study_path).resolve()
    out = Path(out_path).resolve()
    if out.exists() and not append and not overwrite:
        raise FileExistsError(f"answers file already exists: {out}")

    tasks = _study_tasks_for_set(study, task_set=task_set)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    existing_answers = _read_jsonl(out) if append and out.exists() else []
    answered_ids = {answer.get("study_task_id") for answer in existing_answers}
    pending_tasks = [task for task in tasks if task["study_task_id"] not in answered_ids]

    if overwrite and out.exists() and not append:
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    with out.open("a", encoding="utf-8") as handle:
        for index, task in enumerate(pending_tasks, start=1):
            output_fn("")
            output_fn(
                f"[{index}/{len(pending_tasks)}] {task['condition']} "
                f"{task['task_type']} - {task['study_task_id']}"
            )
            output_fn(task["access_rule"] if "access_rule" in task else "Use the listed paths.")
            output_fn(f"Prompt: {task['prompt']}")
            for line in _task_reference_lines(task):
                output_fn(line)
            input_fn("Press Enter when ready to start this timed task.")
            started_at = now_fn()
            start = timer()
            answer = input_fn("Answer: ").strip()
            elapsed_seconds = round(max(timer() - start, 0.0), 6)
            completed_at = now_fn()
            output_fn(f"Expected answer: {task['expected_answer']}")
            if task["task_type"] == "visibility_marker":
                acknowledged = _prompt_bool(
                    "Was the marker visible and acknowledged? [y/n]: ",
                    input_fn=input_fn,
                )
                correct: bool | None = None
            else:
                correct = _prompt_bool(
                    "Mark this answer correct? [y/n]: ",
                    input_fn=input_fn,
                )
                acknowledged = False
            confidence = _prompt_float(
                "Confidence 0-1, blank if unset: ",
                input_fn=input_fn,
            )
            notes = input_fn("Notes, blank if none: ").strip()
            record = {
                "study_task_id": task["study_task_id"],
                "task_group_id": task["task_group_id"],
                "condition": task["condition"],
                "task_type": task["task_type"],
                "reviewer": reviewer,
                "started_at": started_at,
                "completed_at": completed_at,
                "elapsed_seconds": elapsed_seconds,
                "answer": answer,
                "correct": correct,
                "acknowledged": acknowledged,
                "confidence": confidence,
                "notes": notes,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            completed += 1

    return {
        "mode": "human_usefulness_study_conduct",
        "study": str(study),
        "answers": str(out),
        "reviewer": reviewer,
        "task_set": task_set,
        "requested_tasks": len(tasks),
        "skipped_existing": len(tasks) - len(pending_tasks),
        "completed_records": completed,
        "answer_records": len(existing_answers) + completed if append else completed,
        "generated_at": now_fn(),
    }


def _review_questions(
    *,
    mirror: Path,
    documents: list[dict[str, Any]],
    max_questions: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for document in documents:
        for unit in document["units"]:
            for kind in QUESTION_KINDS:
                question = _question_for_kind(mirror=mirror, document=document, unit=unit, kind=kind)
                if question is not None:
                    candidates.append(question)
    candidates.sort(key=lambda item: (-item["priority_score"], item["question_id"]))
    return candidates[:max_questions]


def _question_for_kind(
    *,
    mirror: Path,
    document: dict[str, Any],
    unit: dict[str, Any],
    kind: str,
) -> dict[str, Any] | None:
    claims = _question_claims(unit, kind)
    if not claims:
        return None
    source_path = document["source_path"]
    qualified_name = unit["qualified_name"]
    kind_text = kind.replace("_", " ")
    question_text = (
        f"Using the mirror, what {kind_text} should a reviewer know about "
        f"`{qualified_name}` in `{source_path}`?"
    )
    return {
        "question_id": _review_id("question", source_path, qualified_name, kind),
        "kind": kind,
        "source_path": source_path,
        "unit_id": unit["unit_id"],
        "qualified_name": qualified_name,
        "question": question_text,
        "expected_answer": _answer_summary(claims),
        "evidence_spans": _unique_spans(claims),
        "mirror_paths": _mirror_paths(mirror, source_path),
        "priority_score": _question_priority(unit, claims, kind),
        "reviewer_gate": "answer curated repo questions using mirror evidence before source-only reading",
    }


def _question_claims(unit: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    if kind == "data_ml_behavior":
        claims = []
        for category in (
            "training_loops",
            "losses",
            "model_architecture",
            "optimizer_scheduler",
            "metrics",
            "checkpointing",
            "tensor_shapes",
        ):
            claims.extend(unit["data_ml_details"].get(category, []))
        return claims
    if kind == "side_effects":
        return unit["side_effects"] + unit["state_mutations"] + unit["writes"]
    if kind == "failure_modes":
        return unit["failure_modes"]
    if kind == "return_behavior":
        return unit["returns"]
    if kind == "dependency_calls":
        return unit["calls"] + unit["external_dependencies"]
    if kind == "uncertainty_and_hazards":
        return unit["hazards"] + unit["uncertainty"]
    return []


def _change_tasks(
    *,
    mirror: Path,
    manifest: dict[str, Any],
    documents: list[dict[str, Any]],
    max_change_tasks: int,
) -> list[dict[str, Any]]:
    if manifest["mode"] != "diff":
        return []
    tasks: list[dict[str, Any]] = []
    for document in documents:
        ranges = document.get("diff", {}).get("changed_line_ranges", [])
        for unit in document["units"]:
            if unit.get("change_status") != "changed":
                continue
            claims = _semantic_change_claims(unit)
            tasks.append(
                {
                    "task_id": _review_id("change", document["source_path"], unit["qualified_name"]),
                    "source_path": document["source_path"],
                    "unit_id": unit["unit_id"],
                    "qualified_name": unit["qualified_name"],
                    "task": (
                        "Identify changed behavior from the diff mirror before reading "
                        f"`{document['source_path']}` source."
                    ),
                    "changed_line_ranges": ranges,
                    "semantic_summary": _answer_summary(claims or [unit["algorithm"]]),
                    "evidence_spans": _unique_spans(claims or [unit["algorithm"]]),
                    "mirror_paths": _mirror_paths(mirror, document["source_path"]),
                    "reviewer_gate": "identify changed behavior in a PR from diff IR before reading source",
                }
            )
    tasks.sort(key=lambda item: item["task_id"])
    return tasks[:max_change_tasks]


def _semantic_change_claims(unit: dict[str, Any]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for field in (
        "side_effects",
        "state_mutations",
        "writes",
        "failure_modes",
        "hazards",
        "calls",
        "returns",
        "control_flow",
        "reads",
        "external_dependencies",
        "uncertainty",
    ):
        claims.extend(unit[field])
    for category_claims in unit["data_ml_details"].values():
        claims.extend(category_claims)
    claims.append(unit["algorithm"])
    changed_claims = [claim for claim in claims if claim.get("change_status") == "changed"]
    return changed_claims or claims


def _visibility_items(
    manifest: dict[str, Any],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in manifest["files"].get("unsupported", []):
        items.append(
            {
                "visibility_id": _visibility_id("unsupported_file", item["path"], item["reason"]),
                "kind": "unsupported_file",
                "source_path": item["path"],
                "reason": item["reason"],
                "evidence_spans": item.get("source_spans", []),
                "reviewer_gate": "unsupported areas are visibly marked rather than silently omitted",
            }
        )
    for document in documents:
        for item in document.get("unsupported_reasons", []):
            items.append(
                {
                    "visibility_id": _visibility_id(
                        "unsupported_reason",
                        document["source_path"],
                        item["reason"],
                    ),
                    "kind": "unsupported_reason",
                    "source_path": document["source_path"],
                    "reason": item["reason"],
                    "evidence_spans": item.get("source_spans", []),
                    "reviewer_gate": "unsupported areas are visibly marked rather than silently omitted",
                }
            )
        for unit in document["units"]:
            if unit["confidence"] < 0.75:
                items.append(
                    {
                        "visibility_id": _visibility_id(
                            "low_confidence_unit",
                            document["source_path"],
                            unit["unit_id"],
                        ),
                        "kind": "low_confidence_unit",
                        "source_path": document["source_path"],
                        "unit_id": unit["unit_id"],
                        "qualified_name": unit["qualified_name"],
                        "confidence": unit["confidence"],
                        "evidence_spans": unit["source_spans"],
                        "reviewer_gate": "low-confidence areas are visibly marked rather than silently omitted",
                    }
                )
            for claim in unit["uncertainty"]:
                items.append(_claim_visibility_item(document, unit, claim, kind="uncertainty"))
            for claim in unit["hazards"]:
                if claim.get("confidence", 1.0) < 0.9:
                    items.append(_claim_visibility_item(document, unit, claim, kind="low_confidence_hazard"))
    return _dedupe_visibility(items)


def _claim_visibility_item(
    document: dict[str, Any],
    unit: dict[str, Any],
    claim: dict[str, Any],
    *,
    kind: str,
) -> dict[str, Any]:
    return {
        "visibility_id": _visibility_id(kind, document["source_path"], unit["unit_id"], claim["claim"]),
        "kind": kind,
        "source_path": document["source_path"],
        "unit_id": unit["unit_id"],
        "qualified_name": unit["qualified_name"],
        "claim": claim["claim"],
        "confidence": claim.get("confidence"),
        "evidence_spans": claim.get("source_spans", []),
        "reviewer_gate": "low-confidence or unsupported areas are visibly marked rather than silently omitted",
    }


def _expected_visibility_ids(
    manifest: dict[str, Any],
    documents: list[dict[str, Any]],
) -> set[str]:
    return {item["visibility_id"] for item in _visibility_items(manifest, documents)}


def _dedupe_visibility(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for item in items:
        unique[item["visibility_id"]] = item
    return [unique[key] for key in sorted(unique)]


def _review_markdown(
    *,
    manifest: dict[str, Any],
    questions: list[dict[str, Any]],
    change_tasks: list[dict[str, Any]],
    visibility_items: list[dict[str, Any]],
) -> str:
    lines = [
        "# Semantic Mirror Review Pack",
        "",
        f"- Source mirror mode: `{manifest['mode']}`",
        f"- Repo: `{manifest['repo']}`",
        f"- Profile: `{manifest['profile']}`",
        f"- Zoom: `{manifest['zoom']}`",
        "",
        "## Curated Questions",
    ]
    for question in questions:
        lines.extend(
            [
                "",
                f"### {question['question_id']}",
                "",
                question["question"],
                "",
                f"Expected answer: {question['expected_answer']}",
                f"Evidence: {_format_spans(question['evidence_spans'])}",
            ]
        )
    lines.append("")
    lines.append("## Changed Behavior Tasks")
    if not change_tasks:
        lines.append("")
        lines.append("No diff changed-unit tasks were generated for this mirror.")
    for task in change_tasks:
        lines.extend(
            [
                "",
                f"### {task['task_id']}",
                "",
                task["task"],
                "",
                f"Semantic summary: {task['semantic_summary']}",
                f"Evidence: {_format_spans(task['evidence_spans'])}",
            ]
        )
    lines.append("")
    lines.append("## Visibility Items")
    for item in visibility_items:
        label = item.get("claim") or item.get("reason") or item.get("qualified_name")
        lines.append(f"- `{item['kind']}` `{item['source_path']}`: {label}")
    return "\n".join(lines).rstrip() + "\n"


def _answer_summary(claims: list[dict[str, Any]]) -> str:
    summaries = [claim["claim"] for claim in claims[:4]]
    if len(claims) > 4:
        summaries.append(f"{len(claims) - 4} additional source-backed claims omitted here")
    return " ".join(summaries)


def _question_priority(unit: dict[str, Any], claims: list[dict[str, Any]], kind: str) -> float:
    score = len(claims)
    if kind in {"data_ml_behavior", "side_effects", "uncertainty_and_hazards"}:
        score += 3
    if unit.get("change_status") == "changed":
        score += 4
    score += sum(1 for claim in claims if claim.get("confidence", 1.0) < 0.85)
    return float(score)


def _unique_spans(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int, int]] = set()
    spans: list[dict[str, Any]] = []
    for claim in claims:
        for span in claim.get("source_spans", []):
            key = (span["path"], span["start_line"], span["end_line"])
            if key in seen:
                continue
            seen.add(key)
            spans.append(span)
    return spans


def _mirror_paths(mirror: Path, source_path: str) -> dict[str, str]:
    return {
        "markdown": str(mirror / f"{source_path}.sir.md"),
        "json": str(mirror / f"{source_path}.sir.json"),
    }


def _format_spans(spans: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"`{span['path']}:{span['start_line']}-{span['end_line']}`" for span in spans
    )


def _study_question_task(
    question: dict[str, Any],
    group_id: str,
    *,
    condition: str,
) -> dict[str, Any]:
    task = {
        "study_task_id": f"{condition}-{group_id}",
        "task_group_id": group_id,
        "condition": condition,
        "task_type": "repo_question",
        "source_path": question["source_path"],
        "unit_id": question["unit_id"],
        "qualified_name": question["qualified_name"],
        "prompt": question["question"],
        "expected_answer": question["expected_answer"],
        "evidence_spans": question["evidence_spans"],
        "reviewer_gate": question["reviewer_gate"],
    }
    return _attach_condition_paths(task, question, condition=condition)


def _study_change_task(
    change_task: dict[str, Any],
    group_id: str,
    *,
    condition: str,
) -> dict[str, Any]:
    task = {
        "study_task_id": f"{condition}-{group_id}",
        "task_group_id": group_id,
        "condition": condition,
        "task_type": "changed_behavior",
        "source_path": change_task["source_path"],
        "unit_id": change_task["unit_id"],
        "qualified_name": change_task["qualified_name"],
        "prompt": change_task["task"],
        "expected_answer": change_task["semantic_summary"],
        "changed_line_ranges": change_task["changed_line_ranges"],
        "evidence_spans": change_task["evidence_spans"],
        "reviewer_gate": change_task["reviewer_gate"],
    }
    return _attach_condition_paths(task, change_task, condition=condition)


def _study_visibility_task(item: dict[str, Any], index: int, *, mirror: Path) -> dict[str, Any]:
    label = item.get("claim") or item.get("reason") or item.get("qualified_name") or item["kind"]
    return {
        "study_task_id": f"mirror-visibility-{index}-{item['visibility_id']}",
        "task_group_id": f"visibility-{index}-{item['visibility_id']}",
        "condition": "mirror",
        "task_type": "visibility_marker",
        "source_path": item["source_path"],
        "unit_id": item.get("unit_id"),
        "qualified_name": item.get("qualified_name"),
        "prompt": (
            "Using the mirror, identify the visible unsupported, low-confidence, "
            f"hazard, or uncertainty marker for `{item['source_path']}`."
        ),
        "expected_answer": f"{item['kind']}: {label}",
        "evidence_spans": item.get("evidence_spans", []),
        "mirror_paths": _mirror_paths(mirror, item["source_path"]),
        "source_paths": [],
        "reviewer_gate": item["reviewer_gate"],
    }


def _attach_condition_paths(
    task: dict[str, Any],
    source: dict[str, Any],
    *,
    condition: str,
) -> dict[str, Any]:
    if condition == "mirror":
        task["mirror_paths"] = source.get("mirror_paths", {})
        task["source_paths"] = []
        task["access_rule"] = "Use mirror files before opening source."
    else:
        task["mirror_paths"] = {}
        task["source_paths"] = [source["source_path"]]
        task["access_rule"] = "Use source files only for this timed condition."
    return task


def _answer_template(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "study_task_id": task["study_task_id"],
            "task_group_id": task["task_group_id"],
            "condition": task["condition"],
            "task_type": task["task_type"],
            "reviewer": "",
            "started_at": "",
            "completed_at": "",
            "elapsed_seconds": None,
            "answer": "",
            "correct": None,
            "acknowledged": None if task["task_type"] == "visibility_marker" else False,
            "confidence": None,
            "notes": "",
        }
        for task in tasks
    ]


def _study_tasks_for_set(study: Path, *, task_set: str) -> list[dict[str, Any]]:
    manifest = _read_json(study / "manifest.json")
    mirror_tasks = _read_jsonl(study / manifest["files"]["mirror_tasks"])
    source_tasks = _read_jsonl(study / manifest["files"]["source_tasks"])
    visibility_tasks = _read_jsonl(study / manifest["files"]["visibility_tasks"])
    if task_set == "source":
        return source_tasks
    if task_set == "mirror":
        return mirror_tasks
    if task_set == "visibility":
        return visibility_tasks
    return source_tasks + mirror_tasks + visibility_tasks


def _task_reference_lines(task: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    source_paths = task.get("source_paths", [])
    if source_paths:
        lines.append("Source paths: " + ", ".join(str(path) for path in source_paths))
    mirror_paths = task.get("mirror_paths", {})
    if mirror_paths:
        lines.append(
            "Mirror paths: "
            + ", ".join(f"{kind}={path}" for kind, path in sorted(mirror_paths.items()))
        )
    spans = task.get("evidence_spans", [])
    if spans:
        lines.append("Evidence spans: " + _format_spans(spans[:8]))
    changed = task.get("changed_line_ranges", [])
    if changed:
        lines.append(
            "Changed ranges: "
            + ", ".join(f"{item['start_line']}-{item['end_line']}" for item in changed)
        )
    return lines


def _prompt_bool(prompt: str, *, input_fn: Callable[[str], str]) -> bool:
    while True:
        value = input_fn(prompt).strip().lower()
        if value in {"y", "yes", "true", "1"}:
            return True
        if value in {"n", "no", "false", "0"}:
            return False


def _prompt_float(prompt: str, *, input_fn: Callable[[str], str]) -> float | None:
    while True:
        value = input_fn(prompt).strip()
        if not value:
            return None
        try:
            parsed = float(value)
        except ValueError:
            continue
        if 0.0 <= parsed <= 1.0:
            return parsed


def _normalise_answer(answer: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    elapsed = answer.get("elapsed_seconds")
    return {
        **answer,
        "task_group_id": task["task_group_id"],
        "condition": task["condition"],
        "task_type": task["task_type"],
        "elapsed_seconds": _float_or_none(elapsed),
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_reviewer_identity(answer: dict[str, Any]) -> bool:
    return bool(str(answer.get("reviewer", "")).strip())


def _has_answer_text(answer: dict[str, Any]) -> bool:
    return bool(str(answer.get("answer", "")).strip())


def _has_real_timed_log(answer: dict[str, Any]) -> bool:
    elapsed = answer.get("elapsed_seconds")
    return (
        _has_reviewer_identity(answer)
        and bool(str(answer.get("started_at", "")).strip())
        and bool(str(answer.get("completed_at", "")).strip())
        and isinstance(elapsed, int | float)
        and elapsed > 0.0
    )


def _group_answer(
    group_id: str,
    condition: str,
    tasks: list[dict[str, Any]],
    answers_by_task: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for task in tasks:
        if task["task_group_id"] == group_id and task["condition"] == condition:
            return answers_by_task.get(task["study_task_id"])
    return None


def _accuracy(answers: list[dict[str, Any]]) -> float:
    return _ratio(sum(1 for answer in answers if bool(answer.get("correct"))), len(answers))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[midpoint], 6)
    return round((ordered[midpoint - 1] + ordered[midpoint]) / 2, 6)


def _human_study_readme(
    *,
    review_pack: Path,
    mirror_count: int,
    source_count: int,
    visibility_count: int,
) -> str:
    required_task_sets = ", ".join(PHASE6_HUMAN_STUDY_REQUIREMENTS["required_task_sets"])
    pass_gates = "\n".join(
        f"- `{gate}`" for gate in PHASE6_HUMAN_STUDY_REQUIREMENTS["pass_gates"]
    )
    return f"""# Semantic Mirror Human Usefulness Study

This directory is generated from `{review_pack}`.

Use `source_tasks.jsonl` for the source-only timed condition and
`mirror_tasks.jsonl` for the mirror-first timed condition. Fill one line per task
    in `answers_template.jsonl`, including `reviewer`, `started_at`,
`completed_at`, positive `elapsed_seconds`, `answer`, and reviewer-scored
`correct`. For `visibility_tasks.jsonl`, set `acknowledged` when the reviewer
found the unsupported, low-confidence, hazard, or uncertainty marker in the
mirror.

Task counts:

- Mirror paired tasks: {mirror_count}
- Source paired tasks: {source_count}
- Visibility tasks: {visibility_count}

Phase 6 contract requirements:

- Required task sets: {required_task_sets}
- Required answer source: real timed reviewer logs

Pass gates:

{pass_gates}

Evaluate completed answers with:

```powershell
{PHASE6_HUMAN_STUDY_REQUIREMENTS["evaluation_command"]}
```
"""


def _changed_unit_ids(documents: list[dict[str, Any]]) -> set[str]:
    return {
        unit["unit_id"]
        for document in documents
        for unit in document["units"]
        if unit.get("change_status") == "changed"
    }


def _changed_unit_count(documents: list[dict[str, Any]]) -> int:
    return len(_changed_unit_ids(documents))


def _load_documents(mirror: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _read_json(mirror / f"{source_path}.sir.json")
        for source_path in manifest["files"]["generated"]
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _gate(
    name: str,
    actual: Any,
    *,
    minimum: float | None = None,
    expected: Any = None,
) -> dict[str, Any]:
    if minimum is not None:
        passed = actual >= minimum
        rule = {"minimum": minimum}
    else:
        passed = actual == expected
        rule = {"expected": expected}
    return {"name": name, "actual": actual, "passed": bool(passed), **rule}


def _ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else round(numerator / denominator, 6)


def _review_id(prefix: str, *parts: str) -> str:
    safe = "-".join(_safe_part(part) for part in parts if part)
    return f"{prefix}-{safe}"


def _visibility_id(kind: str, *parts: str) -> str:
    return _review_id(kind, *parts)


def _safe_part(value: str) -> str:
    safe = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in safe.split("-") if part)[:80] or "item"


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
