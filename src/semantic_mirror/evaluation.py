"""Gate evaluation for Semantic Mirror mirrors, datasets, and regressions."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semantic_mirror.extractors import extract_source_file
from semantic_mirror.rewards import score_document, score_mirror
from semantic_mirror.schema import validate_ir_document, validate_manifest
from semantic_mirror.source import source_from_text

MIRROR_GATE_DEFAULTS = {
    "parsed_symbol_ir_coverage": 0.9,
    "claim_evidence_coverage": 1.0,
    "diff_changed_unit_recall": 0.95,
}


def evaluate_mirror(
    mirror_path: Path | str,
    *,
    repo_path: Path | str | None = None,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    mirror = Path(mirror_path).resolve()
    manifest = json.loads((mirror / "manifest.json").read_text(encoding="utf-8"))
    repo = Path(repo_path).resolve() if repo_path is not None else Path(manifest["repo"]).resolve()
    documents = _load_documents(mirror, manifest)
    score_report = score_mirror(mirror, repo_path=repo)

    manifest_valid = _validation_result(lambda: validate_manifest(manifest))
    document_validation = [_validation_result(lambda doc=document: validate_ir_document(doc)) for document in documents]
    sidecar_valid = manifest_valid["valid"] and all(result["valid"] for result in document_validation)
    symbol_coverage = _parsed_symbol_ir_coverage(repo, documents)
    evidence_coverage = manifest["coverage"].get("claim_evidence_coverage", 0.0)
    total_penalties = sum(score_report["penalties"].values())
    invented_side_effects = score_report["penalties"].get("invented_side_effects", 0)

    gates = [
        _gate("manifest_and_sidecars_valid", sidecar_valid, expected=True),
        _gate("tree_sitter_parse_available", _tree_sitter_availability(documents), minimum=1.0),
        _gate(
            "parsed_symbol_ir_coverage",
            symbol_coverage,
            minimum=MIRROR_GATE_DEFAULTS["parsed_symbol_ir_coverage"],
        ),
        _gate(
            "claim_evidence_coverage",
            evidence_coverage,
            minimum=MIRROR_GATE_DEFAULTS["claim_evidence_coverage"],
        ),
        _gate("verifier_penalties", total_penalties, maximum=0),
        _gate("invented_side_effects", invented_side_effects, maximum=0),
    ]
    if manifest["mode"] == "diff":
        gates.append(
            _gate(
                "diff_changed_unit_recall",
                _diff_changed_unit_recall(documents),
                minimum=MIRROR_GATE_DEFAULTS["diff_changed_unit_recall"],
            )
        )

    report = {
        "mode": "mirror_evaluation",
        "mirror": str(mirror),
        "repo": str(repo),
        "generated_at": _now(),
        "passed": all(gate["passed"] for gate in gates),
        "gates": gates,
        "manifest_validation": manifest_valid,
        "document_validation": document_validation,
        "score_report": score_report,
    }
    _maybe_write_report(report, out_path)
    return report


def evaluate_dataset(
    dataset_path: Path | str,
    *,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    dataset = Path(dataset_path).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    silver = _read_jsonl(dataset / manifest["files"]["silver"])
    hard_negatives = _read_jsonl(dataset / manifest["files"]["hard_negative"])
    review_queue = _read_jsonl(dataset / manifest["files"]["review_queue"])
    gold = _read_jsonl(dataset / manifest["files"]["gold"])

    hard_negative_auto_reject_rate = _ratio(
        sum(1 for record in hard_negatives if record.get("auto_reject")),
        len(hard_negatives),
    )
    hard_negative_penalty_rate = _ratio(
        sum(1 for record in hard_negatives if record["verifier_report"]["penalties"]),
        len(hard_negatives),
    )
    silver_schema_rate = _ratio(
        sum(1 for record in silver if _dataset_record_has_training_contract(record)),
        len(silver),
    )
    gold_schema_rate = _ratio(
        sum(1 for record in gold if _dataset_record_has_training_contract(record)),
        len(gold),
    )
    review_budget = manifest["curation_budget"]["requested_review_budget"]
    review_queue_size = len(review_queue)
    expected_review_size = min(review_budget, len(silver))

    gates = [
        _gate("silver_records_present", len(silver) > 0, expected=True),
        _gate("silver_training_contract", silver_schema_rate, minimum=1.0),
        _gate("hard_negative_auto_reject_rate", hard_negative_auto_reject_rate, minimum=1.0),
        _gate("hard_negative_penalty_rate", hard_negative_penalty_rate, minimum=1.0),
        _gate("review_queue_budget", review_queue_size, maximum=review_budget),
        _gate("review_queue_expected_size", review_queue_size, expected=expected_review_size),
        _gate("gold_jsonl_parseable", isinstance(gold, list), expected=True),
        _gate("gold_training_contract", gold_schema_rate, minimum=1.0),
    ]
    report = {
        "mode": "dataset_evaluation",
        "dataset": str(dataset),
        "generated_at": _now(),
        "passed": all(gate["passed"] for gate in gates),
        "gates": gates,
        "counts": {
            "silver": len(silver),
            "hard_negative": len(hard_negatives),
            "review_queue": len(review_queue),
            "gold": len(gold),
        },
        "rates": {
            "hard_negative_auto_reject_rate": hard_negative_auto_reject_rate,
            "hard_negative_penalty_rate": hard_negative_penalty_rate,
            "silver_schema_rate": silver_schema_rate,
            "gold_schema_rate": gold_schema_rate,
        },
    }
    _maybe_write_report(report, out_path)
    return report


def compare_regression_reports(
    baseline_report_path: Path | str,
    current_report_path: Path | str,
    *,
    max_score_drop: float = 0.01,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    baseline = json.loads(Path(baseline_report_path).read_text(encoding="utf-8"))
    current = json.loads(Path(current_report_path).read_text(encoding="utf-8"))
    baseline_score = _report_score(baseline)
    current_score = _report_score(current)
    denominator = max(abs(baseline_score), 1)
    score_drop = max(0.0, (baseline_score - current_score) / denominator)
    baseline_penalties = _report_penalty_count(baseline)
    current_penalties = _report_penalty_count(current)

    gates = [
        _gate("score_drop", round(score_drop, 6), maximum=max_score_drop),
        _gate("verifier_penalty_regression", current_penalties, maximum=baseline_penalties),
        _gate("current_report_passed", bool(current.get("passed", False)), expected=True),
    ]
    report = {
        "mode": "regression_comparison",
        "baseline_report": str(Path(baseline_report_path).resolve()),
        "current_report": str(Path(current_report_path).resolve()),
        "generated_at": _now(),
        "passed": all(gate["passed"] for gate in gates),
        "gates": gates,
        "scores": {
            "baseline": baseline_score,
            "current": current_score,
            "drop_fraction": round(score_drop, 6),
            "max_allowed_drop_fraction": max_score_drop,
        },
        "penalties": {
            "baseline": baseline_penalties,
            "current": current_penalties,
        },
    }
    _maybe_write_report(report, out_path)
    return report


def evaluate_model_candidates(
    dataset_path: Path | str,
    candidates_path: Path | str,
    *,
    model_name: str,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    dataset = Path(dataset_path).resolve()
    candidates_file = Path(candidates_path).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    default_repo = _default_repo_from_manifest(manifest)
    source_repos = _source_repo_lookup(manifest)
    references = _candidate_references(dataset, manifest)
    candidates = _read_jsonl(candidates_file)

    results: list[dict[str, Any]] = []
    seen_units: set[str] = set()
    for index, candidate in enumerate(candidates):
        reference = _candidate_reference(candidate, references)
        repo = _reference_repo(
            reference,
            default_repo=default_repo,
            source_repos=source_repos,
        )
        result = _score_candidate(
            candidate,
            reference=reference,
            repo=repo,
            index=index,
        )
        if reference is not None:
            seen_units.add(reference["unit_id"])
        results.append(result)

    penalties = _aggregate_penalties(results)
    rewards = _aggregate_rewards(results)
    total_score = sum(result["score"] for result in results)
    scored = [result for result in results if result["reference_found"]]
    schema_valid = [result for result in scored if result["schema_valid"]]
    hallucination_penalties = _hallucination_penalty_count(penalties)
    expected_units = {record["unit_id"] for record in references.values()}
    coverage = _ratio(len(seen_units), len(expected_units))
    schema_validity = _ratio(len(schema_valid), len(scored))
    average_score = round(total_score / len(scored), 6) if scored else 0.0

    gates = [
        _gate("candidate_records_present", len(candidates) > 0, expected=True),
        _gate("heldout_unit_coverage", coverage, minimum=1.0),
        _gate("schema_validity", schema_validity, minimum=1.0),
    ]
    report = {
        "mode": "model_candidate_evaluation",
        "model_name": model_name,
        "dataset": str(dataset),
        "candidates": str(candidates_file),
        "repo": None if default_repo is None else str(default_repo),
        "source_repos": {key: str(value) for key, value in sorted(source_repos.items())},
        "generated_at": _now(),
        "passed": all(gate["passed"] for gate in gates),
        "gates": gates,
        "metrics": {
            "candidate_records": len(candidates),
            "expected_units": len(expected_units),
            "covered_units": len(seen_units),
            "heldout_unit_coverage": coverage,
            "schema_validity": schema_validity,
            "total_score": total_score,
            "average_static_faithfulness_score": average_score,
            "hallucination_penalties": hallucination_penalties,
        },
        "rewards": rewards,
        "penalties": penalties,
        "results": results,
    }
    _maybe_write_report(report, out_path)
    return report


def compare_model_evaluations(
    baseline_report_path: Path | str,
    current_report_path: Path | str,
    *,
    stage: str,
    min_score_improvement: float = 0.0,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    baseline = json.loads(Path(baseline_report_path).read_text(encoding="utf-8"))
    current = json.loads(Path(current_report_path).read_text(encoding="utf-8"))
    baseline_metrics = baseline["metrics"]
    current_metrics = current["metrics"]
    score_delta = round(
        current_metrics["average_static_faithfulness_score"]
        - baseline_metrics["average_static_faithfulness_score"],
        6,
    )
    schema_delta = round(
        current_metrics["schema_validity"] - baseline_metrics["schema_validity"],
        6,
    )
    hallucination_delta = (
        current_metrics["hallucination_penalties"] - baseline_metrics["hallucination_penalties"]
    )
    gates = [
        _gate("current_report_passed", bool(current.get("passed")), expected=True),
        _gate("schema_validity_not_lower", schema_delta, minimum=0.0),
        _gate("static_faithfulness_improved", score_delta, minimum=min_score_improvement),
    ]
    if stage == "rl":
        gates.append(_gate("hallucination_penalties_not_increased", hallucination_delta, maximum=0))
    elif stage == "sft":
        gates.append(_gate("hallucination_penalties_not_increased", hallucination_delta, maximum=0))
    else:
        raise ValueError("stage must be 'sft' or 'rl'")

    report = {
        "mode": "model_evaluation_comparison",
        "stage": stage,
        "baseline_report": str(Path(baseline_report_path).resolve()),
        "current_report": str(Path(current_report_path).resolve()),
        "generated_at": _now(),
        "passed": all(gate["passed"] for gate in gates),
        "gates": gates,
        "baseline_model": baseline.get("model_name"),
        "current_model": current.get("model_name"),
        "deltas": {
            "average_static_faithfulness_score": score_delta,
            "schema_validity": schema_delta,
            "hallucination_penalties": hallucination_delta,
        },
    }
    _maybe_write_report(report, out_path)
    return report


def _load_documents(mirror: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for rel_path in manifest["files"]["generated"]:
        documents.append(json.loads((mirror / f"{rel_path}.sir.json").read_text(encoding="utf-8")))
    return documents


def _parsed_symbol_ir_coverage(repo: Path, documents: list[dict[str, Any]]) -> float:
    expected_ids: set[str] = set()
    observed_ids: set[str] = set()
    for document in documents:
        source_path = repo / document["source_path"]
        if source_path.exists():
            expected = extract_source_file(
                source_from_text(
                    repo,
                    document["source_path"],
                    source_path.read_text(encoding="utf-8"),
                ),
                profile=document["profile"],
                zoom=document["zoom"],
            )
            for unit in expected["units"]:
                if unit["symbol_type"] in {"function", "async_function", "class"}:
                    expected_ids.add(unit["unit_id"])
        for unit in document["units"]:
            if unit["symbol_type"] in {"function", "async_function", "class"}:
                observed_ids.add(unit["unit_id"])
    return 1.0 if not expected_ids else round(len(expected_ids & observed_ids) / len(expected_ids), 6)


def _tree_sitter_availability(documents: list[dict[str, Any]]) -> float:
    python_documents = [document for document in documents if document["language"] == "python"]
    if not python_documents:
        return 1.0
    available = sum(
        1
        for document in python_documents
        if document.get("static_analysis", {}).get("backend") == "tree_sitter_python"
        and document.get("static_analysis", {}).get("available")
    )
    return round(available / len(python_documents), 6)


def _diff_changed_unit_recall(documents: list[dict[str, Any]]) -> float:
    expected_changed: set[str] = set()
    observed_changed: set[str] = set()
    for document in documents:
        ranges = [
            (item["start_line"], item["end_line"])
            for item in document.get("diff", {}).get("changed_line_ranges", [])
        ]
        for unit in document["units"]:
            if any(_span_intersects_ranges(span, ranges) for span in unit["source_spans"]):
                expected_changed.add(unit["unit_id"])
            if unit.get("change_status") == "changed":
                observed_changed.add(unit["unit_id"])
    if not expected_changed:
        return 1.0
    return round(len(expected_changed & observed_changed) / len(expected_changed), 6)


def _span_intersects_ranges(span: dict[str, Any], ranges: list[tuple[int, int]]) -> bool:
    return any(span["start_line"] <= end and start <= span["end_line"] for start, end in ranges)


def _dataset_record_has_training_contract(record: dict[str, Any]) -> bool:
    return all(
        key in record
        for key in (
            "record_id",
            "profile",
            "zoom",
            "source_path",
            "unit_id",
            "code_slice",
            "static_facts",
            "static_analysis",
            "target",
        )
    ) and record["target"].get("format") == "sir_json_unit"


def _candidate_references(
    dataset: Path,
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for split in ("gold", "silver"):
        for record in _read_jsonl(dataset / manifest["files"][split]):
            references[record["record_id"]] = record
            references[record["unit_id"]] = record
    return references


def _default_repo_from_manifest(manifest: dict[str, Any]) -> Path | None:
    repo = manifest.get("repo")
    if not repo or repo == "multiple":
        return None
    return Path(repo).resolve()


def _source_repo_lookup(manifest: dict[str, Any]) -> dict[str, Path]:
    repos: dict[str, Path] = {}
    for entry in manifest.get("source_repos", []):
        repo_id = entry.get("repo_id")
        path = entry.get("path")
        if repo_id and path:
            repos[repo_id] = Path(path).resolve()
    return repos


def _reference_repo(
    reference: dict[str, Any] | None,
    *,
    default_repo: Path | None,
    source_repos: dict[str, Path],
) -> Path | None:
    if reference is None:
        return default_repo
    if reference.get("source_repo_path"):
        return Path(reference["source_repo_path"]).resolve()
    repo_id = reference.get("source_repo_id")
    if repo_id in source_repos:
        return source_repos[repo_id]
    return default_repo


def _candidate_reference(
    candidate: dict[str, Any],
    references: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for key in ("dataset_record_id", "record_id", "unit_id", "positive_unit_id"):
        value = candidate.get(key)
        if value in references:
            return references[value]
    sir_unit = _candidate_sir_unit(candidate)
    if sir_unit is not None and sir_unit.get("unit_id") in references:
        return references[sir_unit["unit_id"]]
    return None


def _score_candidate(
    candidate: dict[str, Any],
    *,
    reference: dict[str, Any] | None,
    repo: Path | None,
    index: int,
) -> dict[str, Any]:
    sir_unit = _candidate_sir_unit(candidate)
    if reference is None:
        return {
            "index": index,
            "reference_found": False,
            "schema_valid": False,
            "unit_id": None if sir_unit is None else sir_unit.get("unit_id"),
            "score": -3,
            "rewards": {},
            "penalties": {"missing_reference": 1},
            "issues": [{"kind": "missing_reference"}],
        }
    if sir_unit is None:
        return {
            "index": index,
            "reference_found": True,
            "schema_valid": False,
            "unit_id": reference["unit_id"],
            "score": -3,
            "rewards": {},
            "penalties": {"missing_sir_unit": 1},
            "issues": [{"kind": "missing_sir_unit"}],
        }
    if repo is None:
        return {
            "index": index,
            "reference_found": True,
            "schema_valid": False,
            "unit_id": reference["unit_id"],
            "score": -3,
            "rewards": {},
            "penalties": {"missing_repo_path": 1},
            "issues": [{"kind": "missing_repo_path"}],
        }
    scoring_unit = _sir_unit_for_scoring(sir_unit, reference)
    document = {
        "schema_version": "0.1.0",
        "source_path": reference["source_path"],
        "language": scoring_unit.get("language", "python"),
        "profile": reference["profile"],
        "zoom": reference["zoom"],
        "units": [scoring_unit],
        "unsupported_reasons": [],
    }
    score = score_document(document, repo_path=repo)
    schema_valid = not score["penalties"].get("schema_errors")
    return {
        "index": index,
        "reference_found": True,
        "schema_valid": schema_valid,
        "unit_id": reference["unit_id"],
        "source_repo_path": str(repo),
        "score": score["score"],
        "rewards": score["rewards"],
        "penalties": score["penalties"],
        "issues": score["issues"],
    }


def _sir_unit_for_scoring(sir_unit: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    scoring_unit = copy.deepcopy(sir_unit)
    repo_id = reference.get("source_repo_id")
    unit_id = scoring_unit.get("unit_id")
    if repo_id and isinstance(unit_id, str):
        prefix = f"{repo_id}:"
        if unit_id.startswith(prefix):
            scoring_unit["unit_id"] = unit_id[len(prefix) :]
    return scoring_unit


def _candidate_sir_unit(candidate: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("sir_unit", "candidate", "output"):
        value = candidate.get(key)
        if isinstance(value, dict) and key == "sir_unit":
            return value
        if isinstance(value, dict) and "unit_id" in value:
            return value
        if isinstance(value, dict) and isinstance(value.get("sir_unit"), dict):
            return value["sir_unit"]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _aggregate_penalties(results: list[dict[str, Any]]) -> dict[str, int]:
    aggregate: dict[str, int] = {}
    for result in results:
        for key, value in result["penalties"].items():
            aggregate[key] = aggregate.get(key, 0) + value
    return dict(sorted(aggregate.items()))


def _aggregate_rewards(results: list[dict[str, Any]]) -> dict[str, int]:
    aggregate: dict[str, int] = {}
    for result in results:
        for key, value in result["rewards"].items():
            aggregate[key] = aggregate.get(key, 0) + value
    return dict(sorted(aggregate.items()))


def _hallucination_penalty_count(penalties: dict[str, int]) -> int:
    return sum(
        value
        for key, value in penalties.items()
        if key.startswith("invented_") or key in {"claims_without_valid_source_evidence"}
    )


def _report_score(report: dict[str, Any]) -> float:
    if "score_report" in report:
        return float(report["score_report"]["score"])
    if "score" in report:
        return float(report["score"])
    if "scores" in report and "current" in report["scores"]:
        return float(report["scores"]["current"])
    raise ValueError("report has no score field")


def _report_penalty_count(report: dict[str, Any]) -> int:
    if "score_report" in report:
        return sum(report["score_report"]["penalties"].values())
    if "penalties" in report and isinstance(report["penalties"], dict):
        return sum(value for value in report["penalties"].values() if isinstance(value, int))
    return 0


def _gate(
    name: str,
    actual: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    expected: Any = None,
) -> dict[str, Any]:
    if minimum is not None:
        passed = actual >= minimum
        rule = {"minimum": minimum}
    elif maximum is not None:
        passed = actual <= maximum
        rule = {"maximum": maximum}
    else:
        passed = actual == expected
        rule = {"expected": expected}
    return {
        "name": name,
        "actual": actual,
        "passed": bool(passed),
        **rule,
    }


def _validation_result(callback: Any) -> dict[str, Any]:
    try:
        callback()
    except Exception as exc:
        return {"valid": False, "error": str(exc)}
    return {"valid": True, "error": None}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else round(numerator / denominator, 6)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _maybe_write_report(report: dict[str, Any], out_path: Path | str | None) -> None:
    if out_path is None:
        return
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
