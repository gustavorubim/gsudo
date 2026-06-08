"""Dataset and curation-batch generation for Semantic Mirror training."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semantic_mirror.extractors import extract_source_file
from semantic_mirror.rewards import score_document
from semantic_mirror.schema import (
    GENERATION_MODEL,
    MIRROR_VERSION,
    SCHEMA_VERSION,
    VERIFIER_VERSION,
    make_claim,
)
from semantic_mirror.source import SourceFile, scan_repository

DATASET_VERSION = "0.1.0"


def sample_dataset(
    repo_path: Path | str,
    out_path: Path | str,
    *,
    profile: str,
    zoom: str,
    max_units: int = 200,
    review_budget: int = 50,
    hard_negatives_per_unit: int = 2,
) -> dict[str, Any]:
    repo = Path(repo_path).resolve()
    out = Path(out_path).resolve()
    sources, unsupported_files, language_inventory = scan_repository(repo, output_path=out)
    documents = [extract_source_file(source, profile=profile, zoom=zoom) for source in sources]
    examples = _rank_examples(repo, sources, documents)
    selected = examples[:max_units]
    review_queue = selected[:review_budget]

    silver_records = [_silver_record(example, index) for index, example in enumerate(selected)]
    hard_negative_records = _hard_negative_records(
        repo,
        selected,
        max_variants=hard_negatives_per_unit,
    )
    review_records = [
        _review_record(example, silver_records[index]["record_id"], rank=index + 1)
        for index, example in enumerate(review_queue)
    ]

    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "silver.jsonl", silver_records)
    _write_jsonl(out / "hard_negative.jsonl", hard_negative_records)
    _write_jsonl(out / "review_queue.jsonl", review_records)
    _write_jsonl(out / "gold.jsonl", [])
    (out / "README.md").write_text(_dataset_readme(), encoding="utf-8")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mirror_version": MIRROR_VERSION,
        "dataset_version": DATASET_VERSION,
        "mode": "dataset_sample",
        "repo": str(repo),
        "profile": profile,
        "zoom": zoom,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "generation_model": GENERATION_MODEL,
        "verifier_version": VERIFIER_VERSION,
        "languages": dict(sorted(language_inventory.items())),
        "curation_budget": {
            "target_hours_per_week": [3, 5],
            "target_reviewed_examples_per_week": [25, 50],
            "requested_review_budget": review_budget,
        },
        "files": {
            "silver": "silver.jsonl",
            "hard_negative": "hard_negative.jsonl",
            "review_queue": "review_queue.jsonl",
            "gold": "gold.jsonl",
            "unsupported": unsupported_files,
        },
        "counts": {
            "source_files": len(sources),
            "candidate_units": len(examples),
            "silver_records": len(silver_records),
            "hard_negative_records": len(hard_negative_records),
            "review_queue_records": len(review_records),
            "unsupported_files": len(unsupported_files),
        },
        "selection_policy": {
            "max_units": max_units,
            "hard_negatives_per_unit": hard_negatives_per_unit,
            "priority_features": [
                "data_ml_details",
                "hazards",
                "control_flow",
                "side_effects",
                "failure_modes",
                "low_confidence",
                "source_span_width",
            ],
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def promote_gold_records(
    dataset_path: Path | str,
    record_ids: list[str],
    *,
    labels: list[str] | None = None,
    reviewer: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if not record_ids:
        raise ValueError("at least one record id is required")
    dataset = Path(dataset_path).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    silver = _read_jsonl(dataset / manifest["files"]["silver"])
    gold = _read_jsonl(dataset / manifest["files"]["gold"])
    review_queue = _read_jsonl(dataset / manifest["files"]["review_queue"])
    silver_by_id = {record["record_id"]: record for record in silver}
    silver_by_unit_id = {record["unit_id"]: record for record in silver}
    review_to_silver = {
        record["record_id"]: record["silver_record_id"]
        for record in review_queue
        if record.get("silver_record_id")
    }
    promoted: list[dict[str, Any]] = []
    missing: list[str] = []
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    existing_by_source = {
        record.get("curation", {}).get("source_silver_record_id", record.get("source_silver_record_id")): record
        for record in gold
    }

    for requested_id in record_ids:
        silver_id = review_to_silver.get(requested_id, requested_id)
        record = silver_by_id.get(silver_id) or silver_by_unit_id.get(silver_id)
        if record is None:
            missing.append(requested_id)
            continue
        curated = _gold_record(
            record,
            labels=labels or [],
            reviewer=reviewer,
            notes=notes,
            promoted_at=now,
        )
        existing_by_source[record["record_id"]] = curated
        promoted.append(curated)

    gold_records = list(existing_by_source.values())
    gold_records.sort(key=lambda record: (record["source_path"], record["qualified_name"], record["record_id"]))
    _write_jsonl(dataset / manifest["files"]["gold"], gold_records)

    report = {
        "mode": "dataset_gold_promote",
        "dataset": str(dataset),
        "generated_at": now,
        "requested": len(record_ids),
        "promoted": len(promoted),
        "missing": missing,
        "gold_records": len(gold_records),
        "labels": sorted(set(labels or [])),
        "reviewer": reviewer,
        "records": promoted,
        "passed": not missing and bool(promoted),
    }
    (dataset / "gold_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _rank_examples(
    repo: Path,
    sources: list[SourceFile],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_by_path = {source.rel_path: source for source in sources}
    examples: list[dict[str, Any]] = []
    for document in documents:
        source = source_by_path[document["source_path"]]
        for unit in document["units"]:
            priority, reasons = _priority(unit)
            examples.append(
                {
                    "repo": repo,
                    "document": document,
                    "unit": unit,
                    "source": source,
                    "priority_score": priority,
                    "priority_reasons": reasons,
                }
            )
    return sorted(
        examples,
        key=lambda item: (
            -item["priority_score"],
            item["unit"]["source_spans"][0]["path"],
            item["unit"]["source_spans"][0]["start_line"],
            item["unit"]["qualified_name"],
        ),
    )


def _silver_record(example: dict[str, Any], index: int) -> dict[str, Any]:
    unit = example["unit"]
    document = example["document"]
    return {
        "record_id": _record_id("silver", document["source_path"], unit["unit_id"], index),
        "split": "silver",
        "profile": document["profile"],
        "zoom": document["zoom"],
        "source_path": document["source_path"],
        "unit_id": unit["unit_id"],
        "qualified_name": unit["qualified_name"],
        "symbol_type": unit["symbol_type"],
        "source_spans": unit["source_spans"],
        "code_slice": _code_slice(example["source"], unit),
        "static_facts": _static_facts(unit),
        "static_analysis": document.get("static_analysis", {}),
        "target": {
            "format": "sir_json_unit",
            "sir_unit": unit,
        },
        "priority_score": example["priority_score"],
        "priority_reasons": example["priority_reasons"],
        "labels": [],
    }


def _gold_record(
    silver_record: dict[str, Any],
    *,
    labels: list[str],
    reviewer: str | None,
    notes: str | None,
    promoted_at: str,
) -> dict[str, Any]:
    record = copy.deepcopy(silver_record)
    record["record_id"] = _gold_record_id(silver_record["record_id"])
    record["split"] = "gold"
    record["labels"] = sorted(set([*silver_record.get("labels", []), *labels]))
    record["curation"] = {
        "source_silver_record_id": silver_record["record_id"],
        "promoted_at": promoted_at,
        "reviewer": reviewer,
        "notes": notes,
        "labels": record["labels"],
        "review_task": "verified_or_labeled_semantic_ir",
    }
    return record


def _gold_record_id(silver_record_id: str) -> str:
    if silver_record_id.startswith("silver-"):
        return "gold-" + silver_record_id[len("silver-") :]
    return f"gold-{silver_record_id}"


def _review_record(example: dict[str, Any], silver_record_id: str, *, rank: int) -> dict[str, Any]:
    unit = example["unit"]
    return {
        "record_id": _record_id("review", unit["source_spans"][0]["path"], unit["unit_id"], rank),
        "silver_record_id": silver_record_id,
        "rank": rank,
        "source_path": unit["source_spans"][0]["path"],
        "unit_id": unit["unit_id"],
        "qualified_name": unit["qualified_name"],
        "priority_score": example["priority_score"],
        "priority_reasons": example["priority_reasons"],
        "review_task": "label_errors_or_promote_to_gold",
        "label_schema": [
            "missing_call",
            "missing_return",
            "missing_write",
            "missing_control_flow",
            "missing_data_ml_detail",
            "invented_behavior",
            "bad_source_span",
            "low_confidence_claim",
            "unsupported_runtime_dependency",
        ],
    }


def _hard_negative_records(
    repo: Path,
    examples: list[dict[str, Any]],
    *,
    max_variants: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example_index, example in enumerate(examples):
        for variant_index, candidate in enumerate(_negative_candidates(example["unit"], max_variants)):
            candidate_document = _document_for_unit(example["document"], candidate["unit"])
            verifier_report = score_document(candidate_document, repo_path=repo)
            records.append(
                {
                    "record_id": _record_id(
                        f"hard_negative_{candidate['kind']}",
                        example["document"]["source_path"],
                        example["unit"]["unit_id"],
                        example_index,
                        variant_index,
                    ),
                    "split": "hard_negative",
                    "positive_unit_id": example["unit"]["unit_id"],
                    "source_path": example["document"]["source_path"],
                    "qualified_name": example["unit"]["qualified_name"],
                    "negative_kind": candidate["kind"],
                    "candidate": {
                        "format": "sir_json_unit",
                        "sir_unit": candidate["unit"],
                    },
                    "verifier_report": verifier_report,
                    "auto_reject": bool(verifier_report["penalties"]),
                    "expected_error_labels": candidate["expected_error_labels"],
                }
            )
    return records


def _negative_candidates(unit: dict[str, Any], max_variants: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if unit["calls"]:
        corrupted = copy.deepcopy(unit)
        corrupted["calls"] = corrupted["calls"][1:]
        candidates.append(
            {
                "kind": "missing_call",
                "unit": corrupted,
                "expected_error_labels": ["missing_call"],
            }
        )
    elif unit["returns"]:
        corrupted = copy.deepcopy(unit)
        corrupted["returns"] = corrupted["returns"][1:]
        candidates.append(
            {
                "kind": "missing_return",
                "unit": corrupted,
                "expected_error_labels": ["missing_return"],
            }
        )

    corrupted = copy.deepcopy(unit)
    corrupted["calls"].append(
        make_claim(
            "Calls `invented.semantic_mirror_behavior`.",
            unit["source_spans"],
            confidence=0.1,
            name="invented.semantic_mirror_behavior",
            kind="invented_call",
        )
    )
    candidates.append(
        {
            "kind": "invented_call",
            "unit": corrupted,
            "expected_error_labels": ["invented_behavior"],
        }
    )

    corrupted = copy.deepcopy(unit)
    bad_span = dict(unit["source_spans"][0])
    bad_span["start_line"] = bad_span["end_line"] + 1000
    bad_span["end_line"] = bad_span["start_line"]
    corrupted["algorithm"] = {
        **corrupted["algorithm"],
        "source_spans": [bad_span],
    }
    candidates.append(
        {
            "kind": "bad_source_span",
            "unit": corrupted,
            "expected_error_labels": ["bad_source_span"],
        }
    )
    return candidates[:max_variants]


def _document_for_unit(document: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": document["schema_version"],
        "source_path": document["source_path"],
        "language": document["language"],
        "profile": document["profile"],
        "zoom": document["zoom"],
        "units": [unit],
        "unsupported_reasons": document["unsupported_reasons"],
    }


def _priority(unit: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    data_ml_count = sum(len(items) for items in unit["data_ml_details"].values())
    if data_ml_count:
        score += data_ml_count * 5
        reasons.append("data_ml_details")
    if unit["hazards"]:
        score += len(unit["hazards"]) * 4
        reasons.append("hazards")
    if unit["control_flow"]:
        score += len(unit["control_flow"]) * 3
        reasons.append("control_flow")
    if unit["side_effects"]:
        score += len(unit["side_effects"]) * 3
        reasons.append("side_effects")
    if unit["failure_modes"]:
        score += len(unit["failure_modes"]) * 3
        reasons.append("failure_modes")
    low_confidence = max(0, int((1 - unit["confidence"]) * 10))
    if low_confidence:
        score += low_confidence
        reasons.append("low_confidence")
    span_width = sum(span["end_line"] - span["start_line"] + 1 for span in unit["source_spans"])
    if span_width >= 20:
        score += min(10, span_width // 10)
        reasons.append("source_span_width")
    return score, reasons or ["baseline_static_facts"]


def _static_facts(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": unit["algorithm"],
        "control_flow": unit["control_flow"],
        "reads": unit["reads"],
        "writes": unit["writes"],
        "calls": unit["calls"],
        "returns": unit["returns"],
        "side_effects": unit["side_effects"],
        "failure_modes": unit["failure_modes"],
        "state_mutations": unit["state_mutations"],
        "external_dependencies": unit["external_dependencies"],
        "data_ml_details": unit["data_ml_details"],
        "hazards": unit["hazards"],
        "uncertainty": unit["uncertainty"],
    }


def _code_slice(source: SourceFile, unit: dict[str, Any]) -> dict[str, Any]:
    start = min(span["start_line"] for span in unit["source_spans"])
    end = max(span["end_line"] for span in unit["source_spans"])
    lines = source.lines[start - 1 : end]
    return {
        "path": source.rel_path,
        "start_line": start,
        "end_line": end,
        "text": "\n".join(lines),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _record_id(prefix: str, source_path: str, unit_id: str, *numbers: int) -> str:
    safe = "".join(
        character if character.isalnum() else "-"
        for character in f"{source_path}-{unit_id}"
    ).strip("-")
    suffix = "-".join(str(number) for number in numbers)
    return f"{prefix}-{safe}-{suffix}" if suffix else f"{prefix}-{safe}"


def _dataset_readme() -> str:
    return """# Semantic Mirror Dataset Batch

This directory is generated by `semantic-mirror dataset sample`.

- `silver.jsonl` contains static-verifier-backed SIR unit examples.
- `hard_negative.jsonl` contains deterministic corrupted candidates with verifier reports.
- `review_queue.jsonl` ranks the examples most worth human review.
- `gold.jsonl` starts empty; promote reviewed silver records here after curation.

Prefer labeling errors over rewriting examples from scratch. The default review
budget is designed for 25-50 reviewed examples per week.
"""
