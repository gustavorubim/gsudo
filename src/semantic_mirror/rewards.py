"""Deterministic reward scoring for generated Semantic IR mirrors."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from semantic_mirror.extractors import extract_source_file
from semantic_mirror.schema import SchemaValidationError, validate_ir_document
from semantic_mirror.source import source_from_text

REWARD_FIELDS = {
    "calls": 1,
    "control_flow": 1,
    "side_effects": 1,
    "returns": 1,
    "writes": 1,
    "state_mutations": 1,
    "failure_modes": 1,
}

INVENTED_PENALTY = -2
MISSING_PENALTY = -1
UNSUPPORTED_EVIDENCE_PENALTY = -3


def score_mirror(mirror_path: Path | str, *, repo_path: Path | str | None = None) -> dict[str, Any]:
    mirror = Path(mirror_path).resolve()
    manifest_path = mirror / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo = Path(repo_path).resolve() if repo_path is not None else Path(manifest["repo"]).resolve()

    file_reports: list[dict[str, Any]] = []
    total_score = 0
    aggregate_rewards: Counter[str] = Counter()
    aggregate_penalties: Counter[str] = Counter()

    for rel_path in manifest["files"]["generated"]:
        document_path = mirror / f"{rel_path}.sir.json"
        document = json.loads(document_path.read_text(encoding="utf-8"))
        report = score_document(document, repo_path=repo)
        file_reports.append(report)
        total_score += report["score"]
        aggregate_rewards.update(report["rewards"])
        aggregate_penalties.update(report["penalties"])

    return {
        "mirror": str(mirror),
        "repo": str(repo),
        "score": total_score,
        "rewards": dict(aggregate_rewards),
        "penalties": dict(aggregate_penalties),
        "files": file_reports,
    }


def score_document(document: dict[str, Any], *, repo_path: Path) -> dict[str, Any]:
    rewards: Counter[str] = Counter()
    penalties: Counter[str] = Counter()
    issues: list[dict[str, Any]] = []

    try:
        validate_ir_document(document)
    except SchemaValidationError as exc:
        penalties["schema_errors"] += 1
        issues.append({"kind": "schema_error", "message": str(exc)})
        return {
            "source_path": document.get("source_path", "<unknown>"),
            "score": UNSUPPORTED_EVIDENCE_PENALTY,
            "rewards": dict(rewards),
            "penalties": dict(penalties),
            "issues": issues,
        }

    span_failures = _span_failures(document, repo_path)
    for failure in span_failures:
        penalties["claims_without_valid_source_evidence"] += 1
        issues.append(failure)

    expected = _expected_document(document, repo_path)
    if expected is not None:
        expected_units = {unit["unit_id"]: unit for unit in expected["units"]}
        for unit in document["units"]:
            expected_unit = expected_units.get(unit["unit_id"])
            if expected_unit is None:
                penalties["invented_units"] += 1
                issues.append({"kind": "invented_unit", "unit_id": unit["unit_id"]})
                continue
            for field, reward_value in REWARD_FIELDS.items():
                observed_keys = _claim_keys(unit[field], field)
                expected_keys = _claim_keys(expected_unit[field], field)
                _compare_claim_sets(
                    field=field,
                    observed_keys=observed_keys,
                    expected_keys=expected_keys,
                    reward_value=reward_value,
                    unit_id=unit["unit_id"],
                    rewards=rewards,
                    penalties=penalties,
                    issues=issues,
                )
            for category, expected_claims in expected_unit["data_ml_details"].items():
                observed_keys = _claim_keys(unit["data_ml_details"][category], category)
                expected_keys = _claim_keys(expected_claims, category)
                _compare_claim_sets(
                    field=f"data_ml_{category}",
                    observed_keys=observed_keys,
                    expected_keys=expected_keys,
                    reward_value=1,
                    unit_id=unit["unit_id"],
                    rewards=rewards,
                    penalties=penalties,
                    issues=issues,
                )

    score = sum(rewards.values())
    score += INVENTED_PENALTY * sum(
        count for name, count in penalties.items() if name.startswith("invented_")
    )
    score += MISSING_PENALTY * sum(
        count for name, count in penalties.items() if name.startswith("missing_")
    )
    score += UNSUPPORTED_EVIDENCE_PENALTY * penalties[
        "claims_without_valid_source_evidence"
    ]
    score += UNSUPPORTED_EVIDENCE_PENALTY * penalties["schema_errors"]

    return {
        "source_path": document["source_path"],
        "score": score,
        "rewards": dict(rewards),
        "penalties": dict(penalties),
        "issues": issues,
    }


def _expected_document(document: dict[str, Any], repo_path: Path) -> dict[str, Any] | None:
    source_path = repo_path / document["source_path"]
    if not source_path.exists():
        return None
    source = source_from_text(
        repo_path,
        document["source_path"],
        source_path.read_text(encoding="utf-8"),
    )
    return extract_source_file(source, profile=document["profile"], zoom=document["zoom"])


def _span_failures(document: dict[str, Any], repo_path: Path) -> list[dict[str, Any]]:
    line_counts: dict[str, int] = {}
    failures: list[dict[str, Any]] = []
    for unit in document["units"]:
        for claim in _iter_claims(unit):
            spans = claim.get("source_spans") or []
            if not spans:
                failures.append(
                    {
                        "kind": "missing_source_span",
                        "unit_id": unit["unit_id"],
                        "claim": claim.get("claim"),
                    }
                )
                continue
            for span in spans:
                rel_path = span.get("path")
                if rel_path not in line_counts:
                    source_path = repo_path / rel_path
                    if source_path.exists():
                        text = source_path.read_text(encoding="utf-8")
                        line_counts[rel_path] = len(text.splitlines() or [""])
                    else:
                        line_counts[rel_path] = 0
                if line_counts[rel_path] == 0:
                    failures.append(
                        {
                            "kind": "missing_source_file",
                            "unit_id": unit["unit_id"],
                            "path": rel_path,
                            "claim": claim.get("claim"),
                        }
                    )
                    continue
                if span["end_line"] > line_counts[rel_path]:
                    failures.append(
                        {
                            "kind": "source_span_out_of_range",
                            "unit_id": unit["unit_id"],
                            "path": rel_path,
                            "start_line": span["start_line"],
                            "end_line": span["end_line"],
                            "line_count": line_counts[rel_path],
                            "claim": claim.get("claim"),
                        }
                    )
    return failures


def _iter_claims(unit: dict[str, Any]) -> list[dict[str, Any]]:
    claims = [unit["algorithm"]]
    for field in (
        "control_flow",
        "reads",
        "writes",
        "calls",
        "returns",
        "side_effects",
        "failure_modes",
        "state_mutations",
        "external_dependencies",
        "hazards",
        "uncertainty",
    ):
        claims.extend(unit[field])
    for category_claims in unit["data_ml_details"].values():
        claims.extend(category_claims)
    return claims


def _compare_claim_sets(
    *,
    field: str,
    observed_keys: set[str],
    expected_keys: set[str],
    reward_value: int,
    unit_id: str,
    rewards: Counter[str],
    penalties: Counter[str],
    issues: list[dict[str, Any]],
) -> None:
    preserved = observed_keys & expected_keys
    invented = observed_keys - expected_keys
    missing = expected_keys - observed_keys
    if preserved:
        rewards[f"preserved_{field}"] += len(preserved) * reward_value
    if invented:
        penalties[f"invented_{field}"] += len(invented)
        issues.append(
            {
                "kind": f"invented_{field}",
                "unit_id": unit_id,
                "items": sorted(invented),
            }
        )
    if missing:
        penalties[f"missing_{field}"] += len(missing)
        issues.append(
            {
                "kind": f"missing_{field}",
                "unit_id": unit_id,
                "items": sorted(missing),
            }
        )


def _claim_keys(claims: list[dict[str, Any]], field: str) -> set[str]:
    keys: set[str] = set()
    for claim in claims:
        if field in {"calls", "writes"}:
            keys.add(_claim_identity(claim, "name"))
        elif field == "control_flow":
            keys.add(
                "|".join(
                    [
                        str(claim.get("kind", "")),
                        str(claim.get("predicate", "")),
                        str(claim.get("branch_count", "")),
                    ]
                )
            )
        elif field == "returns":
            keys.add(_claim_identity(claim, "value"))
        elif field == "state_mutations":
            keys.add(_claim_identity(claim, "target"))
        elif field == "failure_modes":
            keys.add(_claim_identity(claim, "exception", "predicate"))
        elif field.startswith("data_ml_"):
            keys.add(_claim_identity(claim, "call", "kind"))
        else:
            keys.add(claim["claim"])
    return keys


def _claim_identity(claim: dict[str, Any], *fields: str) -> str:
    for field in fields:
        value = claim.get(field)
        if value:
            return str(value)
    backticked = _first_backticked(claim["claim"])
    return backticked if backticked is not None else claim["claim"]


def _first_backticked(text: str) -> str | None:
    match = re.search(r"`([^`]+)`", text)
    return None if match is None else match.group(1)
