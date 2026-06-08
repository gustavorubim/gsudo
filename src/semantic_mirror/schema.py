"""Schema helpers and evidence validation for Semantic IR."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

SCHEMA_VERSION = "0.1.0"
MIRROR_VERSION = "0.1.0"
VERIFIER_VERSION = "schema-evidence-v0"
GENERATION_MODEL = "static-python-ast-v0"
SUPPORTED_PROFILES = {"data_ml"}
SUPPORTED_ZOOMS = {"L1", "L2", "L3", "L4"}
DATA_ML_DETAIL_CATEGORIES = (
    "losses",
    "model_architecture",
    "tensor_shapes",
    "training_loops",
    "optimizer_scheduler",
    "metrics",
    "checkpointing",
)


class SchemaValidationError(ValueError):
    """Raised when generated semantic IR violates the evidence contract."""


def make_span(source_path: str, start_line: int, end_line: int) -> dict[str, Any]:
    if start_line < 1 or end_line < start_line:
        raise SchemaValidationError(
            f"invalid span for {source_path}: start={start_line}, end={end_line}"
        )
    return {
        "path": source_path,
        "start_line": start_line,
        "end_line": end_line,
    }


def make_claim(
    claim: str,
    source_spans: Iterable[dict[str, Any]],
    *,
    confidence: float = 0.7,
    **extra: Any,
) -> dict[str, Any]:
    spans = list(source_spans)
    if not spans:
        raise SchemaValidationError(f"claim has no evidence: {claim}")
    payload: dict[str, Any] = {
        "claim": claim,
        "source_spans": spans,
        "confidence": round(confidence, 3),
    }
    payload.update(extra)
    return payload


def validate_profile_and_zoom(profile: str, zoom: str) -> None:
    if profile not in SUPPORTED_PROFILES:
        allowed = ", ".join(sorted(SUPPORTED_PROFILES))
        raise SchemaValidationError(f"unsupported profile {profile!r}; expected one of {allowed}")
    if zoom not in SUPPORTED_ZOOMS:
        allowed = ", ".join(sorted(SUPPORTED_ZOOMS))
        raise SchemaValidationError(f"unsupported zoom {zoom!r}; expected one of {allowed}")


def validate_ir_document(document: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "source_path",
        "language",
        "profile",
        "zoom",
        "units",
        "unsupported_reasons",
    }
    missing = sorted(required - set(document))
    if missing:
        raise SchemaValidationError(f"IR document missing required keys: {missing}")
    if document["schema_version"] != SCHEMA_VERSION:
        raise SchemaValidationError(
            f"unsupported schema version {document['schema_version']!r}"
        )
    validate_profile_and_zoom(document["profile"], document["zoom"])
    if not isinstance(document["units"], list):
        raise SchemaValidationError("IR document units must be a list")
    for unit_index, unit in enumerate(document["units"]):
        validate_unit(unit, f"units[{unit_index}]")


def validate_unit(unit: dict[str, Any], path: str = "unit") -> None:
    required = {
        "unit_id",
        "source_spans",
        "language",
        "symbol_type",
        "name",
        "qualified_name",
        "algorithm",
        "control_flow",
        "reads",
        "writes",
        "calls",
        "returns",
        "side_effects",
        "failure_modes",
        "state_mutations",
        "external_dependencies",
        "data_ml_details",
        "hazards",
        "uncertainty",
        "confidence",
    }
    missing = sorted(required - set(unit))
    if missing:
        raise SchemaValidationError(f"{path} missing required keys: {missing}")
    _validate_spans(unit["source_spans"], f"{path}.source_spans")
    _validate_claimish(unit["algorithm"], f"{path}.algorithm")
    for key in (
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
        if not isinstance(unit[key], list):
            raise SchemaValidationError(f"{path}.{key} must be a list")
        for index, item in enumerate(unit[key]):
            _validate_claimish(item, f"{path}.{key}[{index}]")
    if not isinstance(unit["data_ml_details"], dict):
        raise SchemaValidationError(f"{path}.data_ml_details must be an object")
    missing_data_ml_categories = sorted(
        set(DATA_ML_DETAIL_CATEGORIES) - set(unit["data_ml_details"])
    )
    if missing_data_ml_categories:
        raise SchemaValidationError(
            f"{path}.data_ml_details missing required categories: "
            f"{missing_data_ml_categories}"
        )
    for category, items in unit["data_ml_details"].items():
        if not isinstance(items, list):
            raise SchemaValidationError(f"{path}.data_ml_details.{category} must be a list")
        for index, item in enumerate(items):
            _validate_claimish(item, f"{path}.data_ml_details.{category}[{index}]")


def validate_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "mirror_version",
        "mode",
        "repo",
        "profile",
        "zoom",
        "generated_at",
        "generation_model",
        "verifier_version",
        "languages",
        "files",
        "symbol_graph",
        "coverage",
        "confidence",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise SchemaValidationError(f"manifest missing required keys: {missing}")
    validate_profile_and_zoom(manifest["profile"], manifest["zoom"])
    if manifest["generation_model"] != GENERATION_MODEL:
        raise SchemaValidationError("manifest generation model mismatch")
    if manifest["verifier_version"] != VERIFIER_VERSION:
        raise SchemaValidationError("manifest verifier version mismatch")


def collect_claim_evidence_stats(documents: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    with_evidence = 0
    for document in documents:
        for unit in document["units"]:
            for claim in _iter_unit_claims(unit):
                total += 1
                if claim.get("source_spans"):
                    with_evidence += 1
    coverage = 1.0 if total == 0 else with_evidence / total
    return {
        "generated_claims": total,
        "claims_with_source_span_evidence": with_evidence,
        "claim_evidence_coverage": round(coverage, 6),
    }


def _iter_unit_claims(unit: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield unit["algorithm"]
    for key in (
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
        yield from unit[key]
    for items in unit["data_ml_details"].values():
        yield from items


def _validate_claimish(value: Any, path: str) -> None:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{path} must be an object with source evidence")
    if "claim" not in value:
        raise SchemaValidationError(f"{path} missing claim")
    if "source_spans" not in value:
        raise SchemaValidationError(f"{path} missing source_spans")
    _validate_spans(value["source_spans"], f"{path}.source_spans")
    confidence = value.get("confidence")
    if confidence is not None and not 0 <= confidence <= 1:
        raise SchemaValidationError(f"{path}.confidence must be between 0 and 1")


def _validate_spans(value: Any, path: str) -> None:
    if not isinstance(value, list) or not value:
        raise SchemaValidationError(f"{path} must be a non-empty list")
    for index, span in enumerate(value):
        if not isinstance(span, dict):
            raise SchemaValidationError(f"{path}[{index}] must be an object")
        for key in ("path", "start_line", "end_line"):
            if key not in span:
                raise SchemaValidationError(f"{path}[{index}] missing {key}")
        start = span["start_line"]
        end = span["end_line"]
        if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
            raise SchemaValidationError(f"{path}[{index}] has invalid line range")
