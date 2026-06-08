"""Top-level build and diff orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from semantic_mirror.extractors import extract_source_file
from semantic_mirror.gitdiff import changed_line_ranges, changed_paths, file_at_ref
from semantic_mirror.render import write_mirror
from semantic_mirror.schema import (
    GENERATION_MODEL,
    MIRROR_VERSION,
    SCHEMA_VERSION,
    VERIFIER_VERSION,
    collect_claim_evidence_stats,
    validate_ir_document,
    validate_manifest,
    validate_profile_and_zoom,
)
from semantic_mirror.source import SUPPORTED_EXTENSIONS, scan_repository, source_from_text


def build_repository(repo_path: Path | str, out_path: Path | str, *, profile: str, zoom: str) -> dict[str, Any]:
    repo = Path(repo_path).resolve()
    out = Path(out_path).resolve()
    validate_profile_and_zoom(profile, zoom)
    sources, unsupported_files, language_inventory = scan_repository(repo, output_path=out)
    documents = [extract_source_file(source, profile=profile, zoom=zoom) for source in sources]
    manifest = build_manifest(
        repo=repo,
        mode="build",
        profile=profile,
        zoom=zoom,
        documents=documents,
        unsupported_files=unsupported_files,
        language_inventory=language_inventory,
    )
    write_mirror(out, documents, manifest)
    return manifest


def diff_repository(
    repo_path: Path | str,
    out_path: Path | str,
    *,
    base: str,
    head: str,
    profile: str,
    zoom: str,
) -> dict[str, Any]:
    repo = Path(repo_path).resolve()
    out = Path(out_path).resolve()
    validate_profile_and_zoom(profile, zoom)

    changed = changed_paths(repo, base, head)
    documents: list[dict[str, Any]] = []
    unsupported_files: list[dict[str, str]] = []
    language_inventory: dict[str, int] = {}
    diff_entries: list[dict[str, Any]] = []

    for item in changed:
        suffix = Path(item.path).suffix.lower()
        language = SUPPORTED_EXTENSIONS.get(suffix)
        language_inventory[language or suffix.lstrip(".") or "unknown"] = (
            language_inventory.get(language or suffix.lstrip(".") or "unknown", 0) + 1
        )
        diff_entry: dict[str, Any] = {
            "path": item.path,
            "old_path": item.old_path,
            "status": item.status,
            "changed_line_ranges": [],
        }
        if item.status.startswith("D"):
            diff_entries.append(diff_entry)
            unsupported_files.append({"path": item.path, "reason": "deleted in diff head"})
            continue
        if language is None:
            unsupported_files.append(
                {"path": item.path, "reason": f"unsupported extension {suffix!r} in diff"}
            )
            diff_entries.append(diff_entry)
            continue
        text = file_at_ref(repo, head, item.path)
        if text is None:
            unsupported_files.append({"path": item.path, "reason": "file unavailable at diff head"})
            diff_entries.append(diff_entry)
            continue
        source = source_from_text(repo, item.path, text)
        document = extract_source_file(source, profile=profile, zoom=zoom)
        ranges = changed_line_ranges(repo, base, head, item.path)
        diff_entry["changed_line_ranges"] = [
            {"start_line": start, "end_line": end} for start, end in ranges
        ]
        _mark_changed_units(document, ranges)
        document["diff"] = {
            "base": base,
            "head": head,
            "path_status": item.status,
            "old_path": item.old_path,
            "changed_line_ranges": diff_entry["changed_line_ranges"],
        }
        validate_ir_document(document)
        documents.append(document)
        diff_entries.append(diff_entry)

    manifest = build_manifest(
        repo=repo,
        mode="diff",
        profile=profile,
        zoom=zoom,
        documents=documents,
        unsupported_files=unsupported_files,
        language_inventory=language_inventory,
        diff={
            "base": base,
            "head": head,
            "changed_paths": diff_entries,
        },
    )
    write_mirror(out, documents, manifest)
    return manifest


def build_manifest(
    *,
    repo: Path,
    mode: str,
    profile: str,
    zoom: str,
    documents: list[dict[str, Any]],
    unsupported_files: list[dict[str, str]],
    language_inventory: dict[str, int],
    diff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    for document in documents:
        validate_ir_document(document)

    units = [unit for document in documents for unit in document["units"]]
    unsupported_units = [
        unit for unit in units if unit["symbol_type"] == "unsupported_file"
    ]
    evidence_stats = collect_claim_evidence_stats(documents)
    confidence = round(mean(unit["confidence"] for unit in units), 4) if units else 0.0
    supported_files = len(documents)
    coverage = {
        "source_files_seen": supported_files + len(unsupported_files),
        "supported_files": supported_files,
        "files_with_ir": sum(1 for document in documents if document["units"]),
        "generated_units": len(units),
        "unsupported_files": len(unsupported_files),
        "unsupported_units": len(unsupported_units),
        **evidence_stats,
    }
    if supported_files:
        coverage["parsed_file_ir_coverage"] = round(coverage["files_with_ir"] / supported_files, 6)
    else:
        coverage["parsed_file_ir_coverage"] = 1.0

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mirror_version": MIRROR_VERSION,
        "mode": mode,
        "repo": str(repo),
        "profile": profile,
        "zoom": zoom,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "generation_model": GENERATION_MODEL,
        "verifier_version": VERIFIER_VERSION,
        "languages": dict(sorted(language_inventory.items())),
        "static_analysis": _static_analysis_summary(documents),
        "files": {
            "generated": [document["source_path"] for document in documents],
            "unsupported": unsupported_files,
        },
        "symbol_graph": _symbol_graph(documents),
        "coverage": coverage,
        "confidence": confidence,
    }
    if diff is not None:
        manifest["diff"] = diff
    validate_manifest(manifest)
    return manifest


def _symbol_graph(documents: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    unit_by_name: dict[str, str] = {}
    for document in documents:
        for unit in document["units"]:
            node = {
                "id": unit["unit_id"],
                "source_path": document["source_path"],
                "qualified_name": unit["qualified_name"],
                "symbol_type": unit["symbol_type"],
            }
            nodes.append(node)
            unit_by_name[unit["qualified_name"].split(".")[-1]] = unit["unit_id"]

    for document in documents:
        for unit in document["units"]:
            for call in unit["calls"]:
                call_name = call["name"].split(".")[-1]
                target = unit_by_name.get(call_name)
                if target:
                    edges.append(
                        {
                            "source": unit["unit_id"],
                            "target": target,
                            "kind": "static_call_reference",
                        }
                    )
    return {"nodes": nodes, "edges": edges}


def _static_analysis_summary(documents: list[dict[str, Any]]) -> dict[str, Any]:
    backends: dict[str, int] = {}
    files_with_errors = 0
    for document in documents:
        analysis = document.get("static_analysis", {})
        backend = analysis.get("backend", "unknown")
        backends[backend] = backends.get(backend, 0) + 1
        if analysis.get("has_error"):
            files_with_errors += 1
    return {
        "parser_backends": dict(sorted(backends.items())),
        "files_with_tree_sitter_errors": files_with_errors,
    }


def _mark_changed_units(document: dict[str, Any], ranges: list[tuple[int, int]]) -> None:
    for unit in document["units"]:
        spans = unit["source_spans"]
        unit["change_status"] = (
            "changed" if any(_spans_intersect(span, ranges) for span in spans) else "context"
        )
        for claim in _iter_unit_claims(unit):
            claim["change_status"] = (
                "changed"
                if any(_spans_intersect(span, ranges) for span in claim["source_spans"])
                else "context"
            )


def _iter_unit_claims(unit: dict[str, Any]) -> list[dict[str, Any]]:
    claims = [unit["algorithm"]]
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
        claims.extend(unit[key])
    for items in unit["data_ml_details"].values():
        claims.extend(items)
    return claims


def _spans_intersect(span: dict[str, Any], ranges: list[tuple[int, int]]) -> bool:
    for start, end in ranges:
        if span["start_line"] <= end and start <= span["end_line"]:
            return True
    return False
