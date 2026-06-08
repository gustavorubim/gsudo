"""Render semantic IR documents to mirror-repository files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from semantic_mirror.schema import validate_manifest


def write_mirror(out_path: Path, documents: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    out_path.mkdir(parents=True, exist_ok=True)
    for document in documents:
        rel_path = document["source_path"]
        json_path = out_path / f"{rel_path}.sir.json"
        md_path = out_path / f"{rel_path}.sir.md"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(_render_markdown(document), encoding="utf-8")

    validate_manifest(manifest)
    (out_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _render_markdown(document: dict[str, Any]) -> str:
    lines = [
        f"# Semantic IR: `{document['source_path']}`",
        "",
        f"- Schema: `{document['schema_version']}`",
        f"- Language: `{document['language']}`",
        f"- Profile: `{document['profile']}`",
        f"- Zoom: `{document['zoom']}`",
        f"- Parser: `{document.get('static_analysis', {}).get('backend', 'unknown')}`",
        "",
    ]
    if document.get("unsupported_reasons"):
        lines.append("## Unsupported Reasons")
        for item in document["unsupported_reasons"]:
            lines.append(f"- {item['reason']}")
        lines.append("")

    lines.append("## Units")
    for unit in document["units"]:
        lines.extend(_render_unit(unit))
    return "\n".join(lines).rstrip() + "\n"


def _render_unit(unit: dict[str, Any]) -> list[str]:
    span_text = ", ".join(_format_span(span) for span in unit["source_spans"])
    lines = [
        "",
        f"### `{unit['qualified_name']}`",
        "",
        f"- Type: `{unit['symbol_type']}`",
        f"- Source: {span_text}",
        f"- Confidence: `{unit['confidence']}`",
    ]
    if unit.get("change_status"):
        lines.append(f"- Change status: `{unit['change_status']}`")
    lines.extend(["", f"**Algorithm:** {_change_label(unit['algorithm'])}{unit['algorithm']['claim']}"])
    for section, key in (
        ("Control Flow", "control_flow"),
        ("Reads", "reads"),
        ("Writes", "writes"),
        ("Calls", "calls"),
        ("Returns", "returns"),
        ("Side Effects", "side_effects"),
        ("Failure Modes", "failure_modes"),
        ("State Mutations", "state_mutations"),
        ("External Dependencies", "external_dependencies"),
        ("Hazards", "hazards"),
        ("Uncertainty", "uncertainty"),
    ):
        lines.extend(_render_claim_list(section, unit[key]))

    data_lines: list[str] = []
    for category, claims in unit["data_ml_details"].items():
        if not claims:
            continue
        data_lines.append(f"- `{category}`")
        for claim in claims:
            data_lines.append(
                f"  - {_change_label(claim)}{claim['claim']} ({_claim_spans(claim)})"
            )
    if data_lines:
        lines.extend(["", "**Data/ML Details:**", *data_lines])
    return lines


def _render_claim_list(section: str, claims: list[dict[str, Any]]) -> list[str]:
    if not claims:
        return []
    lines = ["", f"**{section}:**"]
    for claim in claims:
        label = f"`{claim['name']}`: " if "name" in claim else ""
        lines.append(f"- {_change_label(claim)}{label}{claim['claim']} ({_claim_spans(claim)})")
    return lines


def _claim_spans(claim: dict[str, Any]) -> str:
    return ", ".join(_format_span(span) for span in claim["source_spans"])


def _format_span(span: dict[str, Any]) -> str:
    start = span["start_line"]
    end = span["end_line"]
    if start == end:
        return f"`{span['path']}:{start}`"
    return f"`{span['path']}:{start}-{end}`"


def _change_label(claim: dict[str, Any]) -> str:
    status = claim.get("change_status")
    return f"[{status}] " if status else ""
