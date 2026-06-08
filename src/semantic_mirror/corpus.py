"""Collect multi-repository Data/ML corpora for Semantic Mirror training."""

from __future__ import annotations

import copy
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semantic_mirror.dataset import sample_dataset

CORPUS_VERSION = "0.1.0"


def collect_corpus(
    repo_specs: list[str],
    out_path: Path | str,
    *,
    profile: str,
    zoom: str,
    max_units_per_repo: int = 100,
    review_budget_per_repo: int = 25,
    hard_negatives_per_unit: int = 2,
) -> dict[str, Any]:
    if not repo_specs:
        raise ValueError("at least one --repo is required")
    out = Path(out_path).resolve()
    repos_root = out / "repos"
    datasets_root = out / "datasets"
    aggregate_root = out / "aggregate"
    repos_root.mkdir(parents=True, exist_ok=True)
    datasets_root.mkdir(parents=True, exist_ok=True)
    aggregate_root.mkdir(parents=True, exist_ok=True)

    repo_entries: list[dict[str, Any]] = []
    aggregate_records: dict[str, list[dict[str, Any]]] = {
        "silver": [],
        "hard_negative": [],
        "review_queue": [],
        "gold": [],
    }
    aggregate_languages: dict[str, int] = {}

    for index, spec in enumerate(repo_specs):
        entry = _prepare_repo(spec, repos_root=repos_root, index=index)
        repo_entries.append(entry)
        if not entry["available"]:
            continue
        dataset_dir = datasets_root / entry["repo_id"]
        try:
            dataset_manifest = sample_dataset(
                entry["path"],
                dataset_dir,
                profile=profile,
                zoom=zoom,
                max_units=max_units_per_repo,
                review_budget=review_budget_per_repo,
                hard_negatives_per_unit=hard_negatives_per_unit,
            )
        except Exception as exc:
            entry["available"] = False
            entry["error"] = str(exc)
            continue
        entry["dataset"] = str(dataset_dir)
        entry["dataset_counts"] = dataset_manifest["counts"]
        for language, count in dataset_manifest["languages"].items():
            aggregate_languages[language] = aggregate_languages.get(language, 0) + count
        _append_dataset_records(
            dataset_dir,
            dataset_manifest,
            repo_entry=entry,
            aggregate_records=aggregate_records,
        )

    _write_jsonl(aggregate_root / "silver.jsonl", aggregate_records["silver"])
    _write_jsonl(aggregate_root / "hard_negative.jsonl", aggregate_records["hard_negative"])
    _write_jsonl(aggregate_root / "review_queue.jsonl", aggregate_records["review_queue"])
    _write_jsonl(aggregate_root / "gold.jsonl", aggregate_records["gold"])
    (aggregate_root / "README.md").write_text(_aggregate_readme(), encoding="utf-8")
    aggregate_manifest = _aggregate_manifest(
        out=out,
        aggregate_root=aggregate_root,
        profile=profile,
        zoom=zoom,
        repo_entries=repo_entries,
        aggregate_records=aggregate_records,
        aggregate_languages=aggregate_languages,
        max_units_per_repo=max_units_per_repo,
        review_budget_per_repo=review_budget_per_repo,
        hard_negatives_per_unit=hard_negatives_per_unit,
    )
    (aggregate_root / "manifest.json").write_text(
        json.dumps(aggregate_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "mode": "corpus_collect",
        "corpus_version": CORPUS_VERSION,
        "generated_at": _now(),
        "out": str(out),
        "profile": profile,
        "zoom": zoom,
        "repo_count": len(repo_entries),
        "successful_repos": sum(1 for entry in repo_entries if entry.get("dataset")),
        "failed_repos": sum(1 for entry in repo_entries if not entry.get("dataset")),
        "repos": repo_entries,
        "aggregate_dataset": str(aggregate_root),
        "aggregate_counts": aggregate_manifest["counts"],
        "passed": any(entry.get("dataset") for entry in repo_entries),
        "files": {
            "aggregate_manifest": str(aggregate_root / "manifest.json"),
            "aggregate_silver": str(aggregate_root / "silver.jsonl"),
            "aggregate_hard_negative": str(aggregate_root / "hard_negative.jsonl"),
            "aggregate_review_queue": str(aggregate_root / "review_queue.jsonl"),
            "aggregate_gold": str(aggregate_root / "gold.jsonl"),
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _prepare_repo(spec: str, *, repos_root: Path, index: int) -> dict[str, Any]:
    label, location = _split_repo_spec(spec)
    repo_id = _repo_id(label or location, index)
    entry = {
        "repo_id": repo_id,
        "spec": spec,
        "location": location,
        "source_kind": "git" if _is_git_location(location) else "local",
        "available": False,
        "path": None,
    }
    if entry["source_kind"] == "local":
        path = Path(location).resolve()
        entry["path"] = str(path)
        if path.exists():
            entry["available"] = True
        else:
            entry["error"] = "local repository path does not exist"
        return entry

    clone_path = repos_root / repo_id
    entry["path"] = str(clone_path)
    if clone_path.exists():
        entry["available"] = True
        entry["reused_existing_clone"] = True
        return entry
    result = subprocess.run(
        ["git", "clone", "--depth", "1", location, str(clone_path)],
        check=False,
        capture_output=True,
        encoding="utf-8",
    )
    entry["clone_returncode"] = result.returncode
    if result.returncode == 0:
        entry["available"] = True
    else:
        entry["error"] = result.stderr.strip() or result.stdout.strip()
    return entry


def _append_dataset_records(
    dataset_dir: Path,
    manifest: dict[str, Any],
    *,
    repo_entry: dict[str, Any],
    aggregate_records: dict[str, list[dict[str, Any]]],
) -> None:
    for split in ("silver", "hard_negative", "review_queue", "gold"):
        for record in _read_jsonl(dataset_dir / manifest["files"][split]):
            aggregate_records[split].append(_annotate_record(record, repo_entry=repo_entry, split=split))


def _annotate_record(record: dict[str, Any], *, repo_entry: dict[str, Any], split: str) -> dict[str, Any]:
    annotated = copy.deepcopy(record)
    repo_id = repo_entry["repo_id"]
    annotated["source_repo_id"] = repo_id
    annotated["source_repo_path"] = repo_entry["path"]
    annotated["source_repo_location"] = repo_entry["location"]
    if "record_id" in annotated:
        annotated["record_id"] = f"{repo_id}:{annotated['record_id']}"
    if "unit_id" in annotated:
        annotated["unit_id"] = f"{repo_id}:{annotated['unit_id']}"
    if "positive_unit_id" in annotated:
        annotated["positive_unit_id"] = f"{repo_id}:{annotated['positive_unit_id']}"
    if "silver_record_id" in annotated:
        annotated["silver_record_id"] = f"{repo_id}:{annotated['silver_record_id']}"
    if split in {"silver", "gold"} and isinstance(annotated.get("target"), dict):
        sir_unit = annotated["target"].get("sir_unit")
        if isinstance(sir_unit, dict) and "unit_id" in sir_unit:
            sir_unit["unit_id"] = f"{repo_id}:{sir_unit['unit_id']}"
    if split == "hard_negative" and isinstance(annotated.get("candidate"), dict):
        sir_unit = annotated["candidate"].get("sir_unit")
        if isinstance(sir_unit, dict) and "unit_id" in sir_unit:
            sir_unit["unit_id"] = f"{repo_id}:{sir_unit['unit_id']}"
    return annotated


def _aggregate_manifest(
    *,
    out: Path,
    aggregate_root: Path,
    profile: str,
    zoom: str,
    repo_entries: list[dict[str, Any]],
    aggregate_records: dict[str, list[dict[str, Any]]],
    aggregate_languages: dict[str, int],
    max_units_per_repo: int,
    review_budget_per_repo: int,
    hard_negatives_per_unit: int,
) -> dict[str, Any]:
    return {
        "mode": "corpus_aggregate_dataset",
        "corpus_version": CORPUS_VERSION,
        "repo": "multiple",
        "corpus_root": str(out),
        "profile": profile,
        "zoom": zoom,
        "generated_at": _now(),
        "languages": dict(sorted(aggregate_languages.items())),
        "source_repos": [
            {
                "repo_id": entry["repo_id"],
                "location": entry["location"],
                "path": entry["path"],
                "dataset": entry.get("dataset"),
                "available": entry["available"],
                "error": entry.get("error"),
            }
            for entry in repo_entries
        ],
        "curation_budget": {
            "target_hours_per_week": [3, 5],
            "target_reviewed_examples_per_week": [25, 50],
            "requested_review_budget": len(aggregate_records["review_queue"]),
        },
        "files": {
            "silver": "silver.jsonl",
            "hard_negative": "hard_negative.jsonl",
            "review_queue": "review_queue.jsonl",
            "gold": "gold.jsonl",
            "unsupported": [],
        },
        "counts": {
            "source_repos": len(repo_entries),
            "successful_repos": sum(1 for entry in repo_entries if entry.get("dataset")),
            "silver_records": len(aggregate_records["silver"]),
            "hard_negative_records": len(aggregate_records["hard_negative"]),
            "review_queue_records": len(aggregate_records["review_queue"]),
            "gold_records": len(aggregate_records["gold"]),
        },
        "selection_policy": {
            "max_units_per_repo": max_units_per_repo,
            "review_budget_per_repo": review_budget_per_repo,
            "hard_negatives_per_unit": hard_negatives_per_unit,
        },
        "files_absolute": {
            "manifest": str(aggregate_root / "manifest.json"),
            "silver": str(aggregate_root / "silver.jsonl"),
            "hard_negative": str(aggregate_root / "hard_negative.jsonl"),
            "review_queue": str(aggregate_root / "review_queue.jsonl"),
            "gold": str(aggregate_root / "gold.jsonl"),
        },
    }


def _split_repo_spec(spec: str) -> tuple[str | None, str]:
    if "=" not in spec:
        return None, spec
    label, location = spec.split("=", 1)
    return label.strip() or None, location.strip()


def _is_git_location(location: str) -> bool:
    return location.startswith(("http://", "https://", "ssh://", "git@")) or location.endswith(".git")


def _repo_id(value: str, index: int) -> str:
    name = value.rstrip("/\\").split("/")[-1].split("\\")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    safe = "".join(character.lower() if character.isalnum() else "-" for character in name)
    safe = "-".join(part for part in safe.split("-") if part)
    return f"{index:03d}-{safe or 'repo'}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _aggregate_readme() -> str:
    return """# Semantic Mirror Corpus Aggregate

This directory is generated by `semantic-mirror corpus collect`.

It combines per-repository silver, hard-negative, review-queue, and gold JSONL
files into one aggregate training dataset. Records keep their original
repository-relative source paths and include `source_repo_id` plus
`source_repo_path` metadata so training and curation can trace examples back to
the source repository.
"""


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
