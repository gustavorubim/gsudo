"""Git diff helpers for Semantic Mirror diff mode."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChangedPath:
    status: str
    path: str
    old_path: str | None = None


HUNK_RE = re.compile(r"@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@")


def changed_paths(repo_path: Path, base: str, head: str) -> list[ChangedPath]:
    result = _git(
        repo_path,
        "diff",
        "--name-status",
        "--find-renames",
        base,
        head,
        "--",
    )
    paths: list[ChangedPath] = []
    for line in result.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") and len(parts) == 3:
            paths.append(ChangedPath(status=status, old_path=parts[1], path=parts[2]))
        elif len(parts) >= 2:
            paths.append(ChangedPath(status=status, path=parts[1]))
    return paths


def file_at_ref(repo_path: Path, ref: str, rel_path: str) -> str | None:
    try:
        return _git(repo_path, "show", f"{ref}:{rel_path}")
    except subprocess.CalledProcessError:
        disk_path = repo_path / rel_path
        if disk_path.exists():
            return disk_path.read_text(encoding="utf-8")
        return None


def changed_line_ranges(repo_path: Path, base: str, head: str, rel_path: str) -> list[tuple[int, int]]:
    try:
        diff_text = _git(repo_path, "diff", "--unified=0", base, head, "--", rel_path)
    except subprocess.CalledProcessError:
        return []
    ranges: list[tuple[int, int]] = []
    for match in HUNK_RE.finditer(diff_text):
        start = int(match.group("new_start"))
        count = int(match.group("new_count") or "1")
        if count == 0:
            continue
        ranges.append((start, start + count - 1))
    return ranges


def _git(repo_path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    return result.stdout

