"""Repository scanning and source-file models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".py": "python",
}

EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}

GENERATED_SUFFIXES = (".sir.json", ".sir.md")


@dataclass(frozen=True)
class SourceFile:
    root: Path
    path: Path
    rel_path: str
    text: str
    language: str

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines() or [""]


def source_from_text(root: Path, rel_path: str, text: str) -> SourceFile:
    path = root / Path(rel_path)
    suffix = path.suffix.lower()
    language = SUPPORTED_EXTENSIONS.get(suffix, "unsupported")
    return SourceFile(
        root=root,
        path=path,
        rel_path=Path(rel_path).as_posix(),
        text=text,
        language=language,
    )


def scan_repository(
    repo_path: Path,
    *,
    include_rel_paths: set[str] | None = None,
    output_path: Path | None = None,
) -> tuple[list[SourceFile], list[dict[str, str]], dict[str, int]]:
    repo_path = repo_path.resolve()
    output_path = output_path.resolve() if output_path is not None else None
    include_normalized = {Path(path).as_posix() for path in include_rel_paths or set()}
    sources: list[SourceFile] = []
    unsupported: list[dict[str, str]] = []
    language_inventory: dict[str, int] = {}

    for path in sorted(repo_path.rglob("*")):
        if not path.is_file():
            continue
        if _is_excluded(path, repo_path, output_path):
            continue
        rel_path = path.relative_to(repo_path).as_posix()
        if include_normalized and rel_path not in include_normalized:
            continue
        if rel_path.endswith(GENERATED_SUFFIXES) or rel_path == "manifest.json":
            continue
        suffix = path.suffix.lower()
        language = SUPPORTED_EXTENSIONS.get(suffix)
        language_inventory[language or suffix.lstrip(".") or "unknown"] = (
            language_inventory.get(language or suffix.lstrip(".") or "unknown", 0) + 1
        )
        if language is None:
            unsupported.append({"path": rel_path, "reason": f"unsupported extension {suffix!r}"})
            continue
        text = path.read_text(encoding="utf-8")
        sources.append(
            SourceFile(
                root=repo_path,
                path=path,
                rel_path=rel_path,
                text=text,
                language=language,
            )
        )
    return sources, unsupported, language_inventory


def _is_excluded(path: Path, repo_path: Path, output_path: Path | None) -> bool:
    try:
        rel_parts = path.relative_to(repo_path).parts
    except ValueError:
        return True
    if any(part in EXCLUDED_DIRS for part in rel_parts):
        return True
    if output_path is None:
        return False
    try:
        path.resolve().relative_to(output_path)
    except ValueError:
        return False
    return True

