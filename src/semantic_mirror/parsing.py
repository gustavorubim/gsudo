"""Tree-sitter static parsing helpers."""

from __future__ import annotations

from typing import Any

from semantic_mirror.source import SourceFile


def analyze_source(source: SourceFile) -> dict[str, Any]:
    if source.language != "python":
        return {
            "backend": "unsupported",
            "language": source.language,
            "available": False,
            "root_node_type": None,
            "node_count": 0,
            "has_error": False,
            "errors": [],
        }
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_python
    except Exception as exc:
        return {
            "backend": "python_ast_fallback",
            "language": source.language,
            "available": False,
            "root_node_type": None,
            "node_count": 0,
            "has_error": False,
            "errors": [
                {
                    "kind": "tree_sitter_unavailable",
                    "message": str(exc),
                    "source_spans": [{"path": source.rel_path, "start_line": 1, "end_line": 1}],
                }
            ],
        }

    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(source.text.encode("utf-8"))
    root = tree.root_node
    errors = _collect_errors(source, root)
    return {
        "backend": "tree_sitter_python",
        "language": source.language,
        "available": True,
        "root_node_type": root.type,
        "node_count": _count_nodes(root),
        "has_error": bool(root.has_error),
        "errors": errors,
    }


def _count_nodes(node: Any) -> int:
    return 1 + sum(_count_nodes(child) for child in node.children)


def _collect_errors(source: SourceFile, root: Any) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if len(errors) >= 20:
            return
        if node.type == "ERROR" or node.is_missing:
            start_line = int(node.start_point[0]) + 1
            end_line = max(start_line, int(node.end_point[0]) + 1)
            errors.append(
                {
                    "kind": "syntax_error_node",
                    "node_type": node.type,
                    "is_missing": bool(node.is_missing),
                    "source_spans": [
                        {
                            "path": source.rel_path,
                            "start_line": start_line,
                            "end_line": end_line,
                        }
                    ],
                }
            )
        for child in node.children:
            walk(child)

    walk(root)
    return errors
