"""Static semantic extraction for supported source languages."""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from semantic_mirror.parsing import analyze_source
from semantic_mirror.schema import SCHEMA_VERSION, make_claim, make_span, validate_ir_document
from semantic_mirror.source import SourceFile


DATA_ML_CATEGORIES = (
    "losses",
    "model_architecture",
    "tensor_shapes",
    "training_loops",
    "optimizer_scheduler",
    "metrics",
    "checkpointing",
)

MUTATING_METHODS = {
    "add",
    "append",
    "clear",
    "extend",
    "insert",
    "pop",
    "popitem",
    "remove",
    "reverse",
    "setdefault",
    "sort",
    "update",
}

SIDE_EFFECT_CALLS = {
    "print",
    "open",
    "input",
    "exec",
    "eval",
    "subprocess.run",
    "subprocess.call",
    "subprocess.Popen",
    "os.remove",
    "os.unlink",
    "os.rename",
    "os.replace",
    "os.rmdir",
    "shutil.rmtree",
    "Path.write_text",
    "Path.write_bytes",
    "torch.save",
    "json.dump",
    "pickle.dump",
}

LOSS_TOKENS = {
    "loss",
    "criterion",
    "cross_entropy",
    "mse_loss",
    "l1_loss",
    "nll_loss",
    "bce_loss",
    "smooth_l1_loss",
}

ARCHITECTURE_TOKENS = {
    "Module",
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "Embedding",
    "LSTM",
    "GRU",
    "RNN",
    "Transformer",
    "Sequential",
    "Dropout",
    "BatchNorm1d",
    "BatchNorm2d",
    "LayerNorm",
    "ReLU",
    "GELU",
    "Softmax",
}

TENSOR_TOKENS = {
    "reshape",
    "view",
    "permute",
    "transpose",
    "unsqueeze",
    "squeeze",
    "flatten",
    "shape",
    "size",
    "to",
    "cuda",
    "cpu",
    "detach",
    "float",
    "long",
}

OPTIMIZER_TOKENS = {
    "optimizer",
    "scheduler",
    "Adam",
    "AdamW",
    "SGD",
    "RMSprop",
    "Adagrad",
    "lr_scheduler",
    "zero_grad",
}

METRIC_TOKENS = {
    "accuracy",
    "acc",
    "precision",
    "recall",
    "f1",
    "auc",
    "metric",
    "metrics",
    "confusion_matrix",
}

CHECKPOINT_TOKENS = {
    "checkpoint",
    "state_dict",
    "load_state_dict",
    "torch.save",
    "torch.load",
    "save_pretrained",
    "from_pretrained",
}

HAZARD_CALLS = {
    "getattr": "dynamic attribute lookup",
    "setattr": "dynamic attribute mutation",
    "hasattr": "dynamic attribute test",
    "eval": "dynamic code evaluation",
    "exec": "dynamic code execution",
    "__import__": "dynamic import",
    "importlib.import_module": "dynamic import",
    "random.random": "nondeterminism",
    "torch.randn": "nondeterminism",
    "numpy.random": "nondeterminism",
}


@dataclass
class FactBucket:
    items: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))

    def add(self, name: str, span: dict[str, Any]) -> None:
        if span not in self.items[name]:
            self.items[name].append(span)

    def claims(self, template: str, *, confidence: float = 0.8, **extra: Any) -> list[dict[str, Any]]:
        claims: list[dict[str, Any]] = []
        for name in sorted(self.items):
            claims.append(
                make_claim(
                    template.format(name=name),
                    self.items[name],
                    confidence=confidence,
                    name=name,
                    **extra,
                )
            )
        return claims


@dataclass
class UnitFacts:
    reads: FactBucket = field(default_factory=FactBucket)
    writes: FactBucket = field(default_factory=FactBucket)
    calls: FactBucket = field(default_factory=FactBucket)
    returns: list[dict[str, Any]] = field(default_factory=list)
    control_flow: list[dict[str, Any]] = field(default_factory=list)
    side_effects: list[dict[str, Any]] = field(default_factory=list)
    failure_modes: list[dict[str, Any]] = field(default_factory=list)
    state_mutations: list[dict[str, Any]] = field(default_factory=list)
    external_dependencies: list[dict[str, Any]] = field(default_factory=list)
    hazards: list[dict[str, Any]] = field(default_factory=list)
    data_ml_details: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {category: [] for category in DATA_ML_CATEGORIES}
    )


def extract_source_file(source: SourceFile, *, profile: str, zoom: str) -> dict[str, Any]:
    static_analysis = analyze_source(source)
    if source.language != "python":
        document = _unsupported_document(
            source,
            profile,
            zoom,
            "unsupported language",
            static_analysis=static_analysis,
        )
        validate_ir_document(document)
        return document

    try:
        tree = ast.parse(source.text, filename=source.rel_path)
    except SyntaxError as exc:
        line = exc.lineno or 1
        document = _unsupported_document(
            source,
            profile,
            zoom,
            f"python syntax error at line {line}: {exc.msg}",
            static_analysis=static_analysis,
        )
        validate_ir_document(document)
        return document

    units = [_module_unit(source, tree, profile, zoom)]
    for node, qualified_name, symbol_type in _iter_symbol_nodes(tree):
        units.append(_symbol_unit(source, node, qualified_name, symbol_type, profile, zoom))

    document = {
        "schema_version": SCHEMA_VERSION,
        "source_path": source.rel_path,
        "language": source.language,
        "profile": profile,
        "zoom": zoom,
        "static_analysis": static_analysis,
        "units": units,
        "unsupported_reasons": [],
    }
    validate_ir_document(document)
    return document


def _unsupported_document(
    source: SourceFile,
    profile: str,
    zoom: str,
    reason: str,
    *,
    static_analysis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_path": source.rel_path,
        "language": source.language,
        "profile": profile,
        "zoom": zoom,
        "static_analysis": static_analysis,
        "units": [
            {
                "unit_id": f"{source.rel_path}::unsupported",
                "source_spans": [make_span(source.rel_path, 1, max(1, len(source.lines)))],
                "language": source.language,
                "symbol_type": "unsupported_file",
                "name": source.rel_path,
                "qualified_name": source.rel_path,
                "algorithm": make_claim(
                    f"Semantic extraction is unsupported for this file: {reason}.",
                    [make_span(source.rel_path, 1, max(1, len(source.lines)))],
                    confidence=1.0,
                ),
                "control_flow": [],
                "reads": [],
                "writes": [],
                "calls": [],
                "returns": [],
                "side_effects": [],
                "failure_modes": [],
                "state_mutations": [],
                "external_dependencies": [],
                "data_ml_details": {category: [] for category in DATA_ML_CATEGORIES},
                "hazards": [],
                "uncertainty": [
                    make_claim(
                        reason,
                        [make_span(source.rel_path, 1, max(1, len(source.lines)))],
                        confidence=1.0,
                        kind="unsupported",
                    )
                ],
                "confidence": 0.0,
            }
        ],
        "unsupported_reasons": [{"reason": reason, "source_spans": [make_span(source.rel_path, 1, 1)]}],
    }


def _module_unit(source: SourceFile, tree: ast.Module, profile: str, zoom: str) -> dict[str, Any]:
    end_line = max(1, len(source.lines))
    span = make_span(source.rel_path, 1, end_line)
    facts = _collect_facts(source, tree, zoom)
    unit = _unit_payload(
        source=source,
        node=tree,
        unit_id=f"{source.rel_path}::module",
        source_spans=[span],
        symbol_type="module",
        name=source.rel_path,
        qualified_name=source.rel_path,
        algorithm_claim=(
            "Module-level semantic IR records imports, definitions, top-level effects, "
            "and static data/control facts visible in the source file."
        ),
        facts=facts,
        profile=profile,
        zoom=zoom,
        confidence=0.72,
    )
    return unit


def _symbol_unit(
    source: SourceFile,
    node: ast.AST,
    qualified_name: str,
    symbol_type: str,
    profile: str,
    zoom: str,
) -> dict[str, Any]:
    span = _node_span(source, node)
    facts = _collect_facts(source, node, zoom)
    noun = {
        "class": "Class",
        "function": "Function",
        "async_function": "Async function",
    }[symbol_type]
    unit = _unit_payload(
        source=source,
        node=node,
        unit_id=f"{source.rel_path}::{qualified_name}:{span['start_line']}-{span['end_line']}",
        source_spans=[span],
        symbol_type=symbol_type,
        name=getattr(node, "name", qualified_name),
        qualified_name=qualified_name,
        algorithm_claim=(
            f"{noun} `{qualified_name}` executes the source-backed behavior captured by "
            "the calls, control-flow facts, reads, writes, returns, effects, and hazards "
            "listed in this unit."
        ),
        facts=facts,
        profile=profile,
        zoom=zoom,
        confidence=0.75,
    )
    return unit


def _unit_payload(
    *,
    source: SourceFile,
    node: ast.AST,
    unit_id: str,
    source_spans: list[dict[str, Any]],
    symbol_type: str,
    name: str,
    qualified_name: str,
    algorithm_claim: str,
    facts: UnitFacts,
    profile: str,
    zoom: str,
    confidence: float,
) -> dict[str, Any]:
    visible_hazards = _zoom_claims(facts.hazards, zoom, category="hazards")
    uncertainty = list(visible_hazards)
    uncertainty.append(
        make_claim(
            "Static extraction did not execute code or inspect runtime dependency implementations.",
            source_spans,
            confidence=0.95,
            kind="static_limit",
        )
    )
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        uncertainty.append(
            make_claim(
                "Decorators may alter runtime behavior beyond the visible function or class body.",
                [_node_span(source, decorator) for decorator in decorators],
                confidence=0.85,
                kind="decorator_runtime_effect",
            )
        )

    visible_control_flow = _control_flow_for_zoom(facts.control_flow, source_spans, zoom)
    visible_reads = (
        facts.reads.claims("Reads `{name}` from local, closure, module, or imported scope.")
        if zoom in {"L3", "L4"}
        else []
    )
    visible_writes = (
        facts.writes.claims("Writes or binds `{name}` in the source unit.")
        if zoom in {"L3", "L4"}
        else []
    )
    visible_calls = (
        facts.calls.claims("Calls `{name}`.")
        if zoom in {"L2", "L3", "L4"}
        else []
    )
    visible_returns = _zoom_claims(
        facts.returns,
        zoom,
        category="returns",
        minimum_zoom="L2",
    )
    visible_side_effects = _zoom_claims(
        facts.side_effects,
        zoom,
        category="side_effects",
        minimum_zoom="L1",
    )
    visible_failure_modes = _zoom_claims(
        facts.failure_modes,
        zoom,
        category="failure_modes",
        minimum_zoom="L2",
    )
    visible_state_mutations = _zoom_claims(
        facts.state_mutations,
        zoom,
        category="state_mutations",
        minimum_zoom="L2",
    )
    visible_external_dependencies = _zoom_claims(
        facts.external_dependencies,
        zoom,
        category="external_dependencies",
        minimum_zoom="L1",
    )
    visible_data_ml_details = _data_ml_details_for_zoom(facts.data_ml_details, zoom)

    return {
        "unit_id": unit_id,
        "source_spans": source_spans,
        "language": source.language,
        "symbol_type": symbol_type,
        "name": name,
        "qualified_name": qualified_name,
        "algorithm": make_claim(algorithm_claim, source_spans, confidence=confidence),
        "control_flow": visible_control_flow,
        "reads": _ordered_claims(visible_reads, zoom, "reads"),
        "writes": _ordered_claims(visible_writes, zoom, "writes"),
        "calls": _ordered_claims(visible_calls, zoom, "calls"),
        "returns": visible_returns,
        "side_effects": visible_side_effects,
        "failure_modes": visible_failure_modes,
        "state_mutations": visible_state_mutations,
        "external_dependencies": visible_external_dependencies,
        "data_ml_details": visible_data_ml_details,
        "hazards": visible_hazards,
        "uncertainty": uncertainty,
        "confidence": confidence,
        "profile": profile,
        "zoom": zoom,
        "zoom_policy": _zoom_policy(zoom),
    }


def _collect_facts(source: SourceFile, node: ast.AST, zoom: str) -> UnitFacts:
    visitor = _FactVisitor(source, zoom)
    visitor.visit(node)
    return visitor.facts


def _control_flow_for_zoom(
    control_flow: list[dict[str, Any]],
    source_spans: list[dict[str, Any]],
    zoom: str,
) -> list[dict[str, Any]]:
    if zoom == "L1":
        if not control_flow:
            return []
        kinds = sorted({claim.get("kind", "unknown") for claim in control_flow})
        branch_count = sum(int(claim.get("branch_count", 1)) for claim in control_flow)
        return [
            make_claim(
                (
                    f"Unit contains {len(control_flow)} control-flow construct(s) "
                    f"across {branch_count} visible branch path(s)."
                ),
                _claim_spans(control_flow) or source_spans,
                confidence=0.82,
                kind="control_flow_summary",
                control_kinds=kinds,
                branch_count=branch_count,
            )
        ]
    return _ordered_claims(control_flow, zoom, "control_flow")


def _data_ml_details_for_zoom(
    data_ml_details: dict[str, list[dict[str, Any]]],
    zoom: str,
) -> dict[str, list[dict[str, Any]]]:
    allowed_by_zoom = {
        "L1": {"model_architecture", "training_loops", "checkpointing"},
        "L2": {
            "losses",
            "model_architecture",
            "training_loops",
            "optimizer_scheduler",
            "metrics",
            "checkpointing",
        },
        "L3": set(DATA_ML_CATEGORIES),
        "L4": set(DATA_ML_CATEGORIES),
    }
    allowed = allowed_by_zoom[zoom]
    return {
        category: _ordered_claims(claims if category in allowed else [], zoom, category)
        for category, claims in data_ml_details.items()
    }


def _zoom_claims(
    claims: list[dict[str, Any]],
    zoom: str,
    *,
    category: str,
    minimum_zoom: str = "L3",
) -> list[dict[str, Any]]:
    if _zoom_rank(zoom) < _zoom_rank(minimum_zoom):
        return []
    return _ordered_claims(claims, zoom, category)


def _ordered_claims(
    claims: list[dict[str, Any]],
    zoom: str,
    category: str,
) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(claim) for claim in claims),
        key=lambda claim: (
            claim["source_spans"][0]["start_line"],
            claim["source_spans"][0]["end_line"],
            claim.get("name") or claim.get("call") or claim.get("kind") or claim["claim"],
        ),
    )
    if zoom in {"L3", "L4"}:
        for index, claim in enumerate(ordered, start=1):
            claim.setdefault("order", index)
            claim.setdefault("order_scope", category)
    return ordered


def _claim_spans(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for claim in claims:
        for span in claim.get("source_spans", []):
            if span not in spans:
                spans.append(span)
    return spans


def _zoom_rank(zoom: str) -> int:
    return {"L1": 1, "L2": 2, "L3": 3, "L4": 4}[zoom]


def _zoom_policy(zoom: str) -> dict[str, Any]:
    policies = {
        "L1": {
            "intent": "repo_module_intent_and_major_flows",
            "included": ["algorithm", "control_flow_summary", "side_effects", "external_dependencies"],
            "omitted": ["reads", "writes", "calls", "returns", "state_mutation_order"],
        },
        "L2": {
            "intent": "function_class_behavior_and_side_effects",
            "included": ["calls", "returns", "side_effects", "failure_modes", "state_mutations"],
            "omitted": ["reads", "writes", "exact_order_annotations"],
        },
        "L3": {
            "intent": "branch_predicates_data_dependencies_and_mutation_order",
            "included": ["reads", "writes", "branch_predicates", "order_annotations"],
            "omitted": [],
        },
        "L4": {
            "intent": "implementation_sensitive_details_and_data_ml_mechanics",
            "included": ["all_static_facts", "data_ml_details", "order_annotations"],
            "omitted": [],
        },
    }
    return policies[zoom]


def _iter_symbol_nodes(tree: ast.Module) -> Iterable[tuple[ast.AST, str, str]]:
    def walk_body(body: list[ast.stmt], prefix: str = "") -> Iterable[tuple[ast.AST, str, str]]:
        for child in body:
            if isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                yield child, qualified, "class"
                yield from walk_body(child.body, qualified)
            elif isinstance(child, ast.AsyncFunctionDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                yield child, qualified, "async_function"
                yield from walk_body(child.body, qualified)
            elif isinstance(child, ast.FunctionDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                yield child, qualified, "function"
                yield from walk_body(child.body, qualified)

    yield from walk_body(tree.body)


class _FactVisitor(ast.NodeVisitor):
    def __init__(self, source: SourceFile, zoom: str) -> None:
        self.source = source
        self.zoom = zoom
        self.facts = UnitFacts()

    def visit_Name(self, node: ast.Name) -> None:
        span = _node_span(self.source, node)
        if isinstance(node.ctx, ast.Load):
            self.facts.reads.add(node.id, span)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.facts.writes.add(node.id, span)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        span = _node_span(self.source, node)
        self.facts.writes.add(node.arg, span)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        span = _node_span(self.source, node)
        attribute_name = _expr_name(node)
        if isinstance(node.ctx, ast.Load):
            self.facts.reads.add(attribute_name, span)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.facts.writes.add(attribute_name, span)
            self.facts.state_mutations.append(
                make_claim(
                    f"Mutates attribute `{attribute_name}`.",
                    [span],
                    confidence=0.9,
                    kind="attribute_write",
                    target=attribute_name,
                )
            )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            span = _node_span(self.source, node)
            target = _safe_unparse(node)
            self.facts.state_mutations.append(
                make_claim(
                    f"Mutates indexed or sliced target `{target}`.",
                    [span],
                    confidence=0.86,
                    kind="subscript_write",
                    target=target,
                )
            )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        span = _node_span(self.source, node)
        target = _safe_unparse(node.target)
        self.facts.state_mutations.append(
            make_claim(
                f"Updates `{target}` with augmented assignment `{type(node.op).__name__}`.",
                [span],
                confidence=0.88,
                kind="augmented_assignment",
                target=target,
            )
        )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        span = _node_span(self.source, node)
        call_name = _call_name(node)
        self.facts.calls.add(call_name, span)
        self._maybe_side_effect(call_name, span)
        self._maybe_state_mutation(call_name, node, span)
        self._maybe_hazard(call_name, span)
        self._maybe_data_ml_call(call_name, node, span)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        span = _node_span(self.source, node)
        detail = "None" if node.value is None else _safe_unparse(node.value)
        self.facts.returns.append(
            make_claim(
                f"Returns `{detail}`.",
                [span],
                confidence=0.9,
                value=detail,
            )
        )
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:
        span = _node_span(self.source, node)
        detail = "None" if node.value is None else _safe_unparse(node.value)
        self.facts.returns.append(
            make_claim(
                f"Yields `{detail}`.",
                [span],
                confidence=0.86,
                value=detail,
            )
        )
        self.facts.hazards.append(
            make_claim(
                "Generator yield splits execution across caller-driven resume points.",
                [span],
                confidence=0.9,
                kind="generator_boundary",
            )
        )
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        span = _node_span(self.source, node)
        detail = _safe_unparse(node.value)
        self.facts.returns.append(
            make_claim(
                f"Yields from `{detail}`.",
                [span],
                confidence=0.86,
                value=detail,
            )
        )
        self.facts.hazards.append(
            make_claim(
                "Delegated generator yield depends on another iterator's runtime behavior.",
                [span],
                confidence=0.88,
                kind="generator_boundary",
            )
        )
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        span = _node_span(self.source, node)
        detail = "re-raise" if node.exc is None else _safe_unparse(node.exc)
        self.facts.failure_modes.append(
            make_claim(
                f"Raises `{detail}`.",
                [span],
                confidence=0.9,
                kind="raise",
                exception=detail,
            )
        )
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        span = _node_span(self.source, node)
        predicate = _safe_unparse(node.test)
        self.facts.failure_modes.append(
            make_claim(
                f"Assertion can fail when `{predicate}` is false.",
                [span],
                confidence=0.86,
                kind="assert",
                predicate=predicate,
            )
        )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        span = _node_span(self.source, node)
        for alias in node.names:
            name = alias.asname or alias.name
            self.facts.writes.add(name, span)
            self.facts.external_dependencies.append(
                make_claim(
                    f"Imports external module `{alias.name}`.",
                    [span],
                    confidence=0.96,
                    module=alias.name,
                    alias=alias.asname,
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        span = _node_span(self.source, node)
        module = "." * node.level + (node.module or "")
        for alias in node.names:
            name = alias.asname or alias.name
            self.facts.writes.add(name, span)
            self.facts.external_dependencies.append(
                make_claim(
                    f"Imports `{alias.name}` from `{module}`.",
                    [span],
                    confidence=0.96,
                    module=module,
                    imported=alias.name,
                    alias=alias.asname,
                )
            )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self._control("if", node, predicate=_safe_unparse(node.test), branches=1 + bool(node.orelse))
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        detail = f"{_safe_unparse(node.target)} in {_safe_unparse(node.iter)}"
        self._control("for", node, predicate=detail, branches=1 + bool(node.orelse))
        self._maybe_training_loop(node, detail)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        detail = f"{_safe_unparse(node.target)} in {_safe_unparse(node.iter)}"
        self._control("async_for", node, predicate=detail, branches=1 + bool(node.orelse))
        self.facts.hazards.append(
            make_claim(
                "Async iteration crosses an awaitable boundary.",
                [_node_span(self.source, node)],
                confidence=0.87,
                kind="async_boundary",
            )
        )
        self._maybe_training_loop(node, detail)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._control("while", node, predicate=_safe_unparse(node.test), branches=1 + bool(node.orelse))
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        handler_names = [
            "bare"
            if handler.type is None
            else _safe_unparse(handler.type)
            for handler in node.handlers
        ]
        detail = ", ".join(handler_names) if handler_names else "no except handlers"
        self._control("try", node, predicate=detail, branches=1 + len(node.handlers) + bool(node.orelse))
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        detail = ", ".join(_safe_unparse(item.context_expr) for item in node.items)
        self._control("with", node, predicate=detail, branches=1)
        self.facts.side_effects.append(
            make_claim(
                f"Enters context manager `{detail}` which may acquire or release resources.",
                [_node_span(self.source, node)],
                confidence=0.74,
                kind="context_manager",
                target=detail,
            )
        )
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        detail = ", ".join(_safe_unparse(item.context_expr) for item in node.items)
        self._control("async_with", node, predicate=detail, branches=1)
        self.facts.hazards.append(
            make_claim(
                f"Async context manager `{detail}` crosses awaitable resource boundaries.",
                [_node_span(self.source, node)],
                confidence=0.84,
                kind="async_boundary",
            )
        )
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self._control(
            "match",
            node,
            predicate=_safe_unparse(node.subject),
            branches=max(1, len(node.cases)),
        )
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        span = _node_span(self.source, node)
        names = ", ".join(node.names)
        self.facts.hazards.append(
            make_claim(
                f"Uses global declaration for `{names}`.",
                [span],
                confidence=0.94,
                kind="global_state",
                names=node.names,
            )
        )
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        span = _node_span(self.source, node)
        names = ", ".join(node.names)
        self.facts.hazards.append(
            make_claim(
                f"Uses nonlocal declaration for `{names}`.",
                [span],
                confidence=0.94,
                kind="closure_state",
                names=node.names,
            )
        )
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> None:
        span = _node_span(self.source, node)
        self.facts.hazards.append(
            make_claim(
                f"Awaits `{_safe_unparse(node.value)}`.",
                [span],
                confidence=0.9,
                kind="async_boundary",
            )
        )
        self.generic_visit(node)

    def _control(self, kind: str, node: ast.AST, *, predicate: str, branches: int) -> None:
        span = _node_span(self.source, node)
        if self.zoom in {"L3", "L4"}:
            claim = f"Control flow `{kind}` uses predicate or target `{predicate}`."
        else:
            claim = f"Control flow `{kind}` introduces {branches} visible branch path(s)."
        self.facts.control_flow.append(
            make_claim(
                claim,
                [span],
                confidence=0.88,
                kind=kind,
                predicate=predicate,
                branch_count=branches,
            )
        )

    def _maybe_side_effect(self, call_name: str, span: dict[str, Any]) -> None:
        if call_name in SIDE_EFFECT_CALLS or call_name.endswith((".write_text", ".write_bytes")):
            self.facts.side_effects.append(
                make_claim(
                    f"Call `{call_name}` may perform I/O, process, persistence, or user-visible effects.",
                    [span],
                    confidence=0.78,
                    kind="call_side_effect",
                    call=call_name,
                )
            )

    def _maybe_state_mutation(self, call_name: str, node: ast.Call, span: dict[str, Any]) -> None:
        method = call_name.rsplit(".", 1)[-1]
        if method not in MUTATING_METHODS:
            return
        target = _safe_unparse(node.func)
        self.facts.state_mutations.append(
            make_claim(
                f"Call `{call_name}` may mutate receiver `{target}`.",
                [span],
                confidence=0.78,
                kind="mutating_method_call",
                call=call_name,
                target=target,
            )
        )

    def _maybe_hazard(self, call_name: str, span: dict[str, Any]) -> None:
        for hazard_call, description in HAZARD_CALLS.items():
            if call_name == hazard_call or call_name.startswith(f"{hazard_call}."):
                self.facts.hazards.append(
                    make_claim(
                        f"Call `{call_name}` introduces {description}.",
                        [span],
                        confidence=0.83,
                        kind=description.replace(" ", "_"),
                        call=call_name,
                    )
                )

    def _maybe_training_loop(self, node: ast.For | ast.AsyncFor, detail: str) -> None:
        lowered = detail.lower()
        if any(token in lowered for token in ("loader", "dataloader", "batch", "epoch")):
            self.facts.data_ml_details["training_loops"].append(
                make_claim(
                    f"Loop `{detail}` looks like a training or evaluation data iteration.",
                    [_node_span(self.source, node)],
                    confidence=0.72,
                    kind="data_iteration_loop",
                    loop=detail,
                )
            )

    def _maybe_data_ml_call(self, call_name: str, node: ast.Call, span: dict[str, Any]) -> None:
        lowered = call_name.lower()
        self._maybe_category(
            "losses",
            call_name,
            span,
            any(token in lowered for token in LOSS_TOKENS),
            "Call `{name}` participates in loss computation or loss handling.",
        )
        self._maybe_category(
            "model_architecture",
            call_name,
            span,
            any(token.lower() in lowered for token in ARCHITECTURE_TOKENS),
            "Call `{name}` constructs or applies a model architecture component.",
        )
        self._maybe_category(
            "tensor_shapes",
            call_name,
            span,
            any(token in lowered.split(".") or lowered.endswith(f".{token}") for token in TENSOR_TOKENS),
            "Call `{name}` changes or inspects tensor shape, device, dtype, or graph attachment.",
        )
        self._maybe_category(
            "optimizer_scheduler",
            call_name,
            span,
            any(token.lower() in lowered for token in OPTIMIZER_TOKENS),
            "Call `{name}` affects optimizer, scheduler, or gradient update behavior.",
        )
        self._maybe_category(
            "metrics",
            call_name,
            span,
            any(token in lowered for token in METRIC_TOKENS),
            "Call `{name}` computes or records a metric.",
        )
        self._maybe_category(
            "checkpointing",
            call_name,
            span,
            any(token.lower() in lowered for token in CHECKPOINT_TOKENS),
            "Call `{name}` saves, loads, or exposes checkpoint/model state.",
        )
        if call_name.endswith(".backward"):
            self.facts.data_ml_details["training_loops"].append(
                make_claim(
                    "Backward pass computes gradients from the current loss graph.",
                    [span],
                    confidence=0.9,
                    kind="backward_pass",
                    call=call_name,
                )
            )
        if call_name.endswith(".step"):
            self.facts.data_ml_details["training_loops"].append(
                make_claim(
                    f"Call `{call_name}` advances optimizer, scheduler, or iterative training state.",
                    [span],
                    confidence=0.82,
                    kind="training_step",
                    call=call_name,
                )
            )
        if call_name.endswith(".zero_grad"):
            self.facts.data_ml_details["training_loops"].append(
                make_claim(
                    f"Call `{call_name}` clears accumulated gradients before a training update.",
                    [span],
                    confidence=0.88,
                    kind="gradient_reset",
                    call=call_name,
                )
            )

        for arg in node.args:
            arg_text = _safe_unparse(arg)
            if "shape" in arg_text.lower() or "size" in arg_text.lower():
                self.facts.data_ml_details["tensor_shapes"].append(
                    make_claim(
                        f"Argument `{arg_text}` carries explicit shape or size information.",
                        [_node_span(self.source, arg)],
                        confidence=0.72,
                        kind="shape_argument",
                    )
                )

    def _maybe_category(
        self,
        category: str,
        call_name: str,
        span: dict[str, Any],
        condition: bool,
        template: str,
    ) -> None:
        if condition:
            self.facts.data_ml_details[category].append(
                make_claim(
                    template.format(name=call_name),
                    [span],
                    confidence=0.78,
                    kind=category,
                    call=call_name,
                )
            )


def _node_span(source: SourceFile, node: ast.AST) -> dict[str, Any]:
    start = getattr(node, "lineno", 1) or 1
    end = getattr(node, "end_lineno", start) or start
    return make_span(source.rel_path, start, end)


def _call_name(node: ast.Call) -> str:
    return _expr_name(node.func)


def _expr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _expr_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    if isinstance(node, ast.Call):
        return _call_name(node)
    if isinstance(node, ast.Subscript):
        return _expr_name(node.value)
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return _safe_unparse(node)


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__
