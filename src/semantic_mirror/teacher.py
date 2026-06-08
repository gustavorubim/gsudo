"""Teacher-model request export and response ingest workflows."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from semantic_mirror.rewards import score_document
from semantic_mirror.schema import (
    DATA_ML_DETAIL_CATEGORIES,
    SchemaValidationError,
    validate_unit,
)

TEACHER_VERSION = "0.1.0"
DEFAULT_TEACHER_MODELS = ("frontier-generator-a", "frontier-generator-b", "frontier-generator-c")
DEFAULT_OPENAI_MODEL = "gpt-4.1"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
SUPPORTED_TEACHER_PROVIDERS = ("openai", "anthropic", "gemini")
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
HttpTransport = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]


def export_teacher_requests(
    dataset_path: Path | str,
    out_path: Path | str,
    *,
    candidates_per_unit: int = 3,
    models: list[str] | None = None,
    max_units: int | None = None,
) -> dict[str, Any]:
    dataset = Path(dataset_path).resolve()
    out = Path(out_path).resolve()
    dataset_manifest = _read_json(dataset / "manifest.json")
    silver = _read_jsonl(dataset / dataset_manifest["files"]["silver"])
    records = silver[:max_units] if max_units is not None else silver
    model_slots = models or list(DEFAULT_TEACHER_MODELS)
    if candidates_per_unit < 1:
        raise ValueError("candidates_per_unit must be at least 1")

    requests: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        for candidate_index in range(candidates_per_unit):
            model = model_slots[candidate_index % len(model_slots)]
            requests.append(_candidate_request(record, record_index, candidate_index, model))

    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "candidate_requests.jsonl", requests)
    (out / "README.md").write_text(_teacher_readme(), encoding="utf-8")
    manifest = {
        "mode": "teacher_export",
        "teacher_version": TEACHER_VERSION,
        "dataset": str(dataset),
        "generated_at": _now(),
        "request_counts": {
            "candidate_generation": len(requests),
            "source_units": len(records),
            "candidates_per_unit": candidates_per_unit,
        },
        "models": model_slots,
        "files": {
            "candidate_requests": "candidate_requests.jsonl",
        },
        "response_contract": {
            "required": ["request_id", "sir_unit"],
            "optional": ["model", "rationale", "uncertainty_notes"],
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def ingest_teacher_responses(
    dataset_path: Path | str,
    requests_path: Path | str,
    responses_path: Path | str,
    out_path: Path | str,
) -> dict[str, Any]:
    dataset = Path(dataset_path).resolve()
    out = Path(out_path).resolve()
    dataset_manifest = _read_json(dataset / "manifest.json")
    requests = _read_jsonl(Path(requests_path))
    responses = _read_jsonl(Path(responses_path))
    request_by_id = {request["request_id"]: request for request in requests}
    positive_by_id = {
        record["record_id"]: record for record in _read_jsonl(dataset / dataset_manifest["files"]["silver"])
    }
    positive_by_id.update(
        {
            record["record_id"]: {**record, "split": "gold"}
            for record in _read_jsonl(dataset / dataset_manifest["files"]["gold"])
        }
    )
    repo_path = Path(dataset_manifest["repo"]).resolve()

    candidate_results: list[dict[str, Any]] = []
    eval_candidates: list[dict[str, Any]] = []
    critic_requests: list[dict[str, Any]] = []
    preference_pairs: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []

    for response_index, response in enumerate(responses):
        result = _candidate_result(
            response=response,
            request=request_by_id.get(response.get("request_id")),
            positive_by_id=positive_by_id,
            default_repo_path=repo_path,
            response_index=response_index,
        )
        candidate_results.append(result)
        eval_candidates.append(_eval_candidate_record(result, len(eval_candidates)))
        if result["auto_reject"] or result["needs_human_review"]:
            critic_requests.append(_critic_request(result, len(critic_requests)))
            review_queue.append(_review_item(result, len(review_queue) + 1))
        preference = _preference_pair(result)
        if preference is not None:
            preference_pairs.append(preference)

    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "candidate_results.jsonl", candidate_results)
    _write_jsonl(out / "teacher_candidates.jsonl", eval_candidates)
    _write_jsonl(out / "critic_requests.jsonl", critic_requests)
    _write_jsonl(out / "teacher_preference_pairs.jsonl", preference_pairs)
    _write_jsonl(out / "teacher_review_queue.jsonl", review_queue)
    manifest = {
        "mode": "teacher_ingest",
        "teacher_version": TEACHER_VERSION,
        "dataset": str(dataset),
        "requests": str(Path(requests_path).resolve()),
        "responses": str(Path(responses_path).resolve()),
        "generated_at": _now(),
        "counts": {
            "responses": len(responses),
            "candidate_results": len(candidate_results),
            "teacher_candidates": len(eval_candidates),
            "accepted_candidates": sum(1 for result in candidate_results if result["accepted"]),
            "auto_rejected_candidates": sum(1 for result in candidate_results if result["auto_reject"]),
            "critic_requests": len(critic_requests),
            "preference_pairs": len(preference_pairs),
            "review_queue": len(review_queue),
        },
        "files": {
            "candidate_results": "candidate_results.jsonl",
            "teacher_candidates": "teacher_candidates.jsonl",
            "critic_requests": "critic_requests.jsonl",
            "teacher_preference_pairs": "teacher_preference_pairs.jsonl",
            "teacher_review_queue": "teacher_review_queue.jsonl",
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_critic_requests(
    requests_path: Path | str,
    out_path: Path | str,
    *,
    provider: str = "openai",
    model: str | None = None,
    env_file: Path | str | None = None,
    max_requests: int | None = None,
    max_input_chars: int | None = 24000,
    max_output_tokens: int = 2048,
    transport: HttpTransport | None = None,
) -> dict[str, Any]:
    """Call an API provider for exported critic requests and write critique JSONL."""

    if provider not in SUPPORTED_TEACHER_PROVIDERS:
        raise ValueError(f"provider must be one of {', '.join(SUPPORTED_TEACHER_PROVIDERS)}")
    env = _load_env(env_file)
    api_key = _teacher_api_key(provider, env)
    selected_model = _teacher_model(provider, model, env)
    requests = _read_jsonl(Path(requests_path))
    if max_requests is not None:
        requests = requests[:max_requests]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")
    error_path = out.with_suffix(".errors.jsonl")
    if error_path.exists():
        error_path.unlink()

    responses: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for request in requests:
        try:
            response = _run_provider_critic_request(
                request,
                provider=provider,
                api_key=api_key,
                model=selected_model,
                max_input_chars=max_input_chars,
                max_output_tokens=max_output_tokens,
                transport=transport or _http_post_json,
            )
            responses.append(response)
            _append_jsonl(out, response)
        except Exception as exc:
            error = {
                "request_id": request["request_id"],
                "candidate_result_id": request.get("candidate_result_id"),
                "model": selected_model,
                "error": str(exc),
            }
            errors.append(error)
            _append_jsonl(error_path, error)

    return {
        "mode": "critic_run",
        "provider": provider,
        "model": selected_model,
        "requests": str(Path(requests_path).resolve()),
        "responses": str(out.resolve()),
        "generated_at": _now(),
        "counts": {
            "requested": len(requests),
            "responses": len(responses),
            "errors": len(errors),
        },
        "limits": {
            "max_input_chars": max_input_chars,
            "max_output_tokens": max_output_tokens,
        },
        "files": {
            "responses": str(out),
            "errors": str(error_path) if errors else None,
        },
    }


def ingest_critic_responses(
    teacher_results_path: Path | str,
    responses_path: Path | str,
    out_path: Path | str,
) -> dict[str, Any]:
    """Ingest critic model responses into structured error labels for curation."""

    teacher_results = _resolve_teacher_results_root(Path(teacher_results_path))
    out = Path(out_path).resolve()
    candidate_results = _read_jsonl(teacher_results / "candidate_results.jsonl")
    critic_requests = _read_jsonl(teacher_results / "critic_requests.jsonl")
    responses = _read_jsonl(Path(responses_path))
    candidate_by_id = {item["result_id"]: item for item in candidate_results}
    request_by_id = {item["request_id"]: item for item in critic_requests}

    labels: list[dict[str, Any]] = []
    invalid_responses: list[dict[str, Any]] = []
    for index, response in enumerate(responses):
        request = request_by_id.get(response.get("request_id"))
        if request is None:
            invalid_responses.append(
                {
                    "response_index": index,
                    "request_id": response.get("request_id"),
                    "error": "critic response request_id does not match a critic request",
                }
            )
            continue
        candidate = candidate_by_id.get(request["candidate_result_id"])
        if candidate is None:
            invalid_responses.append(
                {
                    "response_index": index,
                    "request_id": response.get("request_id"),
                    "candidate_result_id": request["candidate_result_id"],
                    "error": "critic request candidate_result_id does not match a candidate result",
                }
            )
            continue
        labels.append(_critic_label_record(response, request, candidate, len(labels)))

    labeled_review_queue = _labeled_review_queue(teacher_results, labels)
    summary = _critic_label_summary(labels, invalid_responses)

    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "critic_labels.jsonl", labels)
    _write_jsonl(out / "critic_invalid_responses.jsonl", invalid_responses)
    _write_jsonl(out / "critic_review_queue.jsonl", labeled_review_queue)
    (out / "critic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "mode": "critic_ingest",
        "teacher_version": TEACHER_VERSION,
        "teacher_results": str(teacher_results),
        "responses": str(Path(responses_path).resolve()),
        "generated_at": _now(),
        "counts": {
            "responses": len(responses),
            "labeled_candidates": len(labels),
            "invalid_responses": len(invalid_responses),
            "error_labels": sum(len(item["error_labels"]) for item in labels),
            "review_queue": len(labeled_review_queue),
        },
        "files": {
            "critic_labels": "critic_labels.jsonl",
            "critic_invalid_responses": "critic_invalid_responses.jsonl",
            "critic_review_queue": "critic_review_queue.jsonl",
            "critic_summary": "critic_summary.json",
        },
    }
    (out / "critic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_teacher_requests(
    requests_path: Path | str,
    out_path: Path | str,
    *,
    provider: str = "openai",
    model: str | None = None,
    env_file: Path | str | None = None,
    max_requests: int | None = None,
    max_input_chars: int | None = 24000,
    max_output_tokens: int = 4096,
    transport: HttpTransport | None = None,
) -> dict[str, Any]:
    if provider not in SUPPORTED_TEACHER_PROVIDERS:
        raise ValueError(f"provider must be one of {', '.join(SUPPORTED_TEACHER_PROVIDERS)}")
    env = _load_env(env_file)
    api_key = _teacher_api_key(provider, env)
    selected_model = _teacher_model(provider, model, env)
    requests = _read_jsonl(Path(requests_path))
    if max_requests is not None:
        requests = requests[:max_requests]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")
    error_path = out.with_suffix(".errors.jsonl")
    if error_path.exists():
        error_path.unlink()

    responses: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for request in requests:
        try:
            response = _run_provider_request(
                request,
                provider=provider,
                api_key=api_key,
                model=selected_model,
                max_input_chars=max_input_chars,
                max_output_tokens=max_output_tokens,
                transport=transport or _http_post_json,
            )
            responses.append(response)
            _append_jsonl(out, response)
        except Exception as exc:
            error = {
                "request_id": request["request_id"],
                "model": selected_model,
                "error": str(exc),
            }
            errors.append(error)
            _append_jsonl(error_path, error)

    return {
        "mode": "teacher_run",
        "provider": provider,
        "model": selected_model,
        "requests": str(Path(requests_path).resolve()),
        "responses": str(out.resolve()),
        "generated_at": _now(),
        "counts": {
            "requested": len(requests),
            "responses": len(responses),
            "errors": len(errors),
        },
        "limits": {
            "max_input_chars": max_input_chars,
            "max_output_tokens": max_output_tokens,
        },
        "files": {
            "responses": str(out),
            "errors": str(error_path) if errors else None,
        },
    }


def run_teacher_pipeline(
    dataset_path: Path | str,
    out_path: Path | str,
    *,
    providers: list[str] | None = None,
    models: list[str | None] | None = None,
    candidates_per_provider: int = 1,
    max_units: int | None = None,
    env_file: Path | str | None = None,
    max_input_chars: int | None = 24000,
    max_output_tokens: int = 4096,
    transport: HttpTransport | None = None,
) -> dict[str, Any]:
    """Export, run, combine, and ingest teacher candidates across providers."""

    selected_providers = providers or ["openai"]
    for provider in selected_providers:
        if provider not in SUPPORTED_TEACHER_PROVIDERS:
            raise ValueError(f"provider must be one of {', '.join(SUPPORTED_TEACHER_PROVIDERS)}")
    if candidates_per_provider < 1:
        raise ValueError("candidates_per_provider must be at least 1")
    if models is not None and len(models) > len(selected_providers):
        raise ValueError("models may not contain more entries than providers")

    dataset = Path(dataset_path).resolve()
    out = Path(out_path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = _load_env(env_file)
    all_requests: list[dict[str, Any]] = []
    all_responses: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    provider_runs: list[dict[str, Any]] = []

    requests_root = out / "provider_requests"
    responses_root = out / "provider_responses"
    requests_root.mkdir(parents=True, exist_ok=True)
    responses_root.mkdir(parents=True, exist_ok=True)

    for provider_index, provider in enumerate(selected_providers):
        model = _model_for_pipeline_provider(provider, provider_index, models, env)
        provider_request_dir = requests_root / provider
        export_manifest = export_teacher_requests(
            dataset,
            provider_request_dir,
            candidates_per_unit=candidates_per_provider,
            models=[model],
            max_units=max_units,
        )
        provider_requests_path = provider_request_dir / export_manifest["files"]["candidate_requests"]
        provider_requests = _read_jsonl(provider_requests_path)
        provider_requests = [
            _pipeline_provider_request(request, provider=provider, model=model)
            for request in provider_requests
        ]
        _write_jsonl(provider_requests_path, provider_requests)
        all_requests.extend(provider_requests)

        provider_response_path = responses_root / f"{provider}.responses.jsonl"
        try:
            run_manifest = run_teacher_requests(
                provider_requests_path,
                provider_response_path,
                provider=provider,
                model=model,
                env_file=env_file,
                max_input_chars=max_input_chars,
                max_output_tokens=max_output_tokens,
                transport=transport,
            )
        except Exception as exc:
            run_manifest = {
                "mode": "teacher_run",
                "provider": provider,
                "model": model,
                "requests": str(provider_requests_path.resolve()),
                "responses": str(provider_response_path.resolve()),
                "generated_at": _now(),
                "counts": {
                    "requested": len(provider_requests),
                    "responses": 0,
                    "errors": len(provider_requests) or 1,
                },
                "files": {
                    "responses": str(provider_response_path),
                    "errors": str(provider_response_path.with_suffix(".errors.jsonl")),
                },
                "error": str(exc),
            }
            _write_jsonl(
                provider_response_path.with_suffix(".errors.jsonl"),
                [
                    {
                        "provider": provider,
                        "model": model,
                        "request_id": request.get("request_id"),
                        "error": str(exc),
                    }
                    for request in provider_requests
                ]
                or [{"provider": provider, "model": model, "request_id": None, "error": str(exc)}],
            )

        provider_responses = _read_jsonl(provider_response_path) if provider_response_path.exists() else []
        provider_error_path = provider_response_path.with_suffix(".errors.jsonl")
        provider_errors = _read_jsonl(provider_error_path) if provider_error_path.exists() else []
        for error in provider_errors:
            error.setdefault("provider", provider)
            error.setdefault("model", model)
        all_responses.extend(provider_responses)
        all_errors.extend(provider_errors)
        provider_runs.append(
            {
                "provider": provider,
                "model": model,
                "export": export_manifest,
                "run": run_manifest,
                "counts": {
                    "requests": len(provider_requests),
                    "responses": len(provider_responses),
                    "errors": len(provider_errors),
                },
            }
        )

    combined_requests_path = out / "candidate_requests.jsonl"
    combined_responses_path = out / "teacher_responses.jsonl"
    combined_errors_path = out / "teacher_errors.jsonl"
    _write_jsonl(combined_requests_path, all_requests)
    _write_jsonl(combined_responses_path, all_responses)
    _write_jsonl(combined_errors_path, all_errors)
    ingest_manifest = ingest_teacher_responses(
        dataset,
        combined_requests_path,
        combined_responses_path,
        out / "teacher_results",
    )
    manifest = {
        "mode": "teacher_pipeline",
        "teacher_version": TEACHER_VERSION,
        "dataset": str(dataset),
        "generated_at": _now(),
        "providers": selected_providers,
        "models": [run["model"] for run in provider_runs],
        "limits": {
            "candidates_per_provider": candidates_per_provider,
            "max_units": max_units,
            "max_input_chars": max_input_chars,
            "max_output_tokens": max_output_tokens,
        },
        "counts": {
            "providers": len(selected_providers),
            "requests": len(all_requests),
            "responses": len(all_responses),
            "errors": len(all_errors),
            "accepted_candidates": ingest_manifest["counts"]["accepted_candidates"],
            "auto_rejected_candidates": ingest_manifest["counts"]["auto_rejected_candidates"],
            "critic_requests": ingest_manifest["counts"]["critic_requests"],
            "preference_pairs": ingest_manifest["counts"]["preference_pairs"],
            "review_queue": ingest_manifest["counts"]["review_queue"],
        },
        "provider_runs": provider_runs,
        "files": {
            "candidate_requests": str(combined_requests_path),
            "teacher_responses": str(combined_responses_path),
            "teacher_errors": str(combined_errors_path),
            "teacher_results_manifest": str(out / "teacher_results" / "manifest.json"),
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _candidate_request(
    record: dict[str, Any],
    record_index: int,
    candidate_index: int,
    model: str,
) -> dict[str, Any]:
    return {
        "request_id": f"teacher-candidate-{record_index}-{candidate_index}-{record['record_id']}",
        "kind": "candidate_generation",
        "model": model,
        "dataset_record_id": record["record_id"],
        "source_path": record["source_path"],
        "unit_id": record["unit_id"],
        "qualified_name": record["qualified_name"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "Generate faithful Semantic IR JSON for a single code unit. Use the static "
                    "facts as constraints, not as optional hints. Do not invent behavior. Every "
                    "claim must include source_spans from the provided code slice. Return exactly "
                    "one JSON object with top-level request_id and sir_unit. The sir_unit must use "
                    "the exact Semantic IR field names: source_spans plural array, symbol_type not "
                    "kind, data_ml_details not data_ml, and confidence on the unit and every claim."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "profile": record["profile"],
                        "zoom": record["zoom"],
                        "code_slice": record["code_slice"],
                        "static_facts": record["static_facts"],
                        "static_analysis": record.get("static_analysis", {}),
                        "response_contract": _sir_unit_response_contract(record),
                    },
                    indent=2,
                    sort_keys=True,
                ),
            },
        ],
    }


def _sir_unit_response_contract(record: dict[str, Any]) -> dict[str, Any]:
    target_unit = record.get("target", {}).get("sir_unit", {})
    language = target_unit.get("language", "python")
    symbol_type = record.get("symbol_type") or target_unit.get("symbol_type", "function")
    unit_name = target_unit.get("name") or record["qualified_name"].rsplit(".", 1)[-1]
    return {
        "top_level_required": ["request_id", "sir_unit"],
        "sir_unit_required": [
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
        ],
        "claim_shape": {
            "claim": "source-backed natural language claim",
            "source_spans": [{"path": record["source_path"], "start_line": 1, "end_line": 1}],
            "confidence": 0.0,
        },
        "data_ml_details_required_categories": [
            "losses",
            "model_architecture",
            "tensor_shapes",
            "training_loops",
            "optimizer_scheduler",
            "metrics",
            "checkpointing",
        ],
        "field_name_rules": [
            "Use source_spans, never source_span.",
            "Use symbol_type, never kind for the unit type.",
            "Use data_ml_details, never data_ml.",
            "Every item in control_flow, reads, writes, calls, returns, side_effects, "
            "failure_modes, state_mutations, external_dependencies, hazards, uncertainty, "
            "and data_ml_details categories must be a claim object, never a bare string.",
            "Keep empty categories as empty arrays instead of omitting them.",
            "Use the provided unit_id, source_path, profile, zoom, and qualified_name exactly.",
        ],
        "sir_unit_skeleton": {
            "unit_id": record["unit_id"],
            "source_spans": [{"path": record["source_path"], "start_line": 1, "end_line": 1}],
            "language": language,
            "symbol_type": symbol_type,
            "name": unit_name,
            "qualified_name": record["qualified_name"],
            "algorithm": "claim object",
            "control_flow": [],
            "reads": [],
            "writes": [],
            "calls": [],
            "returns": [],
            "side_effects": [],
            "failure_modes": [],
            "state_mutations": [],
            "external_dependencies": [],
            "data_ml_details": {
                "losses": [],
                "model_architecture": [],
                "tensor_shapes": [],
                "training_loops": [],
                "optimizer_scheduler": [],
                "metrics": [],
                "checkpointing": [],
            },
            "hazards": [],
            "uncertainty": [],
            "confidence": 0.0,
        },
        "rationale": "short explanation of difficult facts",
    }


def _model_for_pipeline_provider(
    provider: str,
    provider_index: int,
    models: list[str | None] | None,
    env: dict[str, str],
) -> str:
    model = None
    if models is not None and provider_index < len(models):
        model = models[provider_index]
    return _teacher_model(provider, model, env)


def _pipeline_provider_request(
    request: dict[str, Any],
    *,
    provider: str,
    model: str,
) -> dict[str, Any]:
    updated = dict(request)
    updated["request_id"] = f"{provider}-{request['request_id']}"
    updated["provider"] = provider
    updated["model"] = model
    updated["original_request_id"] = request["request_id"]
    return updated


def _run_openai_request(
    request: dict[str, Any],
    *,
    api_key: str,
    model: str,
    max_input_chars: int | None,
    max_output_tokens: int,
    transport: HttpTransport,
) -> dict[str, Any]:
    request = _compact_request_for_provider(request, max_input_chars=max_input_chars)
    payload = {
        "model": model,
        "max_output_tokens": max_output_tokens,
        "input": [
            _openai_message("system", request["messages"][0]["content"]),
            _openai_message("user", request["messages"][1]["content"]),
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "semantic_mirror_teacher_response",
                "schema": _teacher_response_schema(),
                "strict": False,
            }
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    raw_response = transport(OPENAI_RESPONSES_URL, headers, payload)
    text = _openai_response_text(raw_response)
    return _provider_response_record(
        _parse_teacher_response_json(text),
        request=request,
        model=model,
        provider="openai",
        provider_response_id=raw_response.get("id"),
    )


def _run_provider_request(
    request: dict[str, Any],
    *,
    provider: str,
    api_key: str,
    model: str,
    max_input_chars: int | None,
    max_output_tokens: int,
    transport: HttpTransport,
) -> dict[str, Any]:
    if provider == "openai":
        return _run_openai_request(
            request,
            api_key=api_key,
            model=model,
            max_input_chars=max_input_chars,
            max_output_tokens=max_output_tokens,
            transport=transport,
        )
    if provider == "anthropic":
        return _run_anthropic_request(
            request,
            api_key=api_key,
            model=model,
            max_input_chars=max_input_chars,
            max_output_tokens=max_output_tokens,
            transport=transport,
        )
    if provider == "gemini":
        return _run_gemini_request(
            request,
            api_key=api_key,
            model=model,
            max_input_chars=max_input_chars,
            max_output_tokens=max_output_tokens,
            transport=transport,
        )
    raise ValueError(f"unsupported provider {provider!r}")


def _run_provider_critic_request(
    request: dict[str, Any],
    *,
    provider: str,
    api_key: str,
    model: str,
    max_input_chars: int | None,
    max_output_tokens: int,
    transport: HttpTransport,
) -> dict[str, Any]:
    if provider == "openai":
        return _run_openai_critic_request(
            request,
            api_key=api_key,
            model=model,
            max_input_chars=max_input_chars,
            max_output_tokens=max_output_tokens,
            transport=transport,
        )
    if provider == "anthropic":
        return _run_anthropic_critic_request(
            request,
            api_key=api_key,
            model=model,
            max_input_chars=max_input_chars,
            max_output_tokens=max_output_tokens,
            transport=transport,
        )
    if provider == "gemini":
        return _run_gemini_critic_request(
            request,
            api_key=api_key,
            model=model,
            max_input_chars=max_input_chars,
            max_output_tokens=max_output_tokens,
            transport=transport,
        )
    raise ValueError(f"unsupported provider {provider!r}")


def _run_openai_critic_request(
    request: dict[str, Any],
    *,
    api_key: str,
    model: str,
    max_input_chars: int | None,
    max_output_tokens: int,
    transport: HttpTransport,
) -> dict[str, Any]:
    request = _compact_request_for_provider(request, max_input_chars=max_input_chars)
    payload = {
        "model": model,
        "max_output_tokens": max_output_tokens,
        "input": [
            _openai_message("system", request["messages"][0]["content"]),
            _openai_message("user", request["messages"][1]["content"]),
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "semantic_mirror_critic_response",
                "schema": _critic_response_schema(),
                "strict": False,
            }
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    raw_response = transport(OPENAI_RESPONSES_URL, headers, payload)
    return _provider_critic_response_record(
        _parse_teacher_response_json(_openai_response_text(raw_response)),
        request=request,
        model=model,
        provider="openai",
        provider_response_id=raw_response.get("id"),
    )


def _run_anthropic_critic_request(
    request: dict[str, Any],
    *,
    api_key: str,
    model: str,
    max_input_chars: int | None,
    max_output_tokens: int,
    transport: HttpTransport,
) -> dict[str, Any]:
    request = _compact_request_for_provider(request, max_input_chars=max_input_chars)
    payload = {
        "model": model,
        "max_tokens": max_output_tokens,
        "system": (
            request["messages"][0]["content"]
            + "\nReturn only one JSON object with request_id, candidate_result_id, "
            "error_labels, critique, and repair_priority."
        ),
        "messages": [
            {
                "role": "user",
                "content": request["messages"][1]["content"],
            }
        ],
        "tools": [
            {
                "name": "emit_semantic_mirror_critic_response",
                "description": "Emit one Semantic Mirror critic response object.",
                "input_schema": _critic_response_schema(),
            }
        ],
        "tool_choice": {
            "type": "tool",
            "name": "emit_semantic_mirror_critic_response",
        },
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    raw_response = transport(ANTHROPIC_MESSAGES_URL, headers, payload)
    return _provider_critic_response_record(
        _anthropic_tool_payload(raw_response, "emit_semantic_mirror_critic_response"),
        request=request,
        model=model,
        provider="anthropic",
        provider_response_id=raw_response.get("id"),
    )


def _run_gemini_critic_request(
    request: dict[str, Any],
    *,
    api_key: str,
    model: str,
    max_input_chars: int | None,
    max_output_tokens: int,
    transport: HttpTransport,
) -> dict[str, Any]:
    request = _compact_request_for_provider(request, max_input_chars=max_input_chars)
    encoded_model = urllib.parse.quote(model, safe="")
    payload = {
        "systemInstruction": {
            "parts": [{"text": request["messages"][0]["content"]}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": request["messages"][1]["content"]}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
        },
    }
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    raw_response = transport(GEMINI_GENERATE_URL.format(model=encoded_model), headers, payload)
    return _provider_critic_response_record(
        _parse_teacher_response_json(_gemini_response_text(raw_response)),
        request=request,
        model=model,
        provider="gemini",
        provider_response_id=raw_response.get("responseId") or raw_response.get("id"),
    )


def _run_anthropic_request(
    request: dict[str, Any],
    *,
    api_key: str,
    model: str,
    max_input_chars: int | None,
    max_output_tokens: int,
    transport: HttpTransport,
) -> dict[str, Any]:
    request = _compact_request_for_provider(request, max_input_chars=max_input_chars)
    payload = {
        "model": model,
        "max_tokens": max_output_tokens,
        "system": (
            request["messages"][0]["content"]
            + "\nReturn only one JSON object with request_id, sir_unit, rationale, "
            "and optional uncertainty_notes."
        ),
        "messages": [
            {
                "role": "user",
                "content": request["messages"][1]["content"],
            }
        ],
        "tools": [
            {
                "name": "emit_semantic_mirror_response",
                "description": "Emit one Semantic Mirror teacher response object.",
                "input_schema": _teacher_response_schema(),
            }
        ],
        "tool_choice": {
            "type": "tool",
            "name": "emit_semantic_mirror_response",
        },
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    raw_response = transport(ANTHROPIC_MESSAGES_URL, headers, payload)
    parsed = _anthropic_response_payload(raw_response)
    return _provider_response_record(
        parsed,
        request=request,
        model=model,
        provider="anthropic",
        provider_response_id=raw_response.get("id"),
    )


def _run_gemini_request(
    request: dict[str, Any],
    *,
    api_key: str,
    model: str,
    max_input_chars: int | None,
    max_output_tokens: int,
    transport: HttpTransport,
) -> dict[str, Any]:
    request = _compact_request_for_provider(request, max_input_chars=max_input_chars)
    encoded_model = urllib.parse.quote(model, safe="")
    payload = {
        "systemInstruction": {
            "parts": [{"text": request["messages"][0]["content"]}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": request["messages"][1]["content"]}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
        },
    }
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    raw_response = transport(GEMINI_GENERATE_URL.format(model=encoded_model), headers, payload)
    text = _gemini_response_text(raw_response)
    return _provider_response_record(
        _parse_teacher_response_json(text),
        request=request,
        model=model,
        provider="gemini",
        provider_response_id=raw_response.get("responseId") or raw_response.get("id"),
    )


def _compact_request_for_provider(
    request: dict[str, Any],
    *,
    max_input_chars: int | None,
) -> dict[str, Any]:
    if max_input_chars is None:
        return request
    compacted = json.loads(json.dumps(request))
    user_content = compacted["messages"][1]["content"]
    total_chars = len(compacted["messages"][0]["content"]) + len(user_content)
    if total_chars <= max_input_chars:
        return compacted
    allowed_user_chars = max(1000, max_input_chars - len(compacted["messages"][0]["content"]))
    compacted["messages"][1]["content"] = (
        user_content[:allowed_user_chars]
        + "\n\n[semantic-mirror note: provider input was truncated for API token budget; "
        "candidate must mark any unsupported or missing context as uncertainty.]"
    )
    return compacted


def _openai_message(role: str, text: str) -> dict[str, Any]:
    return {
        "role": role,
        "content": [
            {
                "type": "input_text",
                "text": text,
            }
        ],
    }


def _http_post_json(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"teacher provider API error {exc.code}: {body}") from exc


def _openai_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    for item in response.get("output", []):
        for content in item.get("content", []):
            if isinstance(content.get("text"), str):
                return content["text"]
    raise ValueError("OpenAI response did not include output text")


def _anthropic_response_text(response: dict[str, Any]) -> str:
    for block in response.get("content", []):
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            return block["text"]
    raise ValueError("Anthropic response did not include text content")


def _anthropic_response_payload(response: dict[str, Any]) -> dict[str, Any]:
    return _anthropic_tool_payload(response, "emit_semantic_mirror_response")


def _anthropic_tool_payload(response: dict[str, Any], tool_name: str) -> dict[str, Any]:
    for block in response.get("content", []):
        if (
            block.get("type") == "tool_use"
            and block.get("name") == tool_name
            and isinstance(block.get("input"), dict)
        ):
            return block["input"]
    return _parse_teacher_response_json(_anthropic_response_text(response))


def _gemini_response_text(response: dict[str, Any]) -> str:
    for candidate in response.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            if isinstance(part.get("text"), str):
                return part["text"]
    raise ValueError("Gemini response did not include candidate text")


def _parse_teacher_response_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("teacher response JSON must be an object")
    return parsed


def _provider_response_record(
    parsed: dict[str, Any],
    *,
    request: dict[str, Any],
    model: str,
    provider: str,
    provider_response_id: str | None,
) -> dict[str, Any]:
    if parsed.get("request_id") != request["request_id"]:
        parsed["request_id"] = request["request_id"]
    parsed.setdefault("model", model)
    parsed["provider"] = provider
    parsed["provider_response_id"] = provider_response_id
    return parsed


def _provider_critic_response_record(
    parsed: dict[str, Any],
    *,
    request: dict[str, Any],
    model: str,
    provider: str,
    provider_response_id: str | None,
) -> dict[str, Any]:
    if parsed.get("request_id") != request["request_id"]:
        parsed["request_id"] = request["request_id"]
    if parsed.get("candidate_result_id") != request.get("candidate_result_id"):
        parsed["candidate_result_id"] = request.get("candidate_result_id")
    parsed.setdefault("error_labels", [])
    parsed.setdefault("critique", "")
    parsed.setdefault("repair_priority", "medium")
    parsed.setdefault("model", model)
    parsed["provider"] = provider
    parsed["provider_response_id"] = provider_response_id
    return parsed


def _teacher_response_schema() -> dict[str, Any]:
    claim_object = {
        "type": "object",
        "additionalProperties": True,
        "required": ["claim", "source_spans"],
        "properties": {
            "claim": {"type": "string"},
            "source_spans": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
            "confidence": {"type": "number"},
        },
    }
    claim_array = {"type": "array", "items": claim_object}
    data_ml_details = {
        "type": "object",
        "required": list(DATA_ML_DETAIL_CATEGORIES),
        "properties": {category: claim_array for category in DATA_ML_DETAIL_CATEGORIES},
        "additionalProperties": claim_array,
    }
    return {
        "type": "object",
        "additionalProperties": True,
        "required": ["request_id", "sir_unit"],
        "properties": {
            "request_id": {"type": "string"},
            "sir_unit": {
                "type": "object",
                "additionalProperties": True,
                "required": [
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
                ],
                "properties": {
                    "unit_id": {"type": "string"},
                    "source_spans": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": True},
                    },
                    "language": {"type": "string"},
                    "symbol_type": {"type": "string"},
                    "name": {"type": "string"},
                    "qualified_name": {"type": "string"},
                    "algorithm": claim_object,
                    "control_flow": claim_array,
                    "reads": claim_array,
                    "writes": claim_array,
                    "calls": claim_array,
                    "returns": claim_array,
                    "side_effects": claim_array,
                    "failure_modes": claim_array,
                    "state_mutations": claim_array,
                    "external_dependencies": claim_array,
                    "data_ml_details": data_ml_details,
                    "hazards": claim_array,
                    "uncertainty": claim_array,
                    "confidence": {"type": "number"},
                },
            },
            "rationale": {"type": "string"},
            "uncertainty_notes": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }


def _critic_response_schema() -> dict[str, Any]:
    error_label = {
        "type": "object",
        "additionalProperties": True,
        "required": ["label", "severity"],
        "properties": {
            "label": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            "evidence": {"type": "string"},
            "source_spans": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": True,
        "required": ["request_id", "candidate_result_id", "error_labels", "critique", "repair_priority"],
        "properties": {
            "request_id": {"type": "string"},
            "candidate_result_id": {"type": "string"},
            "error_labels": {
                "type": "array",
                "items": error_label,
            },
            "critique": {"type": "string"},
            "repair_priority": {"type": "string", "enum": ["low", "medium", "high"]},
        },
    }


def _teacher_api_key(provider: str, env: dict[str, str]) -> str:
    key_by_provider = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_key = key_by_provider[provider]
    value = env.get(env_key) or os.environ.get(env_key)
    if not value:
        raise RuntimeError(f"{env_key} is required for provider={provider!r} teacher run")
    return value


def _teacher_model(provider: str, model: str | None, env: dict[str, str]) -> str:
    if model:
        return model
    provider_key = f"SEMANTIC_MIRROR_{provider.upper()}_MODEL"
    if env.get(provider_key) or os.environ.get(provider_key):
        return env.get(provider_key) or os.environ[provider_key]
    configured_provider = env.get("SEMANTIC_MIRROR_TEACHER_PROVIDER") or os.environ.get(
        "SEMANTIC_MIRROR_TEACHER_PROVIDER"
    )
    configured_model = env.get("SEMANTIC_MIRROR_TEACHER_MODEL") or os.environ.get(
        "SEMANTIC_MIRROR_TEACHER_MODEL"
    )
    if configured_provider == provider and configured_model:
        return configured_model
    defaults = {
        "openai": DEFAULT_OPENAI_MODEL,
        "anthropic": DEFAULT_ANTHROPIC_MODEL,
        "gemini": DEFAULT_GEMINI_MODEL,
    }
    return defaults[provider]


def _candidate_result(
    *,
    response: dict[str, Any],
    request: dict[str, Any] | None,
    positive_by_id: dict[str, dict[str, Any]],
    default_repo_path: Path,
    response_index: int,
) -> dict[str, Any]:
    if request is None:
        return _invalid_result(response, response_index, "response request_id does not match export")
    positive = positive_by_id[request["dataset_record_id"]]
    candidate = _extract_sir_unit(response)
    if candidate is None:
        return _invalid_result(response, response_index, "response does not contain parseable sir_unit", request)
    repo_path = _source_repo_path(positive, default_repo_path)

    validation_error: str | None = None
    try:
        validate_unit(candidate)
    except SchemaValidationError as exc:
        validation_error = str(exc)

    candidate_document = {
        "schema_version": "0.1.0",
        "source_path": positive["source_path"],
        "language": candidate.get("language", "python"),
        "profile": positive["profile"],
        "zoom": positive["zoom"],
        "units": [_candidate_for_repo_scoring(candidate, positive)],
        "unsupported_reasons": [],
    }
    verifier_report = score_document(candidate_document, repo_path=repo_path)
    penalties = dict(verifier_report["penalties"])
    if validation_error is not None:
        penalties["schema_validation_errors"] = penalties.get("schema_validation_errors", 0) + 1
    accepted = validation_error is None and not penalties
    disagreement_score = _disagreement_score(verifier_report, validation_error)
    needs_human_review = disagreement_score > 0 or bool(response.get("uncertainty_notes"))

    return {
        "result_id": f"teacher-result-{response_index}-{request['request_id']}",
        "request_id": request["request_id"],
        "dataset_record_id": request["dataset_record_id"],
        "model": response.get("model", request["model"]),
        "source_path": positive["source_path"],
        "source_repo_path": str(repo_path),
        "unit_id": positive["unit_id"],
        "qualified_name": positive["qualified_name"],
        "candidate": {"sir_unit": candidate},
        "positive": {"sir_unit": positive["target"]["sir_unit"]},
        "validation_error": validation_error,
        "verifier_report": {**verifier_report, "penalties": penalties},
        "accepted": accepted,
        "auto_reject": not accepted,
        "needs_human_review": needs_human_review,
        "disagreement_score": disagreement_score,
        "teacher_rationale": response.get("rationale"),
        "uncertainty_notes": response.get("uncertainty_notes", []),
    }


def _source_repo_path(record: dict[str, Any], default_repo_path: Path) -> Path:
    if record.get("source_repo_path"):
        return Path(record["source_repo_path"]).resolve()
    return default_repo_path


def _candidate_for_repo_scoring(candidate: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    scoring_candidate = dict(candidate)
    source_repo_id = record.get("source_repo_id")
    unit_id = scoring_candidate.get("unit_id")
    if source_repo_id and isinstance(unit_id, str) and unit_id.startswith(f"{source_repo_id}:"):
        scoring_candidate["unit_id"] = unit_id.split(":", 1)[1]
    return scoring_candidate


def _invalid_result(
    response: dict[str, Any],
    response_index: int,
    reason: str,
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "result_id": f"teacher-result-{response_index}-invalid",
        "request_id": response.get("request_id"),
        "dataset_record_id": None if request is None else request["dataset_record_id"],
        "model": response.get("model") or (request or {}).get("model"),
        "source_path": None,
        "unit_id": None,
        "qualified_name": None,
        "candidate": None,
        "positive": None,
        "validation_error": reason,
        "verifier_report": {"score": -3, "rewards": {}, "penalties": {"invalid_response": 1}, "issues": []},
        "accepted": False,
        "auto_reject": True,
        "needs_human_review": True,
        "disagreement_score": 3,
        "teacher_rationale": response.get("rationale"),
        "uncertainty_notes": response.get("uncertainty_notes", []),
    }


def _extract_sir_unit(response: dict[str, Any]) -> dict[str, Any] | None:
    raw = response.get("sir_unit")
    if raw is None and isinstance(response.get("response"), dict):
        raw = response["response"].get("sir_unit")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _critic_request(result: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "request_id": f"teacher-critic-{index}-{result['result_id']}",
        "kind": "candidate_critique",
        "candidate_result_id": result["result_id"],
        "source_path": result["source_path"],
        "unit_id": result["unit_id"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "Critique this Semantic IR candidate against static facts and verifier output. "
                    "Label missing facts, invented behavior, bad evidence spans, and low-confidence "
                    "claims. Prefer explicit uncertainty over silent omission."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "candidate": result["candidate"],
                        "positive_reference": result["positive"],
                        "verifier_report": result["verifier_report"],
                        "uncertainty_notes": result["uncertainty_notes"],
                        "response_contract": {
                            "error_labels": ["string"],
                            "critique": "short source-grounded critique",
                            "repair_priority": "low|medium|high",
                        },
                    },
                    indent=2,
                    sort_keys=True,
                ),
            },
        ],
    }


def _preference_pair(result: dict[str, Any]) -> dict[str, Any] | None:
    if result["positive"] is None or result["candidate"] is None:
        return None
    if result["accepted"] and result["disagreement_score"] == 0:
        return None
    return {
        "record_id": f"teacher-preference-{result['result_id']}",
        "prompt": json.dumps(
            {
                "source_path": result["source_path"],
                "unit_id": result["unit_id"],
                "qualified_name": result["qualified_name"],
                "task": "generate faithful SIR JSON unit",
            },
            sort_keys=True,
        ),
        "chosen": json.dumps(result["positive"]["sir_unit"], sort_keys=True),
        "rejected": json.dumps(result["candidate"]["sir_unit"], sort_keys=True),
        "metadata": {
            "candidate_result_id": result["result_id"],
            "model": result["model"],
            "verifier_report": result["verifier_report"],
            "preference_reason": "positive reference is verifier-backed; teacher candidate is penalized or uncertain",
        },
    }


def _eval_candidate_record(result: dict[str, Any], index: int) -> dict[str, Any]:
    record = {
        "record_id": f"teacher-candidate-{index}-{result['result_id']}",
        "dataset_record_id": result["dataset_record_id"],
        "unit_id": result["unit_id"],
        "source_path": result["source_path"],
        "source_repo_path": result.get("source_repo_path"),
        "model": result["model"],
        "accepted": result["accepted"],
        "auto_reject": result["auto_reject"],
        "validation_error": result.get("validation_error"),
        "verifier_penalties": result.get("verifier_report", {}).get("penalties", {}),
    }
    if result.get("candidate") and result["candidate"].get("sir_unit"):
        record["sir_unit"] = result["candidate"]["sir_unit"]
    return record


def _review_item(result: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "candidate_result_id": result["result_id"],
        "source_path": result["source_path"],
        "unit_id": result["unit_id"],
        "qualified_name": result["qualified_name"],
        "disagreement_score": result["disagreement_score"],
        "auto_reject": result["auto_reject"],
        "penalties": result["verifier_report"]["penalties"],
        "review_task": "inspect teacher candidate, labels, critique request, and optional repair",
    }


def _resolve_teacher_results_root(path: Path) -> Path:
    root = path.resolve()
    if (root / "candidate_results.jsonl").exists() and (root / "critic_requests.jsonl").exists():
        return root
    nested = root / "teacher_results"
    if (nested / "candidate_results.jsonl").exists() and (nested / "critic_requests.jsonl").exists():
        return nested
    raise FileNotFoundError(f"{path} does not contain teacher result files")


def _critic_label_record(
    response: dict[str, Any],
    request: dict[str, Any],
    candidate: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    labels = _normalise_error_labels(response.get("error_labels", []), candidate)
    return {
        "label_id": f"critic-label-{index}-{candidate['result_id']}",
        "request_id": request["request_id"],
        "candidate_result_id": candidate["result_id"],
        "source_path": candidate["source_path"],
        "unit_id": candidate["unit_id"],
        "qualified_name": candidate["qualified_name"],
        "model": response.get("model"),
        "provider": response.get("provider"),
        "provider_response_id": response.get("provider_response_id"),
        "repair_priority": _repair_priority(response.get("repair_priority"), labels),
        "critique": response.get("critique", ""),
        "error_labels": labels,
        "verifier_penalties": candidate.get("verifier_report", {}).get("penalties", {}),
        "validation_error": candidate.get("validation_error"),
    }


def _normalise_error_labels(value: Any, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            label = _normalise_error_label(item)
            if label is not None:
                labels.append(label)
    if not labels:
        labels.extend(_labels_from_verifier(candidate))
    return labels


def _normalise_error_label(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        return {
            "label": value,
            "severity": "medium",
            "source": "critic",
        }
    if not isinstance(value, dict):
        return None
    label = value.get("label") or value.get("kind") or value.get("error")
    if not isinstance(label, str) or not label:
        return None
    severity = value.get("severity")
    if severity not in {"low", "medium", "high"}:
        severity = "medium"
    normalised = {
        "label": label,
        "severity": severity,
        "source": value.get("source", "critic"),
    }
    for key in ("evidence", "source_spans", "field", "expected", "actual"):
        if key in value:
            normalised[key] = value[key]
    return normalised


def _labels_from_verifier(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    penalties = candidate.get("verifier_report", {}).get("penalties", {})
    for name, count in sorted(penalties.items()):
        labels.append(
            {
                "label": name,
                "severity": "high" if count else "low",
                "source": "verifier",
                "count": count,
            }
        )
    if candidate.get("validation_error"):
        labels.append(
            {
                "label": "schema_validation_error",
                "severity": "high",
                "source": "verifier",
                "evidence": candidate["validation_error"],
            }
        )
    return labels


def _repair_priority(value: Any, labels: list[dict[str, Any]]) -> str:
    if value in {"low", "medium", "high"}:
        return value
    if any(label.get("severity") == "high" for label in labels):
        return "high"
    return "medium" if labels else "low"


def _labeled_review_queue(teacher_results: Path, labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    review_path = teacher_results / "teacher_review_queue.jsonl"
    if not review_path.exists():
        return []
    labels_by_candidate = {item["candidate_result_id"]: item for item in labels}
    items = []
    for item in _read_jsonl(review_path):
        label = labels_by_candidate.get(item["candidate_result_id"])
        if label is None:
            items.append(item)
            continue
        items.append(
            {
                **item,
                "critic_label_id": label["label_id"],
                "critic_error_labels": label["error_labels"],
                "critic_repair_priority": label["repair_priority"],
                "critic_summary": label["critique"],
            }
        )
    return items


def _critic_label_summary(
    labels: list[dict[str, Any]],
    invalid_responses: list[dict[str, Any]],
) -> dict[str, Any]:
    by_label: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for item in labels:
        for label in item["error_labels"]:
            by_label[label["label"]] = by_label.get(label["label"], 0) + 1
            severity = label.get("severity", "medium")
            by_severity[severity] = by_severity.get(severity, 0) + 1
    return {
        "labeled_candidates": len(labels),
        "invalid_responses": len(invalid_responses),
        "error_labels": sum(len(item["error_labels"]) for item in labels),
        "by_label": dict(sorted(by_label.items())),
        "by_severity": dict(sorted(by_severity.items())),
    }


def _disagreement_score(verifier_report: dict[str, Any], validation_error: str | None) -> int:
    score = sum(verifier_report["penalties"].values())
    if validation_error is not None:
        score += 1
    return score


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_env(env_file: Path | str | None) -> dict[str, str]:
    candidates: list[Path] = []
    if env_file is not None:
        candidates.append(Path(env_file))
    candidates.append(Path.cwd() / ".env")
    values: dict[str, str] = {}
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        break
    return values


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _teacher_readme() -> str:
    return """# Semantic Mirror Teacher Requests

This directory is generated by `semantic-mirror teacher export`.

Send `candidate_requests.jsonl` to one or more frontier teacher models. Each
response line should include the original `request_id` and a `sir_unit` object.
Then run `semantic-mirror teacher ingest` to validate responses, auto-reject
unfaithful candidates, create eval-ready `teacher_candidates.jsonl`, create
critic requests, and emit preference pairs.
Use `teacher_candidates.jsonl` with `semantic-mirror eval candidates` as a
teacher baseline for the dataset units covered by the teacher run.
Run `semantic-mirror teacher run-critic` and `semantic-mirror teacher
ingest-critic` on the generated `critic_requests.jsonl` to attach structured
error labels to the review queue and teacher preference metadata.
"""
