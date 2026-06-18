"""Prepare SFT and RL training artifacts from curated Semantic Mirror datasets."""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from semantic_mirror.schema import (
    DATA_ML_DETAIL_CATEGORIES,
    SchemaValidationError,
    validate_unit,
)

TRAINING_VERSION = "0.1.0"
DEFAULT_BASE_MODEL = "unsloth/Qwen3.5-4B"
REQUIRED_TRAINING_MODULES = (
    "torch",
    "unsloth",
    "trl",
    "datasets",
    "transformers",
    "bitsandbytes",
    "peft",
)
OPTIONAL_TRAINING_MODULES = (
    "mergekit",
    "llm_blender",
    "weave",
)
AUDITED_TRAINING_MODULES = REQUIRED_TRAINING_MODULES + OPTIONAL_TRAINING_MODULES
UNSLOTH_PYTHON_MIN = (3, 11)
UNSLOTH_PYTHON_MAX_EXCLUSIVE = (3, 14)
UNSLOTH_PYTHON_RANGE = ">=3.11,<3.14"

ModuleProbe = Callable[[str], bool]
TorchProbe = Callable[[], dict[str, Any]]
NvidiaSmiRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
TrainingRunner = Callable[[list[str], Path, Path, Path], int]

COMPACT_FACT_LIMITS = {
    "control_flow": 1,
    "reads": 0,
    "writes": 2,
    "calls": 2,
    "returns": 2,
    "side_effects": 1,
    "failure_modes": 1,
    "state_mutations": 1,
    "external_dependencies": 1,
    "hazards": 1,
    "uncertainty": 1,
}
COMPACT_DATA_ML_LIMIT = 1
COMPACT_CODE_CHARS = 240
SIR_UNIT_TOP_LEVEL_KEYS = (
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
)
SIR_IDENTITY_FIELDS = (
    "unit_id",
    "language",
    "symbol_type",
    "name",
    "qualified_name",
    "source_spans",
)
SIR_LIST_FIELDS = (
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
)


def prepare_training_data(
    dataset_path: Path | str,
    out_path: Path | str,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    max_records: int | None = None,
    include_silver_when_gold_exists: bool = True,
    teacher_results_path: Path | str | None = None,
) -> dict[str, Any]:
    dataset = Path(dataset_path).resolve()
    out = Path(out_path).resolve()
    teacher_results = _resolve_teacher_results_path(teacher_results_path)
    dataset_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    silver = _read_jsonl(dataset / dataset_manifest["files"]["silver"])
    hard_negatives = _read_jsonl(dataset / dataset_manifest["files"]["hard_negative"])
    gold = _read_jsonl(dataset / dataset_manifest["files"]["gold"])
    teacher_preferences = _teacher_preference_records(teacher_results)

    positives = _positive_records(
        silver=silver,
        gold=gold,
        include_silver_when_gold_exists=include_silver_when_gold_exists,
    )
    if max_records is not None:
        positives = positives[:max_records]
    positive_by_unit = {record["unit_id"]: record for record in positives}
    filtered_negatives = [
        record for record in hard_negatives if record["positive_unit_id"] in positive_by_unit
    ]

    sft_records = [_sft_record(record, index) for index, record in enumerate(positives)]
    contrastive_records = [
        _contrastive_record(positive_by_unit[record["positive_unit_id"]], record, index)
        for index, record in enumerate(filtered_negatives)
    ]
    preference_pairs = [
        _preference_pair(positive_by_unit[record["positive_unit_id"]], record, index)
        for index, record in enumerate(filtered_negatives)
    ]
    preference_pairs.extend(teacher_preferences)
    rl_prompts = [_rl_prompt(record, index) for index, record in enumerate(positives)]

    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "sft.jsonl", sft_records)
    _write_jsonl(out / "contrastive_repair.jsonl", contrastive_records)
    _write_jsonl(out / "preference_pairs.jsonl", preference_pairs)
    _write_jsonl(out / "rl_prompts.jsonl", rl_prompts)

    sft_config = _sft_config(base_model)
    reward_config = _reward_config()
    (out / "unsloth_sft_config.json").write_text(
        json.dumps(sft_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "rl_reward_config.json").write_text(
        json.dumps(reward_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(_training_readme(), encoding="utf-8")
    (out / "run_unsloth_sft.py").write_text(_unsloth_sft_script(), encoding="utf-8")
    (out / "run_preference_dpo.py").write_text(_preference_dpo_script(), encoding="utf-8")
    (out / "run_reward_rl.py").write_text(_reward_rl_script(), encoding="utf-8")
    (out / "generate_sir_candidates.py").write_text(_generate_candidates_script(), encoding="utf-8")
    (out / "score_sir_candidates.py").write_text(_score_candidates_script(), encoding="utf-8")

    manifest = {
        "mode": "training_prepare",
        "training_version": TRAINING_VERSION,
        "dataset": str(dataset),
        "base_model": base_model,
        "generated_at": _now(),
        "input_counts": {
            "silver": len(silver),
            "gold": len(gold),
            "hard_negative": len(hard_negatives),
            "teacher_preference_pairs": len(teacher_preferences),
        },
        "output_counts": {
            "sft_records": len(sft_records),
            "contrastive_repair_records": len(contrastive_records),
            "preference_pairs": len(preference_pairs),
            "rl_prompts": len(rl_prompts),
        },
        "files": {
            "sft": "sft.jsonl",
            "contrastive_repair": "contrastive_repair.jsonl",
            "preference_pairs": "preference_pairs.jsonl",
            "rl_prompts": "rl_prompts.jsonl",
            "sft_config": "unsloth_sft_config.json",
            "reward_config": "rl_reward_config.json",
            "sft_script": "run_unsloth_sft.py",
            "preference_script": "run_preference_dpo.py",
            "rl_script": "run_reward_rl.py",
            "candidate_generation_script": "generate_sir_candidates.py",
            "reward_script": "score_sir_candidates.py",
        },
        "training_defaults": {
            "method": sft_config["method"],
            "target_model_size": sft_config["model_size_target"],
            "faithfulness_priority": "preserve required static facts before compactness",
        },
        "teacher_results": None if teacher_results is None else str(teacher_results),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_training_batch(training_path: Path | str) -> dict[str, Any]:
    root = Path(training_path).resolve()
    manifest_path = root / "manifest.json"
    issues: list[dict[str, Any]] = []
    if not manifest_path.exists():
        return _training_validation_report(root, [{"kind": "missing_file", "path": "manifest.json"}])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_files = [
        "sft",
        "contrastive_repair",
        "preference_pairs",
        "rl_prompts",
        "sft_config",
        "reward_config",
        "sft_script",
        "preference_script",
        "rl_script",
        "candidate_generation_script",
        "reward_script",
    ]
    for key in required_files:
        rel_path = manifest["files"].get(key)
        if not rel_path:
            issues.append({"kind": "missing_manifest_file_entry", "key": key})
            continue
        path = root / rel_path
        if not path.exists():
            issues.append({"kind": "missing_file", "path": rel_path})

    for key in ("sft", "contrastive_repair", "preference_pairs", "rl_prompts"):
        rel_path = manifest["files"].get(key)
        if rel_path and (root / rel_path).exists():
            issues.extend(_validate_jsonl(root / rel_path, key))

    if (root / manifest["files"].get("sft_config", "")).exists():
        issues.extend(_validate_sft_config(root / manifest["files"]["sft_config"]))
    if (root / manifest["files"].get("reward_config", "")).exists():
        issues.extend(_validate_reward_config(root / manifest["files"]["reward_config"]))
    for key in (
        "sft_script",
        "preference_script",
        "rl_script",
        "candidate_generation_script",
        "reward_script",
    ):
        rel_path = manifest["files"].get(key)
        if rel_path and (root / rel_path).exists():
            issues.extend(_validate_python_syntax(root / rel_path))

    return _training_validation_report(root, issues)


def audit_training_environment(
    training_path: Path | str,
    *,
    env_file: Path | str | None = None,
    require_gpu: bool = True,
    require_hf_token: bool = False,
    python_executable: str | None = None,
    module_probe: ModuleProbe | None = None,
    torch_probe: TorchProbe | None = None,
    nvidia_smi_runner: NvidiaSmiRunner | None = None,
    env_values: dict[str, str] | None = None,
    platform_name: str | None = None,
    python_version: str | None = None,
) -> dict[str, Any]:
    """Report whether a prepared batch can launch Unsloth/TRL training locally."""

    root = Path(training_path).resolve()
    validation = validate_training_batch(root)
    runtime_probe = (
        _probe_python_runtime(python_executable, AUDITED_TRAINING_MODULES)
        if python_executable is not None
        and module_probe is None
        and torch_probe is None
        and python_version is None
        else None
    )
    if runtime_probe is not None:
        module_details = runtime_probe["module_details"]
        torch_info = runtime_probe["torch"]
    else:
        module_details = {
            module: _module_detail(module, module_probe=module_probe)
            for module in REQUIRED_TRAINING_MODULES
            + OPTIONAL_TRAINING_MODULES
        }
        torch_info = _probe_torch(torch_probe=torch_probe)
    module_status = {
        module: bool(details.get("importable")) for module, details in module_details.items()
    }
    missing_modules = [
        module for module in REQUIRED_TRAINING_MODULES if not module_status.get(module)
    ]
    missing_optional_modules = [
        module for module in OPTIONAL_TRAINING_MODULES if not module_status.get(module)
    ]
    nvidia_smi = _probe_nvidia_smi(nvidia_smi_runner=nvidia_smi_runner)
    env, loaded_env_path = (
        (dict(env_values), None)
        if env_values is not None
        else _load_training_env(env_file)
    )
    secrets = {
        "hf_token_present": bool(env.get("HF_TOKEN") or env.get("HUGGINGFACE_HUB_TOKEN")),
        "wandb_api_key_present": bool(env.get("WANDB_API_KEY")),
    }
    current_platform = platform_name or platform.system()
    current_python_version = (
        python_version
        or (runtime_probe or {}).get("python_version")
        or platform.python_version()
    )
    runtime_python_executable = (
        (runtime_probe or {}).get("python_executable")
        or python_executable
        or sys.executable
    )
    python_supported = _python_version_supported_for_unsloth(current_python_version)

    checks = []
    if python_executable is not None:
        checks.append(
            _check(
                "python_executable_probe",
                bool((runtime_probe or {}).get("ok", False)),
                required=True,
                actual={
                    "python_executable": python_executable,
                    "probe_error": (runtime_probe or {}).get("probe_error"),
                },
                detail="The requested training Python executable can run the runtime probe.",
            )
        )
    checks.extend([
        _check(
            "training_batch_valid",
            validation["passed"],
            required=True,
            actual={"issues": len(validation["issues"])},
            detail="Prepared SFT/RL files and generated scripts validate.",
        ),
        _check(
            "python_version_supported_for_unsloth",
            python_supported,
            required=True,
            actual={
                "python_version": current_python_version,
                "supported_range": UNSLOTH_PYTHON_RANGE,
            },
            detail=(
                "The Unsloth training target requires a Python version in "
                f"{UNSLOTH_PYTHON_RANGE}; create the GPU venv with Python 3.11-3.13."
            ),
        ),
        _check(
            "required_training_modules",
            not missing_modules,
            required=True,
            actual={
                "missing": missing_modules,
                "available": module_status,
                "details": module_details,
            },
            detail=(
                "Unsloth, TRL, DPO optional dependencies, datasets, transformers, "
                "torch, bitsandbytes, and peft are importable."
            ),
        ),
        _check(
            "optional_training_modules",
            not missing_optional_modules,
            required=False,
            actual={
                "missing": missing_optional_modules,
                "available": module_status,
                "details": module_details,
            },
            detail="Optional merge/eval/telemetry helpers are importable when available.",
        ),
        _check(
            "torch_importable",
            bool(torch_info.get("importable")),
            required=True,
            actual=torch_info,
            detail="PyTorch imports without runtime errors.",
        ),
        _check(
            "torch_cuda_available",
            (not require_gpu) or bool(torch_info.get("cuda_available")),
            required=require_gpu,
            actual={
                "cuda_available": torch_info.get("cuda_available"),
                "device_count": torch_info.get("device_count"),
                "cuda_version": torch_info.get("cuda_version"),
                "bf16_supported": torch_info.get("bf16_supported"),
                "error": torch_info.get("error"),
            },
            detail="CUDA is required for the default Qwen3-family LoRA training target.",
        ),
        _check(
            "hf_token_available",
            secrets["hf_token_present"] or not require_hf_token,
            required=require_hf_token,
            actual={"present": secrets["hf_token_present"]},
            detail="A Hugging Face token is available for model or dataset downloads.",
        ),
    ])
    if current_platform == "Windows":
        checks.append(
            _check(
                "native_windows_runtime",
                True,
                required=False,
                actual={"platform": current_platform},
                detail="Native Windows may need WSL/Linux CUDA packaging for Unsloth training.",
            )
        )
    checks.append(
        _check(
            "nvidia_smi_detected",
            bool(nvidia_smi.get("available")) or not require_gpu,
            required=False,
            actual=nvidia_smi,
            detail="nvidia-smi is useful evidence for GPU identity and memory, but torch CUDA is the launch gate.",
        )
    )

    passed = all(check["passed"] for check in checks if check["required"])
    failed_required_checks = [
        check["name"] for check in checks if check["required"] and not check["passed"]
    ]
    repro_command = _training_audit_repro_command(
        root,
        env_file=loaded_env_path,
        require_gpu=require_gpu,
        require_hf_token=require_hf_token,
        python_executable=python_executable,
    )
    blocker_summary = _training_audit_blocker_summary(
        failed_required_checks,
        python_supported=python_supported,
        missing_modules=missing_modules,
        torch_info=torch_info,
        require_gpu=require_gpu,
    )
    return {
        "mode": "training_environment_audit",
        "training_version": TRAINING_VERSION,
        "training_dir": str(root),
        "generated_at": _now(),
        "passed": passed,
        "ready_to_launch": passed,
        "require_gpu": require_gpu,
        "require_hf_token": require_hf_token,
        "validation": validation,
        "environment": {
            "python_executable": runtime_python_executable,
            "python_version": current_python_version,
            "platform": current_platform,
            "platform_release": platform.release(),
            "loaded_env_file": None if loaded_env_path is None else str(loaded_env_path),
            "secrets": secrets,
        },
        "repro": {
            "audit_command": repro_command,
            "required_python_range": UNSLOTH_PYTHON_RANGE,
            "required_modules": list(REQUIRED_TRAINING_MODULES),
            "optional_modules": list(OPTIONAL_TRAINING_MODULES),
            "training_requirements": _training_requirements().splitlines(),
        },
        "blocker": {
            "blocked": not passed,
            "failed_required_checks": failed_required_checks,
            "summary": blocker_summary,
            "recommended_fallback": (
                "Use the generated Windows-hosted WSL CUDA bundle path until "
                "this audit passes in a Windows-native Python runtime."
                if not passed
                else None
            ),
        },
        "checks": checks,
        "command_hints": _training_command_hints(
            missing_modules,
            python_supported=python_supported,
            require_gpu=require_gpu,
        ),
    }


def launch_training_job(
    training_path: Path | str,
    *,
    stage: str,
    output_dir: Path | str,
    dry_run: bool = False,
    env_file: Path | str | None = None,
    require_gpu: bool = True,
    require_hf_token: bool = False,
    python_executable: str | None = None,
    model_name_or_path: str | None = None,
    beta: float = 0.1,
    max_steps: int | None = None,
    kl_coef: float = 0.05,
    schema_prefix_mode: str = "schema-scaffold",
    resume_from_checkpoint: str | None = None,
    seed: int | None = None,
    report_out: Path | str | None = None,
    audit_report: dict[str, Any] | None = None,
    runner: TrainingRunner | None = None,
) -> dict[str, Any]:
    """Launch, or dry-run launch, one generated training script after readiness gates."""

    if stage not in {"sft", "dpo", "rl"}:
        raise ValueError("stage must be 'sft', 'dpo', or 'rl'")
    root = Path(training_path).resolve()
    output = Path(output_dir).resolve()
    audit = audit_report or audit_training_environment(
        root,
        env_file=env_file,
        require_gpu=require_gpu,
        require_hf_token=require_hf_token,
        python_executable=python_executable,
    )
    command: list[str] | None = None
    command_error: str | None = None
    try:
        command = _training_launch_command(
            root,
            stage=stage,
            output_dir=output,
            python_executable=python_executable,
            model_name_or_path=model_name_or_path,
            beta=beta,
            max_steps=max_steps,
            kl_coef=kl_coef,
            schema_prefix_mode=schema_prefix_mode,
            resume_from_checkpoint=resume_from_checkpoint,
            seed=seed,
        )
    except Exception as exc:
        command_error = str(exc)
    report: dict[str, Any] = {
        "mode": "training_launch",
        "training_version": TRAINING_VERSION,
        "stage": stage,
        "training_dir": str(root),
        "output_dir": str(output),
        "generated_at": _now(),
        "dry_run": dry_run,
        "launched": False,
        "would_launch": audit["passed"],
        "passed": audit["passed"] if dry_run else False,
        "command": command,
        "command_error": command_error,
        "audit": audit,
    }
    if not audit["passed"]:
        report["reason"] = "environment_not_ready"
        _write_optional_report(report, report_out)
        return report
    if command is None:
        report["reason"] = "command_unavailable"
        report["passed"] = False
        _write_optional_report(report, report_out)
        return report
    if dry_run:
        report["reason"] = "dry_run"
        _write_optional_report(report, report_out)
        return report

    output.mkdir(parents=True, exist_ok=True)
    stdout_path = output / f"{stage}_stdout.log"
    stderr_path = output / f"{stage}_stderr.log"
    exit_code = (runner or _run_training_subprocess)(command, root, stdout_path, stderr_path)
    report.update(
        {
            "launched": True,
            "passed": exit_code == 0,
            "exit_code": exit_code,
            "logs": {
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            },
        }
    )
    _write_optional_report(report, report_out)
    return report


def inspect_full_training_eval_resume(
    run_dir: Path | str,
    *,
    sft_steps: int | None = None,
    dpo_steps: int | None = None,
    rl_steps: int | None = None,
    reuse_stage_outputs: bool = False,
    sft_resume_from_checkpoint: Path | str | None = None,
    dpo_resume_from_checkpoint: Path | str | None = None,
    sft_save_steps: int = 10,
    dpo_save_steps: int = 10,
    sft_save_total_limit: int = 3,
    dpo_save_total_limit: int = 3,
    out_path: Path | str | None = None,
    markdown_out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Preview full-eval stage reuse/resume decisions without launching training."""

    run = Path(run_dir)
    requested_steps = {
        "sft": sft_steps,
        "dpo": dpo_steps,
        "rl": rl_steps,
    }
    resume_from_checkpoint = {
        "sft": Path(sft_resume_from_checkpoint) if sft_resume_from_checkpoint else None,
        "dpo": Path(dpo_resume_from_checkpoint) if dpo_resume_from_checkpoint else None,
        "rl": None,
    }
    stage_dirs = {
        "sft": run / "semantic-mirror-sft",
        "dpo": run / "semantic-mirror-dpo",
        "rl": run / "semantic-mirror-rl",
    }
    decisions = {
        stage: _full_eval_resume_stage_decision(
            stage=stage,
            stage_dir=stage_dir,
            requested_max_steps=requested_steps[stage],
            reuse_stage_outputs=reuse_stage_outputs,
            resume_from_checkpoint=resume_from_checkpoint[stage],
        )
        for stage, stage_dir in stage_dirs.items()
    }
    manifest = {
        "mode": "full_training_eval_resume_inspection",
        "training_version": TRAINING_VERSION,
        "run_dir": str(run),
        "generated_at": _now(),
        "requested_max_steps": requested_steps,
        "reuse_stage_outputs_enabled": reuse_stage_outputs,
        "checkpoint_policy": {
            "sft_save_steps": sft_save_steps,
            "dpo_save_steps": dpo_save_steps,
            "sft_save_total_limit": sft_save_total_limit,
            "dpo_save_total_limit": dpo_save_total_limit,
        },
        "decisions": decisions,
        "files": {
            "json": str(out_path) if out_path is not None else None,
            "markdown": str(markdown_out_path) if markdown_out_path is not None else None,
        },
    }
    if out_path is not None:
        _write_optional_report(manifest, out_path)
    if markdown_out_path is not None:
        Path(markdown_out_path).write_text(
            _full_eval_resume_inspection_markdown(manifest),
            encoding="utf-8",
        )
    return manifest


def package_training_bundle(
    training_path: Path | str,
    out_path: Path | str,
    *,
    env_file: Path | str | None = None,
    require_gpu: bool = True,
    require_hf_token: bool = False,
    python_executable: str | None = None,
    module_probe: ModuleProbe | None = None,
    torch_probe: TorchProbe | None = None,
    nvidia_smi_runner: NvidiaSmiRunner | None = None,
    env_values: dict[str, str] | None = None,
    platform_name: str | None = None,
    python_version: str | None = None,
) -> dict[str, Any]:
    """Create a portable Linux/WSL/Colab handoff bundle for prepared training data."""

    root = Path(training_path).resolve()
    out = Path(out_path).resolve()
    if out == root or _is_relative_to(out, root):
        raise ValueError("training package output must not be inside the training directory")

    validation = validate_training_batch(root)
    out.mkdir(parents=True, exist_ok=True)
    if not validation["passed"]:
        manifest = {
            "mode": "training_package",
            "training_version": TRAINING_VERSION,
            "training_dir": str(root),
            "out": str(out),
            "generated_at": _now(),
            "passed": False,
            "reason": "training_batch_invalid",
            "validation": validation,
            "files": {},
        }
        _write_package_manifest(out, manifest)
        return manifest

    audit = audit_training_environment(
        root,
        env_file=env_file,
        require_gpu=require_gpu,
        require_hf_token=require_hf_token,
        python_executable=python_executable,
        module_probe=module_probe,
        torch_probe=torch_probe,
        nvidia_smi_runner=nvidia_smi_runner,
        env_values=env_values,
        platform_name=platform_name,
        python_version=python_version,
    )
    training_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    training_target = out / "training"
    source_target = out / "src" / "semantic_mirror"
    audit_target = out / "audit"
    launch_target = out / "launch"
    setup_target = out / "setup"

    _replace_directory(training_target)
    _copy_training_artifacts(root, training_target)
    _replace_directory(source_target)
    shutil.copytree(
        Path(__file__).resolve().parent,
        source_target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    _replace_directory(audit_target)
    audit_target.mkdir(parents=True, exist_ok=True)
    (audit_target / "current_environment.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _replace_directory(launch_target)
    launch_target.mkdir(parents=True, exist_ok=True)
    _replace_directory(setup_target)
    setup_target.mkdir(parents=True, exist_ok=True)
    _write_project_metadata(out)
    (out / "requirements-training.txt").write_text(
        _training_requirements(),
        encoding="utf-8",
    )
    (out / ".env.training.example").write_text(
        _training_env_example(),
        encoding="utf-8",
    )
    (out / "ENVIRONMENT.md").write_text(
        _training_environment_guide(audit),
        encoding="utf-8",
    )
    commands = _package_launch_commands()
    (launch_target / "commands.json").write_text(
        json.dumps(commands, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command_manifest = _package_launch_command_manifest(commands)
    (launch_target / "commands_manifest.json").write_text(
        json.dumps(command_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_launch_scripts(launch_target)
    _write_bootstrap_scripts(setup_target)
    (out / "README.md").write_text(
        _training_package_readme(audit),
        encoding="utf-8",
    )
    source_freshness = generate_training_package_source_freshness(
        out,
        repo_root=Path(__file__).resolve().parents[2],
        out_path=out / "source_freshness.json",
        markdown_out_path=out / "source_freshness.md",
    )

    files = {
        "training_dir": "training",
        "runtime_source": "src/semantic_mirror",
        "source_freshness": "source_freshness.json",
        "source_freshness_markdown": "source_freshness.md",
        "project_config": "pyproject.toml",
        "requirements": "requirements-training.txt",
        "environment_example": ".env.training.example",
        "environment_guide": "ENVIRONMENT.md",
        "audit": "audit/current_environment.json",
        "launch_commands": "launch/commands.json",
        "launch_command_manifest": "launch/commands_manifest.json",
        "linux_cuda_bootstrap": "setup/bootstrap_linux_cuda.sh",
        "wsl_bootstrap": "setup/bootstrap_wsl_ubuntu.ps1",
        "full_training_eval_launcher": "launch/run_full_training_eval.sh",
        "full_training_eval_resume_inspector": "launch/inspect_full_training_eval_resume.sh",
        "smoke_chain_launcher": "launch/run_smoke_chain.sh",
        "wsl_smoke_chain_launcher": "launch/run_wsl_smoke_chain.ps1",
        "sft_launcher": "launch/run_sft.sh",
        "dpo_launcher": "launch/run_dpo.sh",
        "rl_launcher": "launch/run_rl.sh",
        "candidate_launcher": "launch/generate_candidates.sh",
        "score_launcher": "launch/score_candidates.sh",
    }
    manifest = {
        "mode": "training_package",
        "training_version": TRAINING_VERSION,
        "training_dir": str(root),
        "out": str(out),
        "generated_at": _now(),
        "passed": True,
        "current_runtime_ready": audit["ready_to_launch"],
        "current_runtime_failed_checks": [
            check["name"] for check in audit["checks"] if check["required"] and not check["passed"]
        ],
        "training_manifest": training_manifest,
        "output_counts": training_manifest.get("output_counts", {}),
        "validation": validation,
        "audit_report": files["audit"],
        "source_freshness": {
            "path": files["source_freshness"],
            "markdown_path": files["source_freshness_markdown"],
            "all_compared_files_match": source_freshness["all_compared_files_match"],
            "compared_file_count": source_freshness["compared_file_count"],
            "git_commit": source_freshness["git_commit"],
        },
        "files": files,
        "launch_commands": commands,
        "launch_command_manifest": command_manifest,
    }
    _write_package_manifest(out, manifest)
    return manifest


def generate_training_package_source_freshness(
    package_path: Path | str,
    *,
    repo_root: Path | str | None = None,
    out_path: Path | str | None = None,
    markdown_out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Hash-compare a packaged runtime source tree against repository source."""

    package = Path(package_path).resolve()
    repo = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    repo_source = repo / "src" / "semantic_mirror"
    package_source = package / "src" / "semantic_mirror"
    source_files = _runtime_source_files(repo_source)
    comparisons = [
        _source_freshness_comparison(
            relative_path=relative_path,
            repo_root=repo,
            package_root=package,
        )
        for relative_path in source_files
    ]
    package_specific_docs = [
        {
            "relative_path": "README.md",
            "reason": "Generated training-bundle README, intentionally not the repository README.",
        },
        {
            "relative_path": "ENVIRONMENT.md",
            "reason": "Generated package environment/runbook documentation.",
        },
        {
            "relative_path": ".env.training.example",
            "reason": "Generated sanitized package environment template.",
        },
    ]
    mismatched_files = [row["relative_path"] for row in comparisons if not row["match"]]
    missing_package_specific_docs = [
        doc["relative_path"]
        for doc in package_specific_docs
        if not (package / doc["relative_path"]).exists()
    ]
    report = {
        "mode": "semantic_mirror_package_source_freshness",
        "training_version": TRAINING_VERSION,
        "generated_at": _now(),
        "repo_root": str(repo),
        "package_root": str(package),
        "git_commit": _repo_current_commit(repo),
        "compared_scope": "src/semantic_mirror runtime source tree",
        "repo_source_exists": repo_source.exists(),
        "package_source_exists": package_source.exists(),
        "all_compared_files_match": bool(source_files)
        and not any(not row["match"] for row in comparisons),
        "compared_file_count": len(comparisons),
        "mismatched_files": mismatched_files,
        "comparisons": comparisons,
        "all_package_specific_docs_present": all(
            (package / doc["relative_path"]).exists() for doc in package_specific_docs
        ),
        "missing_package_specific_docs": missing_package_specific_docs,
        "package_specific_docs": [
            {
                **doc,
                "package_exists": (package / doc["relative_path"]).exists(),
                "package_sha256": (
                    _sha256_file(package / doc["relative_path"])
                    if (package / doc["relative_path"]).exists()
                    else None
                ),
            }
            for doc in package_specific_docs
        ],
        "git_status_short_branch_ignored": _repo_git_status_short(repo),
        "files": {},
    }
    json_out = (
        Path(out_path).resolve()
        if out_path is not None
        else package / "source_freshness.json"
    )
    markdown_out = (
        Path(markdown_out_path).resolve()
        if markdown_out_path is not None
        else package / "source_freshness.md"
    )
    report["files"] = {
        "json": str(json_out),
        "markdown": str(markdown_out),
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_out.write_text(_source_freshness_markdown(report), encoding="utf-8")
    return report


DIAGNOSTIC_PLOT_SPECS = {
    "sft_loss": {
        "title": "SFT loss",
        "x_label": "training step",
        "y_label": "loss",
        "metric": "loss",
    },
    "dpo_loss": {
        "title": "DPO loss",
        "x_label": "training step",
        "y_label": "loss",
        "metric": "loss",
    },
    "dpo_reward_accuracy": {
        "title": "DPO reward accuracy",
        "x_label": "training step",
        "y_label": "reward accuracy",
        "metric": "reward_accuracy",
    },
    "rl_reward": {
        "title": "RL reward",
        "x_label": "training step",
        "y_label": "reward",
        "metric": "reward",
    },
    "rl_parseability": {
        "title": "RL raw parseability",
        "x_label": "training step",
        "y_label": "parseable raw output",
        "metric": "raw_parseable",
    },
    "generation_lengths": {
        "title": "Generation lengths",
        "x_label": "sample",
        "y_label": "characters",
        "metric": "generation_length",
    },
    "eval_metrics": {
        "title": "Evaluation metrics",
        "x_label": "report",
        "y_label": "metric value",
        "metric": "eval_metric",
    },
    "schema_coverage": {
        "title": "Schema and coverage",
        "x_label": "report",
        "y_label": "rate",
        "metric": "schema_coverage",
    },
}


def generate_training_diagnostics(
    run_path: Path | str,
    *,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Generate JSON/Markdown/PNG diagnostics from training logs and eval reports."""

    run = Path(run_path).resolve()
    diagnostics = Path(out_path).resolve() if out_path is not None else run / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    collected = _collect_training_metrics(run)
    plots: dict[str, dict[str, Any]] = {}
    for plot_name, spec in DIAGNOSTIC_PLOT_SPECS.items():
        points = collected["series"].get(plot_name, [])
        png_name = f"{plot_name}.png"
        png_path = diagnostics / png_name
        _write_metric_png(
            png_path,
            title=spec["title"],
            x_label=spec["x_label"],
            y_label=spec["y_label"],
            points=points,
        )
        plots[plot_name] = {
            "path": png_name,
            "title": spec["title"],
            "x_axis": spec["x_label"],
            "y_axis": spec["y_label"],
            "metric": spec["metric"],
            "points": len(points),
            "source_files": sorted({point["source"] for point in points}),
            "missing": not points,
        }

    summary = {
        "mode": "training_diagnostics",
        "training_version": TRAINING_VERSION,
        "run_dir": str(run),
        "diagnostics_dir": str(diagnostics),
        "generated_at": _now(),
        "plots": plots,
        "sources": collected["sources"],
        "missing_metrics": [
            name for name, plot in plots.items() if plot["missing"]
        ],
    }
    (diagnostics / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (diagnostics / "training_summary.md").write_text(
        _training_diagnostics_markdown(summary),
        encoding="utf-8",
    )
    return summary


def summarize_full_eval_contract_status(
    run_path: Path | str,
    *,
    sft_steps: int | None = None,
    dpo_steps: int | None = None,
    rl_steps: int | None = None,
    repo_root: Path | str | None = None,
    windows_audit_path: Path | str | None = None,
    wsl_smoke_manifest_path: Path | str | None = None,
    package_source_freshness_path: Path | str | None = None,
    human_study_suite_path: Path | str | None = None,
    human_study_coverage_paths: Iterable[Path | str] | None = None,
    out_path: Path | str | None = None,
    markdown_out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Summarize whether a full-eval outputs directory proves the contract gates."""

    run = Path(run_path).resolve()
    summary_path = run / "training_eval_summary.json"
    eval_summary = _read_json_file(summary_path)
    requested_steps = {
        "sft": sft_steps,
        "dpo": dpo_steps,
        "rl": rl_steps,
    }
    configured_steps = (
        eval_summary.get("eval_run_config", {}).get("requested_max_steps", {})
    if isinstance(eval_summary, dict)
    else {}
    )
    for stage in requested_steps:
        if requested_steps[stage] is None and isinstance(configured_steps, dict):
            configured_value = configured_steps.get(stage)
            if isinstance(configured_value, int):
                requested_steps[stage] = configured_value
    eval_summary_status = _eval_summary_contract_status(eval_summary, requested_steps)

    stage_status = {
        stage: _stage_contract_status(run, stage, requested_steps[stage])
        for stage in ("sft", "dpo", "rl")
    }
    required_reports = {
        "sft_eval": run / "sft_eval.json",
        "sft_vs_baseline": run / "sft_vs_baseline.json",
        "dpo_eval": run / "dpo_eval.json",
        "dpo_vs_sft": run / "dpo_vs_sft.json",
        "rl_eval": run / "rl_eval.json",
        "rl_vs_sft": run / "rl_vs_sft.json",
    }
    report_status = {
        name: _json_report_status(path, require_passed=True)
        for name, path in required_reports.items()
    }
    report_stage_map = {
        "sft_eval": "sft",
        "sft_vs_baseline": "sft",
        "dpo_eval": "dpo",
        "dpo_vs_sft": "dpo",
        "rl_eval": "rl",
        "rl_vs_sft": "rl",
    }
    for name, status in report_status.items():
        stage = report_stage_map[name]
        stage_current = stage_status[stage]["manifest_matches_requested_max_steps"]
        status["stage"] = stage
        status["stage_current_for_requested_steps"] = stage_current
        status["current_for_requested_stage"] = (
            status["exists"] and status["passed"] is True and stage_current
        )
    sample_status = {
        stage: _sample_contract_status(run / "samples" / stage)
        for stage in ("sft", "dpo", "rl")
    }
    for stage, status in sample_status.items():
        complete = (
            status["manifest_exists"]
            and status["raw_candidates_exist"]
            and status["repaired_candidates_exist"]
            and status["inspection_markdown_exists"]
        )
        status["stage_current_for_requested_steps"] = stage_status[stage][
            "manifest_matches_requested_max_steps"
        ]
        status["complete_for_requested_stage"] = (
            complete and status["stage_current_for_requested_steps"]
        )
    diagnostics_status = _diagnostics_contract_status(run / "diagnostics", run=run)
    diagnostics_status["stages_current_for_requested_steps"] = all(
        stage_status[stage]["manifest_matches_requested_max_steps"]
        for stage in ("sft", "dpo", "rl")
    )
    diagnostics_status["stale_or_missing_stages"] = [
        stage
        for stage in ("sft", "dpo", "rl")
        if not stage_status[stage]["manifest_matches_requested_max_steps"]
    ]
    resume_inspection_status = _resume_inspection_contract_status(
        run / "full_training_eval_resume_inspection.json"
    )
    stage_recovery_status = _stage_recovery_contract_status(
        run=run,
        stage_status=stage_status,
        report_status=report_status,
        sample_status=sample_status,
        resume_inspection_status=resume_inspection_status,
    )
    gates = [
        _contract_gate(
            "training_eval_summary_exists",
            eval_summary is not None,
            evidence=str(summary_path),
        ),
        _contract_gate(
            "training_eval_summary_passed",
            bool(eval_summary and eval_summary.get("passed")),
            evidence=str(summary_path),
        ),
        _contract_gate(
            "training_eval_summary_matches_requested_steps",
            eval_summary_status["matches_requested_steps"],
            actual=eval_summary_status["actual"],
            expected=eval_summary_status["expected"],
            evidence=str(summary_path),
        ),
        _contract_gate(
            "all_final_eval_gates_passed",
            bool(eval_summary and eval_summary.get("all_final_eval_gates_passed")),
            evidence=str(summary_path),
        ),
        *[
            _contract_gate(
                f"{stage}_stage_manifest_matches_requested_steps",
                status["manifest_exists"]
                and status["requested_max_steps"] is not None
                and status["manifest_max_steps"] == status["requested_max_steps"],
                actual=status["manifest_max_steps"],
                expected=status["requested_max_steps"],
                evidence=status["manifest_path"],
            )
            for stage, status in stage_status.items()
        ],
        *[
            _contract_gate(
                f"{name}_exists_and_passed",
                status["current_for_requested_stage"],
                actual={
                    "exists": status["exists"],
                    "passed": status["passed"],
                    "stage": status["stage"],
                    "stage_current_for_requested_steps": status[
                        "stage_current_for_requested_steps"
                    ],
                },
                expected=True,
                evidence=status["path"],
            )
            for name, status in report_status.items()
        ],
        *[
            _contract_gate(
                f"{stage}_sample_inspection_complete",
                status["complete_for_requested_stage"],
                actual={
                    "complete": (
                        status["manifest_exists"]
                        and status["raw_candidates_exist"]
                        and status["repaired_candidates_exist"]
                        and status["inspection_markdown_exists"]
                    ),
                    "stage_current_for_requested_steps": status[
                        "stage_current_for_requested_steps"
                    ],
                },
                expected=True,
                evidence=status["sample_dir"],
            )
            for stage, status in sample_status.items()
        ],
        _contract_gate(
            "diagnostic_summary_exists",
            diagnostics_status["summary_exists"],
            evidence=diagnostics_status["summary_path"],
        ),
        _contract_gate(
            "diagnostic_plots_exist",
            (
                diagnostics_status["all_required_plots_exist"]
                and diagnostics_status["sources_current_for_run"]
                and diagnostics_status["stages_current_for_requested_steps"]
            ),
            actual=_diagnostic_gate_actual(diagnostics_status),
            expected={
                "required_plots": diagnostics_status["required_plots"],
                "sources_current_for_run": True,
                "stages_current_for_requested_steps": True,
            },
            evidence=str(run / "diagnostics"),
        ),
    ]
    repo_hygiene_status = _repo_hygiene_contract_status(repo_root)
    windows_readiness_status = _windows_readiness_contract_status(
        windows_audit_path=windows_audit_path,
        wsl_smoke_manifest_path=wsl_smoke_manifest_path,
    )
    if windows_readiness_status.get("checked"):
        gates.append(
            _contract_gate(
                "windows_unsloth_readiness_passed",
                windows_readiness_status.get("passed") is True,
                actual={
                    "native_passed": windows_readiness_status.get("native_passed"),
                    "native_blocked": windows_readiness_status.get("native_blocked"),
                    "wsl_smoke_complete": windows_readiness_status.get(
                        "wsl_smoke_complete"
                    ),
                    "wsl_failed_checks": windows_readiness_status.get(
                        "wsl_failed_checks"
                    ),
                    "wsl_smoke_manifest_mode": windows_readiness_status.get(
                        "wsl_smoke_manifest_mode"
                    ),
                },
                expected={
                    "native_passed": True,
                    "or_native_blocked_and_wsl_smoke_complete": True,
                },
                evidence=windows_readiness_status.get("wsl_smoke_manifest_path")
                or windows_readiness_status.get("windows_audit_path"),
            )
        )
    package_source_status = _package_source_freshness_contract_status(
        package_source_freshness_path,
        repo_hygiene_status=repo_hygiene_status,
    )
    package_command_manifest_status = _package_command_manifest_contract_status(
        package_source_status
    )
    package_metadata_status = _package_metadata_contract_status(package_source_status)
    gates.extend(
        [
            _contract_gate(
                "package_source_freshness_valid_when_checked",
                package_source_status.get("passed") is not False,
                actual={
                    "checked": package_source_status.get("checked"),
                    "git_commit_matches_repo": package_source_status.get(
                        "git_commit_matches_repo"
                    ),
                    "all_compared_files_match": package_source_status.get(
                        "all_compared_files_match"
                    ),
                    "compared_file_count": package_source_status.get("compared_file_count"),
                },
                expected={
                    "git_commit_matches_repo": True,
                    "all_compared_files_match": True,
                    "compared_file_count": ">0",
                    "checked_failures_block_completion": True,
                },
                evidence=package_source_status.get("path"),
            ),
            _contract_gate(
                "package_command_manifest_valid_when_checked",
                package_command_manifest_status.get("passed") is not False,
                actual={
                    "checked": package_command_manifest_status.get("checked"),
                    "failed_checks": package_command_manifest_status.get("failed_checks"),
                },
                expected={
                    "failed_checks": [],
                    "checked_failures_block_completion": True,
                },
                evidence=package_command_manifest_status.get("path"),
            ),
            _contract_gate(
                "package_python_metadata_valid_when_checked",
                package_metadata_status.get("passed") is not False,
                actual={
                    "checked": package_metadata_status.get("checked"),
                    "requires_python": package_metadata_status.get("requires_python"),
                },
                expected={
                    "requires_python": UNSLOTH_PYTHON_RANGE,
                    "checked_failures_block_completion": True,
                },
                evidence=package_metadata_status.get("path"),
            ),
        ]
    )
    missing_or_failed = [gate for gate in gates if not gate["passed"]]
    human_usefulness_status = _human_usefulness_contract_status(
        human_study_suite_path,
        coverage_paths=human_study_coverage_paths,
    )
    next_actions = _full_eval_next_actions(
        run,
        stage_status,
        report_status,
        sample_status,
        diagnostics_status,
        repo_hygiene_status,
        windows_readiness_status,
        package_source_status,
        human_usefulness_status,
    )
    remaining_items = [
        {
            "gate": gate["name"],
            "actual": gate.get("actual"),
            "expected": gate.get("expected"),
            "evidence": gate.get("evidence"),
        }
        for gate in missing_or_failed
    ]
    remaining_recovery_plan = _remaining_recovery_plan(
        remaining_items,
        stage_recovery_status=stage_recovery_status,
        diagnostics_status=diagnostics_status,
    )
    report = {
        "mode": "full_eval_contract_status",
        "training_version": TRAINING_VERSION,
        "run_dir": str(run),
        "generated_at": _now(),
        "passed": not missing_or_failed,
        "requested_max_steps": requested_steps,
        "training_eval_summary_status": eval_summary_status,
        "gates": gates,
        "stage_status": stage_status,
        "report_status": report_status,
        "sample_status": sample_status,
        "stage_evidence_summary": _stage_evidence_summary(
            stage_status, report_status, sample_status
        ),
        "diagnostics_status": diagnostics_status,
        "resume_inspection_status": resume_inspection_status,
        "stage_recovery_status": stage_recovery_status,
        "stage_recovery_summary": _stage_recovery_contract_summary(
            stage_recovery_status
        ),
        "repo_hygiene_status": repo_hygiene_status,
        "repo_hygiene_summary": _repo_hygiene_contract_summary(repo_hygiene_status),
        "windows_readiness_status": windows_readiness_status,
        "windows_readiness_summary": _windows_readiness_contract_summary(
            windows_readiness_status
        ),
        "package_source_status": package_source_status,
        "package_source_summary": _package_source_contract_summary(
            package_source_status
        ),
        "package_command_manifest_status": package_command_manifest_status,
        "package_command_manifest_summary": _package_command_manifest_contract_summary(
            package_command_manifest_status
        ),
        "package_metadata_status": package_metadata_status,
        "package_metadata_summary": _package_metadata_contract_summary(
            package_metadata_status
        ),
        "human_usefulness_status": human_usefulness_status,
        "human_usefulness_summary": _human_usefulness_contract_summary(
            human_usefulness_status
        ),
        "next_actions": next_actions,
        "remaining_items": remaining_items,
        "remaining_by_area": _remaining_by_area(remaining_items),
        "remaining_recovery_plan": remaining_recovery_plan,
    }
    report["contract_scorecard"] = _contract_scorecard(report)
    report["contract_scorecard_summary"] = _contract_scorecard_contract_summary(
        report["contract_scorecard"]
    )
    report["contract_reward_summary"] = _contract_reward_summary(report["contract_scorecard"])
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_out_path is not None:
        markdown_out = Path(markdown_out_path)
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(_full_eval_contract_status_markdown(report), encoding="utf-8")
    return report


def create_sample_inspection(
    dataset_path: Path | str,
    *,
    raw_candidates_path: Path | str,
    repaired_candidates_path: Path | str,
    out_path: Path | str,
    model_name: str,
    model_or_adapter_path: Path | str | None = None,
    generation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write raw/repaired sample evals and a human-readable inspection bundle."""

    from semantic_mirror.evaluation import evaluate_model_candidates

    dataset = Path(dataset_path).resolve()
    raw_source = Path(raw_candidates_path).resolve()
    repaired_source = Path(repaired_candidates_path).resolve()
    out = Path(out_path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw_target = out / "raw_candidates.jsonl"
    repaired_target = out / "repaired_candidates.jsonl"
    _copy_if_different(raw_source, raw_target)
    _copy_if_different(repaired_source, repaired_target)

    raw_eval = evaluate_model_candidates(
        dataset,
        raw_target,
        model_name=f"{model_name}-raw",
        out_path=out / "raw_eval.json",
    )
    repaired_eval = evaluate_model_candidates(
        dataset,
        repaired_target,
        model_name=f"{model_name}-repaired",
        out_path=out / "repaired_eval.json",
    )
    raw_rows = _read_jsonl(raw_target)
    repaired_rows = _read_jsonl(repaired_target)
    references = _sample_references(dataset)
    raw_contract_reports = [
        _sample_raw_contract_report(row, references) for row in raw_rows
    ]
    raw_parseability = sum(1 for row in raw_rows if _sample_row_parseable(row))
    raw_generation_cap_hits = sum(1 for row in raw_rows if row.get("hit_generation_cap"))
    raw_schema_valid = sum(1 for row in raw_eval["results"] if row["schema_valid"])
    raw_repair_free_contract = sum(
        1 for report in raw_contract_reports if report["repair_free_contract_valid"]
    )
    raw_exact_identity = sum(1 for report in raw_contract_reports if report["identity_exact"])
    raw_top_level_valid = sum(
        1 for report in raw_contract_reports if report["top_level_keys_valid"]
    )
    raw_compact_shape = sum(1 for report in raw_contract_reports if report["compact_shape_valid"])
    repaired_schema_valid = sum(
        1 for row in repaired_eval["results"] if row["schema_valid"]
    )
    record_ids = [
        str(row.get("dataset_record_id") or row.get("record_id") or row.get("unit_id"))
        for row in repaired_rows
    ]
    manifest_generation_config = generation_config or _sample_generation_config(raw_rows)
    manifest = {
        "mode": "sample_inspection",
        "training_version": TRAINING_VERSION,
        "model_name": model_name,
        "model_or_adapter_path": None
        if model_or_adapter_path is None
        else str(Path(model_or_adapter_path).resolve()),
        "dataset": str(dataset),
        "generated_at": _now(),
        "record_ids": record_ids,
        "generation_config": manifest_generation_config,
        "raw_parseability_count": raw_parseability,
        "raw_generation_cap_hits": raw_generation_cap_hits,
        "raw_schema_validity_count": raw_schema_valid,
        "raw_repair_free_contract_count": raw_repair_free_contract,
        "raw_exact_identity_count": raw_exact_identity,
        "raw_top_level_key_validity_count": raw_top_level_valid,
        "raw_compact_shape_count": raw_compact_shape,
        "repaired_schema_validity_count": repaired_schema_valid,
        "raw_candidate_count": len(raw_rows),
        "repaired_candidate_count": len(repaired_rows),
        "raw_contract_reports": raw_contract_reports,
        "static_faithfulness_score": repaired_eval["metrics"][
            "average_static_faithfulness_score"
        ],
        "raw_static_faithfulness_score": raw_eval["metrics"][
            "average_static_faithfulness_score"
        ],
        "hallucination_penalties": repaired_eval["metrics"]["hallucination_penalties"],
        "raw_hallucination_penalties": raw_eval["metrics"]["hallucination_penalties"],
        "files": {
            "raw_candidates": "raw_candidates.jsonl",
            "repaired_candidates": "repaired_candidates.jsonl",
            "raw_eval": "raw_eval.json",
            "repaired_eval": "repaired_eval.json",
            "inspection_markdown": "sample_inspection.md",
        },
    }
    (out / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "sample_inspection.md").write_text(
        _sample_inspection_markdown(manifest, raw_rows, repaired_rows, raw_eval, repaired_eval),
        encoding="utf-8",
    )
    return manifest


def _positive_records(
    *,
    silver: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    include_silver_when_gold_exists: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in gold:
        promoted = dict(record)
        promoted["split"] = "gold"
        records.append(promoted)
    gold_unit_ids = {record["unit_id"] for record in gold}
    if not gold or include_silver_when_gold_exists:
        records.extend(record for record in silver if record["unit_id"] not in gold_unit_ids)
    return records


def _sft_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    compact_unit = _compact_sir_unit(record["target"]["sir_unit"])
    return {
        "record_id": f"sft-{index}-{record['record_id']}",
        "task": "semantic_ir_generation",
        "messages": [
            {"role": "system", "content": _generation_system_prompt()},
            {"role": "user", "content": _generation_user_prompt(record)},
            {
                "role": "assistant",
                "content": _sir_unit_json(compact_unit),
            },
        ],
        "metadata": _record_metadata(record),
    }


def _contrastive_record(
    positive: dict[str, Any],
    negative: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    payload = {
        "error_labels": negative["expected_error_labels"],
        "verifier_penalties": negative["verifier_report"]["penalties"],
        "corrected_sir_unit": _compact_sir_unit(positive["target"]["sir_unit"]),
    }
    return {
        "record_id": f"contrastive-{index}-{negative['record_id']}",
        "task": "semantic_ir_repair",
        "messages": [
            {"role": "system", "content": _repair_system_prompt()},
            {"role": "user", "content": _repair_user_prompt(positive, negative)},
            {"role": "assistant", "content": json.dumps(payload, sort_keys=True)},
        ],
        "metadata": {
            **_record_metadata(positive),
            "negative_kind": negative["negative_kind"],
            "negative_record_id": negative["record_id"],
        },
    }


def _preference_pair(
    positive: dict[str, Any],
    negative: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    return {
        "record_id": f"preference-{index}-{negative['record_id']}",
        "prompt": _generation_user_prompt(positive),
        "chosen": _sir_unit_json(_compact_sir_unit(positive["target"]["sir_unit"])),
        "rejected": _sir_unit_json(_compact_sir_unit(negative["candidate"]["sir_unit"])),
        "metadata": {
            **_record_metadata(positive),
            "negative_kind": negative["negative_kind"],
            "verifier_report": negative["verifier_report"],
            "preference_reason": "chosen preserves static facts and rejected candidate is verifier-penalized",
        },
    }


def _resolve_teacher_results_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    root = Path(path).resolve()
    if (root / "teacher_preference_pairs.jsonl").exists():
        return root
    nested = root / "teacher_results"
    if (nested / "teacher_preference_pairs.jsonl").exists():
        return nested
    return root


def _teacher_preference_records(root: Path | None) -> list[dict[str, Any]]:
    if root is None:
        return []
    path = root / "teacher_preference_pairs.jsonl"
    if not path.exists():
        return []
    critic_labels = _critic_labels_by_candidate(root)
    records: list[dict[str, Any]] = []
    for index, record in enumerate(_read_jsonl(path)):
        candidate_result_id = record.get("metadata", {}).get("candidate_result_id")
        records.append(
            {
                **record,
                "record_id": f"teacher-{index}-{record.get('record_id', 'preference')}",
                "chosen": _compact_json_text(record.get("chosen")),
                "rejected": _compact_json_text(record.get("rejected")),
                "metadata": {
                    **record.get("metadata", {}),
                    "source": "teacher_results",
                    "teacher_results_path": str(root),
                    "critic_labels": critic_labels.get(candidate_result_id, []),
                },
            }
        )
    return records


def _critic_labels_by_candidate(root: Path) -> dict[str, list[dict[str, Any]]]:
    path = root / "critic_labels.jsonl"
    if not path.exists():
        return {}
    labels: dict[str, list[dict[str, Any]]] = {}
    for record in _read_jsonl(path):
        labels[record["candidate_result_id"]] = record.get("error_labels", [])
    return labels


def _rl_prompt(record: dict[str, Any], index: int) -> dict[str, Any]:
    compact_target = _schema_output_template(record)
    identity = _schema_contract(record)["identity"]
    return {
        "record_id": f"rl-prompt-{index}-{record['record_id']}",
        "prompt": _generation_user_prompt(record),
        "reward_reference": {
            "unit_id": record["unit_id"],
            "language": identity["language"],
            "symbol_type": identity["symbol_type"],
            "name": identity["name"],
            "qualified_name": identity["qualified_name"],
            "static_facts": record["static_facts"],
            "source_path": record["source_path"],
            "source_spans": record["source_spans"],
            "compact_target": compact_target,
            "compact_expected_counts": _compact_expected_counts(compact_target),
        },
        "metadata": _record_metadata(record),
    }


def _generation_system_prompt() -> str:
    return (
        "You generate one valid Semantic Mirror SIR JSON unit. Use every required top-level "
        "schema key exactly once. Preserve only source-backed static facts supplied in the "
        "prompt. Do not invent behavior. Return minified JSON only: begin with {, end with }, "
        "and do not wrap the object in Markdown fences."
    )


def _repair_system_prompt() -> str:
    return (
        "You critique and repair Semantic IR. Label missing or invented behavior using the "
        "verifier evidence, then produce a corrected source-backed SIR unit. Do not reward "
        "brevity when required source facts are missing."
    )


def _generation_user_prompt(record: dict[str, Any]) -> str:
    code_slice = record["code_slice"]
    final_sir = _schema_output_template(record)
    schema_contract = _schema_contract(record)
    schema_contract["compact_expected_counts"] = _compact_expected_counts(final_sir)
    prompt = {
        "task": "emit_one_complete_sir_unit_as_minified_json",
        "profile": record["profile"],
        "zoom": record["zoom"],
        "source_path": record["source_path"],
        "qualified_name": record["qualified_name"],
        "code_slice": {
            "path": code_slice["path"],
            "start_line": code_slice["start_line"],
            "end_line": code_slice["end_line"],
            "text_excerpt": _compact_text(code_slice.get("text", ""), COMPACT_CODE_CHARS),
        },
        "schema_contract": schema_contract,
        "static_facts": _compact_static_facts(record["static_facts"]),
        "static_analysis": _compact_static_analysis(record.get("static_analysis", {})),
        "output_rules": [
            "return the final SIR JSON object between FINAL_SIR_JSON_START and FINAL_SIR_JSON_END",
            "copy final SIR JSON keys and source-backed values exactly",
            "copy identity fields exactly from final SIR JSON; do not shorten unit_id or qualified_name",
            "the compact_expected_counts values are exact upper bounds; do not exceed them",
            "include all required top-level keys even when a list is empty",
            "do not add any top-level key outside schema_contract.required_top_level_keys",
            "the answer must start with {\"unit_id\" and must not include template or marker keys",
            "do not add facts beyond the final SIR JSON; it already contains the compact allowed facts",
            "do not output output_template, safety_report, summary, code_analysis, analysis, labels, or explanations",
            "return only one minified JSON object with no Markdown fence",
        ],
    }
    return (
        json.dumps(prompt, indent=2)
        + "\n\nFINAL_SIR_JSON_START\n"
        + _sir_unit_json(final_sir)
        + "\nFINAL_SIR_JSON_END"
    )


def _repair_user_prompt(positive: dict[str, Any], negative: dict[str, Any]) -> str:
    prompt = {
        "profile": positive["profile"],
        "zoom": positive["zoom"],
        "source_path": positive["source_path"],
        "qualified_name": positive["qualified_name"],
        "static_facts": _compact_static_facts(positive["static_facts"]),
        "static_analysis": _compact_static_analysis(positive.get("static_analysis", {})),
        "candidate": _compact_sir_unit(negative["candidate"]["sir_unit"]),
        "verifier_report": negative["verifier_report"],
        "requested_output": "error labels, verifier penalties, and corrected SIR JSON unit",
    }
    return json.dumps(prompt, indent=2)


def _schema_contract(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "required_top_level_keys": list(SIR_UNIT_TOP_LEVEL_KEYS),
        "claim_shape": {"claim": "string", "confidence": "number", "source_spans": "array"},
        "data_ml_detail_categories": list(DATA_ML_DETAIL_CATEGORIES),
        "identity": {
            "unit_id": record["unit_id"],
            "language": record.get("language", "python"),
            "symbol_type": record["symbol_type"],
            "name": record["target"]["sir_unit"]["name"],
            "qualified_name": record["qualified_name"],
            "source_spans": record["source_spans"],
        },
    }


def _sir_unit_json(unit: dict[str, Any]) -> str:
    return json.dumps(unit, separators=(",", ":"))


def _compact_expected_counts(unit: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for field in SIR_LIST_FIELDS:
        value = unit.get(field, [])
        counts[field] = len(value) if isinstance(value, list) else 0
    data_ml_details = unit.get("data_ml_details", {})
    counts["data_ml_details"] = {
        category: len(data_ml_details.get(category, []))
        if isinstance(data_ml_details, dict) and isinstance(data_ml_details.get(category), list)
        else 0
        for category in DATA_ML_DETAIL_CATEGORIES
    }
    return counts


def _schema_output_template(record: dict[str, Any]) -> dict[str, Any]:
    static_facts = _compact_static_facts(record["static_facts"])
    identity = _schema_contract(record)["identity"]
    return {
        "unit_id": identity["unit_id"],
        "source_spans": identity["source_spans"],
        "language": identity["language"],
        "symbol_type": identity["symbol_type"],
        "name": identity["name"],
        "qualified_name": identity["qualified_name"],
        "algorithm": static_facts["algorithm"],
        "control_flow": static_facts["control_flow"],
        "reads": static_facts["reads"],
        "writes": static_facts["writes"],
        "calls": static_facts["calls"],
        "returns": static_facts["returns"],
        "side_effects": static_facts["side_effects"],
        "failure_modes": static_facts["failure_modes"],
        "state_mutations": static_facts["state_mutations"],
        "external_dependencies": static_facts["external_dependencies"],
        "data_ml_details": static_facts["data_ml_details"],
        "hazards": static_facts["hazards"],
        "uncertainty": static_facts["uncertainty"],
        "confidence": static_facts["algorithm"].get("confidence", 0.7),
    }


def _compact_sir_unit(unit: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "unit_id": unit.get("unit_id"),
        "source_spans": unit.get("source_spans", []),
        "language": unit.get("language", "python"),
        "symbol_type": unit.get("symbol_type"),
        "name": unit.get("name"),
        "qualified_name": unit.get("qualified_name"),
        "algorithm": _compact_claim(unit.get("algorithm", {})),
        "control_flow": _compact_claims(unit.get("control_flow", []), "control_flow"),
        "reads": _compact_claims(unit.get("reads", []), "reads"),
        "writes": _compact_claims(unit.get("writes", []), "writes"),
        "calls": _compact_claims(unit.get("calls", []), "calls"),
        "returns": _compact_claims(unit.get("returns", []), "returns"),
        "side_effects": _compact_claims(unit.get("side_effects", []), "side_effects"),
        "failure_modes": _compact_claims(unit.get("failure_modes", []), "failure_modes"),
        "state_mutations": _compact_claims(unit.get("state_mutations", []), "state_mutations"),
        "external_dependencies": _compact_claims(
            unit.get("external_dependencies", []), "external_dependencies"
        ),
        "data_ml_details": _compact_data_ml_details(unit.get("data_ml_details", {})),
        "hazards": _compact_claims(unit.get("hazards", []), "hazards"),
        "uncertainty": _compact_claims(unit.get("uncertainty", []), "uncertainty"),
        "confidence": unit.get("confidence", 0.7),
    }
    return compact


def _compact_static_facts(static_facts: dict[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": _compact_claim(static_facts.get("algorithm", {})),
        "control_flow": _compact_claims(static_facts.get("control_flow", []), "control_flow"),
        "reads": _compact_claims(static_facts.get("reads", []), "reads"),
        "writes": _compact_claims(static_facts.get("writes", []), "writes"),
        "calls": _compact_claims(static_facts.get("calls", []), "calls"),
        "returns": _compact_claims(static_facts.get("returns", []), "returns"),
        "side_effects": _compact_claims(static_facts.get("side_effects", []), "side_effects"),
        "failure_modes": _compact_claims(static_facts.get("failure_modes", []), "failure_modes"),
        "state_mutations": _compact_claims(static_facts.get("state_mutations", []), "state_mutations"),
        "external_dependencies": _compact_claims(
            static_facts.get("external_dependencies", []), "external_dependencies"
        ),
        "data_ml_details": _compact_data_ml_details(static_facts.get("data_ml_details", {})),
        "hazards": _compact_claims(static_facts.get("hazards", []), "hazards"),
        "uncertainty": _compact_claims(static_facts.get("uncertainty", []), "uncertainty"),
    }


def _compact_claims(claims: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    limit = COMPACT_FACT_LIMITS.get(field, 4)
    return [_compact_claim(claim) for claim in claims[:limit]]


def _compact_data_ml_details(details: dict[str, Any]) -> dict[str, Any]:
    return {
        category: [
            _compact_claim(claim) for claim in details.get(category, [])[:COMPACT_DATA_ML_LIMIT]
        ]
        for category in DATA_ML_DETAIL_CATEGORIES
    }


def _compact_claim(claim: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(claim, dict):
        return {"claim": "", "confidence": 0.0, "source_spans": []}
    compact = {
        "claim": claim.get("claim", ""),
        "confidence": claim.get("confidence", 0.7),
        "source_spans": claim.get("source_spans", [])[:1],
    }
    for key in (
        "name",
        "value",
        "target",
        "kind",
        "call",
        "module",
        "imported",
        "alias",
        "exception",
        "predicate",
        "branch_count",
    ):
        if key in claim:
            compact[key] = claim[key]
    return compact


def _compact_json_text(text: str | None) -> str:
    if not text:
        return "{}"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, dict):
        return json.dumps(_compact_sir_unit(parsed), sort_keys=True)
    return text


def _compact_static_analysis(static_analysis: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(static_analysis, dict):
        return {}
    compact = {
        "backend": static_analysis.get("backend"),
        "language": static_analysis.get("language"),
    }
    for key in ("imports", "symbols", "calls", "writes", "returns"):
        value = static_analysis.get(key)
        if isinstance(value, list):
            compact[key] = value[:8]
    return {key: value for key, value in compact.items() if value is not None}


def _compact_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head].rstrip() + "\n# ... omitted for compact training prompt ...\n" + text[-tail:].lstrip()


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "record_id": record["record_id"],
        "source_path": record["source_path"],
        "unit_id": record["unit_id"],
        "qualified_name": record["qualified_name"],
        "symbol_type": record["symbol_type"],
        "language": record.get("language", "python"),
        "profile": record["profile"],
        "zoom": record["zoom"],
        "priority_score": record.get("priority_score"),
        "priority_reasons": record.get("priority_reasons", []),
    }
    for key in ("source_repo_id", "source_repo_path", "source_repo_location"):
        if key in record:
            metadata[key] = record[key]
    return metadata


def _sft_config(base_model: str) -> dict[str, Any]:
    is_qwen35_or_newer = "qwen3.5" in base_model.lower() or "qwen3.6" in base_model.lower()
    return {
        "base_model": base_model,
        "method": "bf16 LoRA" if is_qwen35_or_newer else "QLoRA",
        "model_size_target": "Qwen3.5/3.6 local-fit target",
        "load_in_4bit": not is_qwen35_or_newer,
        "load_in_16bit": is_qwen35_or_newer,
        "lora": {
            "r": 16,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "training": {
            "learning_rate": 0.0002,
            "warmup_ratio": 0.03,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 2048,
            "packing": False,
        },
        "inputs": {
            "sft_jsonl": "sft.jsonl",
            "contrastive_repair_jsonl": "contrastive_repair.jsonl",
        },
    }


def _reward_config() -> dict[str, Any]:
    return {
        "objective": "faithfulness_first_compactness_second",
        "positive_rewards": {
            "preserved_call": 1,
            "preserved_return_variant": 1,
            "preserved_write_or_state_mutation": 1,
            "preserved_source_backed_failure_mode": 1,
            "preserved_control_flow": 1,
            "preserved_data_ml_detail": 1,
        },
        "penalties": {
            "invented_call_write_error_or_behavior": -2,
            "claim_without_valid_source_evidence": -3,
            "missing_required_static_fact": -1,
        },
        "inputs": {
            "preference_pairs_jsonl": "preference_pairs.jsonl",
            "rl_prompts_jsonl": "rl_prompts.jsonl",
        },
        "guardrails": [
            "never reward brevity when required source facts are missing",
            "reject hallucinated side effects",
            "prefer low-confidence or unsupported markers over silent omission",
        ],
    }


def _training_readme() -> str:
    return """# Semantic Mirror Training Batch

This directory is generated by `semantic-mirror train prepare`.

- `sft.jsonl` contains chat-style supervised examples for faithful SIR generation.
- `contrastive_repair.jsonl` teaches the model to label and repair verifier-rejected IR.
- `preference_pairs.jsonl` pairs source-backed SIR units against hard negatives for preference/RL training.
- `rl_prompts.jsonl` contains policy prompts plus static facts used by deterministic rewards.
- `unsloth_sft_config.json` and `rl_reward_config.json` capture the LoRA and reward defaults.
- `run_unsloth_sft.py` is an executable Unsloth/TRL SFT entrypoint.
- `run_preference_dpo.py` is an executable TRL DPO preference-training entrypoint.
- `run_reward_rl.py` is an executable Unsloth policy-gradient entrypoint using deterministic rewards.
- `generate_sir_candidates.py` exports model generations as candidate JSONL for held-out evaluation.
- `score_sir_candidates.py` scores JSONL candidate generations with deterministic rewards.

These artifacts prepare training data; they do not claim that a model has been trained.
"""


def _copy_training_artifacts(root: Path, target: Path) -> None:
    shutil.copytree(
        root,
        target,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.log",
            "outputs",
            "wandb",
            ".ipynb_checkpoints",
        ),
    )


def _write_project_metadata(out: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        pyproject_text = pyproject.read_text(encoding="utf-8")
        pyproject_text = pyproject_text.replace(
            'requires-python = ">=3.11"',
            f'requires-python = "{UNSLOTH_PYTHON_RANGE}"',
        )
        (out / "pyproject.toml").write_text(pyproject_text, encoding="utf-8")
        return
    (out / "pyproject.toml").write_text(
        """[project]
name = "semantic-mirror-runtime"
version = "0.1.0"
requires-python = ">=3.11,<3.14"
dependencies = [
  "tree-sitter>=0.25.2",
  "tree-sitter-python>=0.25.0",
]

[tool.setuptools.packages.find]
where = ["src"]
""",
        encoding="utf-8",
    )


def _training_requirements() -> str:
    return """-e .
accelerate
bitsandbytes
datasets
peft
torch
torchvision
transformers>=5.0.0
trl
mergekit
llm-blender
weave
pillow
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
wandb
"""


def _training_env_example() -> str:
    return """# Copy this to .env on the GPU machine if needed.
# Do not commit real secrets.
HF_TOKEN=
HUGGINGFACE_HUB_TOKEN=
WANDB_API_KEY=
WANDB_PROJECT=semantic-mirror
WANDB_ENTITY=
"""


def _training_environment_guide(audit: dict[str, Any]) -> str:
    failed = [check["name"] for check in audit["checks"] if check["required"] and not check["passed"]]
    failed_text = ", ".join(failed) if failed else "none"
    return f"""# Training Environment

This bundle intentionally excludes `.env` and API key values. Set `HF_TOKEN` or
`HUGGINGFACE_HUB_TOKEN` on the target machine before model download, and set
`WANDB_API_KEY` only if experiment logging is wanted.

Current-machine readiness is recorded in `audit/current_environment.json`.
Required checks failed on the packaging machine: {failed_text}.

The default target is a CUDA Linux or WSL runtime suitable for Unsloth LoRA on
the configured Qwen3-family model. Use Python {UNSLOTH_PYTHON_RANGE}; Python 3.14 is intentionally
flagged as not launch-ready for this Unsloth target. A failed local audit does
not invalidate this bundle; it means the bundle should be moved to a machine
that satisfies the audit gates.

Bootstrap helpers:

```bash
bash setup/bootstrap_linux_cuda.sh
```

```powershell
powershell -ExecutionPolicy Bypass -File setup/bootstrap_wsl_ubuntu.ps1
```
"""


def _package_launch_commands() -> dict[str, str]:
    return {
        "bootstrap_linux_cuda": "bash setup/bootstrap_linux_cuda.sh",
        "bootstrap_wsl_ubuntu": "powershell -ExecutionPolicy Bypass -File setup/bootstrap_wsl_ubuntu.ps1",
        "wsl_smoke_chain": (
            "powershell -ExecutionPolicy Bypass -File launch/run_wsl_smoke_chain.ps1 "
            "-HeldOutDataset <windows_dataset_dir>"
        ),
        "install": "python -m pip install --upgrade pip && python -m pip install -r requirements-training.txt",
        "validate": (
            "PYTHONPATH=src python -m semantic_mirror.cli train validate training "
            "--out outputs/validation_report.json"
        ),
        "audit": "PYTHONPATH=src python -m semantic_mirror.cli train audit training --out outputs/audit.json",
        "sft": (
            "SFT_MAX_STEPS=2000 SFT_SEED=42 python training/run_unsloth_sft.py "
            "--training-dir training --output-dir outputs/semantic-mirror-sft "
            "--max-steps $SFT_MAX_STEPS --seed $SFT_SEED"
        ),
        "dpo": (
            "DPO_MAX_STEPS=800 DPO_SEED=43 python training/run_preference_dpo.py --training-dir training "
            "--model-name-or-path outputs/semantic-mirror-sft --output-dir outputs/semantic-mirror-dpo "
            "--max-steps $DPO_MAX_STEPS --seed $DPO_SEED"
        ),
        "rl": (
            "RL_MAX_STEPS=1000 RL_SEED=44 python training/run_reward_rl.py --training-dir training "
            "--model-name-or-path outputs/semantic-mirror-dpo --output-dir outputs/semantic-mirror-rl "
            "--max-steps $RL_MAX_STEPS --seed $RL_SEED --schema-prefix-mode schema-scaffold"
        ),
        "full_training_eval": (
            "HELD_OUT_DATASET=<dataset_dir> "
            "BASELINE_CANDIDATES=<teacher_results_dir>/teacher_candidates.jsonl "
            "bash launch/run_full_training_eval.sh"
        ),
        "inspect_full_training_eval_resume": (
            "SFT_MAX_STEPS=300 DPO_MAX_STEPS=120 RL_MAX_STEPS=120 "
            "REUSE_STAGE_OUTPUTS=1 DPO_RESUME_FROM_CHECKPOINT=<checkpoint_dir> "
            "bash launch/inspect_full_training_eval_resume.sh"
        ),
        "inspect_resume": (
            "PYTHONPATH=src python -m semantic_mirror.cli train inspect-resume outputs "
            "--sft-steps 300 --dpo-steps 120 --rl-steps 120 "
            "--reuse-stage-outputs --dpo-resume-from-checkpoint <checkpoint_dir> "
            "--out outputs/full_training_eval_resume_inspection.json "
            "--markdown-out outputs/full_training_eval_resume_inspection.md"
        ),
        "smoke_chain": "HELD_OUT_DATASET=<dataset_dir> bash launch/run_smoke_chain.sh",
        "generate_candidates": (
            "python training/generate_sir_candidates.py --training-dir training "
            "--model-name-or-path outputs/semantic-mirror-rl "
            "--out outputs/samples/repaired_candidates.jsonl "
            "--raw-out outputs/samples/raw_candidates.jsonl "
            "--repaired-out outputs/samples/repaired_candidates.jsonl "
            "--generation-mode full-json "
            "--schema-prefix-mode schema-scaffold"
        ),
        "score_candidates": (
            "PYTHONPATH=src python training/score_sir_candidates.py "
            "--repo <held_out_repo_path> --candidates outputs/samples/repaired_candidates.jsonl --out outputs/candidate_scores.jsonl"
        ),
        "eval_candidates": (
            "PYTHONPATH=src python -m semantic_mirror.cli eval candidates <dataset_dir> "
            "--candidates outputs/samples/repaired_candidates.jsonl --model-name semantic-mirror-rl --out outputs/rl_eval.json"
        ),
        "inspect_samples": (
            "PYTHONPATH=src python -m semantic_mirror.cli train inspect-samples <dataset_dir> "
            "--raw-candidates outputs/samples/raw_candidates.jsonl "
            "--repaired-candidates outputs/samples/repaired_candidates.jsonl "
            "--out outputs/samples/rl --model-name semantic-mirror-rl "
            "--model-or-adapter-path outputs/semantic-mirror-rl"
        ),
        "report": "PYTHONPATH=src python -m semantic_mirror.cli train report outputs --out outputs/diagnostics",
        "source_freshness": (
            "PYTHONPATH=src python -m semantic_mirror.cli train source-freshness . "
            "--repo-root <repo_root> --out source_freshness.json --markdown-out source_freshness.md"
        ),
        "contract_status": (
            "PYTHONPATH=src python -m semantic_mirror.cli train contract-status outputs "
            "--sft-steps $SFT_MAX_STEPS --dpo-steps $DPO_MAX_STEPS --rl-steps $RL_MAX_STEPS "
            "--windows-audit ${WINDOWS_AUDIT:-audit/current_environment.json} "
            "--wsl-smoke-manifest ${WSL_SMOKE_MANIFEST:-outputs/smoke-chain-wsl/smoke_chain_manifest.json} "
            "--package-source-freshness source_freshness.json "
            "--out outputs/contract_status.json --markdown-out outputs/contract_status.md"
        ),
        "compare_sft": (
            "PYTHONPATH=src python -m semantic_mirror.cli eval model-compare "
            "outputs/baseline_eval.json outputs/sft_eval.json --stage sft --out outputs/sft_vs_baseline.json"
        ),
        "compare_sft_raw": (
            "PYTHONPATH=src python -m semantic_mirror.cli eval model-compare "
            "outputs/baseline_eval.json outputs/sft_raw_eval.json --stage sft "
            "--out outputs/sft_raw_vs_baseline.json"
        ),
        "compare_dpo": (
            "PYTHONPATH=src python -m semantic_mirror.cli eval model-compare "
            "outputs/sft_eval.json outputs/dpo_eval.json --stage dpo --out outputs/dpo_vs_sft.json"
        ),
        "compare_dpo_raw": (
            "PYTHONPATH=src python -m semantic_mirror.cli eval model-compare "
            "outputs/sft_raw_eval.json outputs/dpo_raw_eval.json --stage dpo "
            "--out outputs/dpo_raw_vs_sft.json"
        ),
        "compare_rl": (
            "PYTHONPATH=src python -m semantic_mirror.cli eval model-compare "
            "outputs/sft_eval.json outputs/rl_eval.json --stage rl --out outputs/rl_vs_sft.json"
        ),
        "compare_rl_raw": (
            "PYTHONPATH=src python -m semantic_mirror.cli eval model-compare "
            "outputs/sft_raw_eval.json outputs/rl_raw_eval.json --stage rl "
            "--out outputs/rl_raw_vs_sft.json"
        ),
    }


def _package_launch_command_manifest(commands: dict[str, str]) -> dict[str, Any]:
    categories = {
        "bootstrap_linux_cuda": "setup",
        "bootstrap_wsl_ubuntu": "setup",
        "install": "setup",
        "validate": "validation",
        "audit": "validation",
        "wsl_smoke_chain": "training",
        "sft": "training",
        "dpo": "training",
        "rl": "training",
        "full_training_eval": "training",
        "smoke_chain": "training",
        "inspect_full_training_eval_resume": "inspection",
        "inspect_resume": "inspection",
        "generate_candidates": "generation",
        "score_candidates": "evaluation",
        "eval_candidates": "evaluation",
        "inspect_samples": "inspection",
        "report": "diagnostics",
        "source_freshness": "status",
        "contract_status": "status",
        "compare_sft": "evaluation",
        "compare_sft_raw": "evaluation",
        "compare_dpo": "evaluation",
        "compare_dpo_raw": "evaluation",
        "compare_rl": "evaluation",
        "compare_rl_raw": "evaluation",
    }
    training_commands = {
        "wsl_smoke_chain",
        "sft",
        "dpo",
        "rl",
        "full_training_eval",
        "smoke_chain",
    }
    return {
        "schema_version": 1,
        "commands": {
            name: {
                "command": command,
                "category": categories.get(name, "other"),
                "launches_training": name in training_commands,
            }
            for name, command in commands.items()
        },
    }


def _write_launch_scripts(launch_target: Path) -> None:
    scripts = {
        "run_sft.sh": """#!/usr/bin/env bash
set -euo pipefail
SFT_MAX_STEPS="${SFT_MAX_STEPS:-2000}"
SFT_SEED="${SFT_SEED:-42}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-10}"
SFT_SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-3}"
SFT_RESUME_ARG=()
if [[ -n "${SFT_RESUME_FROM_CHECKPOINT:-}" ]]; then
  SFT_RESUME_ARG=(--resume-from-checkpoint "$SFT_RESUME_FROM_CHECKPOINT")
fi
python training/run_unsloth_sft.py --training-dir training --output-dir outputs/semantic-mirror-sft --max-steps "$SFT_MAX_STEPS" --seed "$SFT_SEED" --save-steps "$SFT_SAVE_STEPS" --save-total-limit "$SFT_SAVE_TOTAL_LIMIT" "${SFT_RESUME_ARG[@]}"
""",
        "run_dpo.sh": """#!/usr/bin/env bash
set -euo pipefail
DPO_MAX_STEPS="${DPO_MAX_STEPS:-800}"
DPO_SEED="${DPO_SEED:-43}"
DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-10}"
DPO_SAVE_TOTAL_LIMIT="${DPO_SAVE_TOTAL_LIMIT:-3}"
DPO_RESUME_ARG=()
if [[ -n "${DPO_RESUME_FROM_CHECKPOINT:-}" ]]; then
  DPO_RESUME_ARG=(--resume-from-checkpoint "$DPO_RESUME_FROM_CHECKPOINT")
fi
python training/run_preference_dpo.py --training-dir training --model-name-or-path outputs/semantic-mirror-sft --output-dir outputs/semantic-mirror-dpo --max-steps "$DPO_MAX_STEPS" --seed "$DPO_SEED" --save-steps "$DPO_SAVE_STEPS" --save-total-limit "$DPO_SAVE_TOTAL_LIMIT" "${DPO_RESUME_ARG[@]}"
""",
        "run_rl.sh": """#!/usr/bin/env bash
set -euo pipefail
RL_MAX_STEPS="${RL_MAX_STEPS:-1000}"
RL_SEED="${RL_SEED:-44}"
SCHEMA_PREFIX_MODE="${SCHEMA_PREFIX_MODE:-schema-scaffold}"
python training/run_reward_rl.py --training-dir training --model-name-or-path outputs/semantic-mirror-dpo --output-dir outputs/semantic-mirror-rl --max-steps "$RL_MAX_STEPS" --seed "$RL_SEED" --schema-prefix-mode "$SCHEMA_PREFIX_MODE"
""",
        "generate_candidates.sh": """#!/usr/bin/env bash
set -euo pipefail
SCHEMA_PREFIX_MODE="${SCHEMA_PREFIX_MODE:-schema-scaffold}"
GENERATION_MODE="${GENERATION_MODE:-full-json}"
FIELD_MAX_NEW_TOKENS="${FIELD_MAX_NEW_TOKENS:-384}"
FIELD_TARGET_MODE="${FIELD_TARGET_MODE:-compact}"
FIELD_TARGET_LIMIT="${FIELD_TARGET_LIMIT:-0}"
FIELD_TARGET_MAX_CHUNKS="${FIELD_TARGET_MAX_CHUNKS:-1}"
FIELD_TARGET_CHUNK_FIELDS="${FIELD_TARGET_CHUNK_FIELDS:-}"
FIELD_OBJECT_PREFIX_MODE="${FIELD_OBJECT_PREFIX_MODE:-off}"
FAITHFULNESS_REPAIR_MODE="${FAITHFULNESS_REPAIR_MODE:-schema-only}"
mkdir -p outputs/samples
python training/generate_sir_candidates.py --training-dir training \
  --model-name-or-path outputs/semantic-mirror-rl \
  --out outputs/samples/repaired_candidates.jsonl \
  --raw-out outputs/samples/raw_candidates.jsonl \
  --repaired-out outputs/samples/repaired_candidates.jsonl \
  --generation-mode "$GENERATION_MODE" \
  --field-max-new-tokens "$FIELD_MAX_NEW_TOKENS" \
  --field-target-mode "$FIELD_TARGET_MODE" \
  --field-target-limit "$FIELD_TARGET_LIMIT" \
  --field-target-max-chunks "$FIELD_TARGET_MAX_CHUNKS" \
  --field-target-chunk-fields "$FIELD_TARGET_CHUNK_FIELDS" \
  --field-object-prefix-mode "$FIELD_OBJECT_PREFIX_MODE" \
  --faithfulness-repair-mode "$FAITHFULNESS_REPAIR_MODE" \
  --schema-prefix-mode "$SCHEMA_PREFIX_MODE"
""",
        "score_candidates.sh": """#!/usr/bin/env bash
set -euo pipefail
: "${HELD_OUT_REPO:?set HELD_OUT_REPO to the source repo path}"
PYTHONPATH=src python training/score_sir_candidates.py --repo "$HELD_OUT_REPO" --candidates outputs/samples/repaired_candidates.jsonl --out outputs/candidate_scores.jsonl
""",
        "run_smoke_chain.sh": """#!/usr/bin/env bash
set -euo pipefail

: "${HELD_OUT_DATASET:?set HELD_OUT_DATASET to a dataset directory containing manifest.json}"

export SMOKE_OUT="${SMOKE_OUT:-outputs/smoke-chain}"
SFT_SMOKE_STEPS="${SFT_SMOKE_STEPS:-1}"
DPO_SMOKE_STEPS="${DPO_SMOKE_STEPS:-1}"
RL_SMOKE_STEPS="${RL_SMOKE_STEPS:-1}"
SMOKE_MAX_PROMPTS="${SMOKE_MAX_PROMPTS:-1}"
SMOKE_SCHEMA_PREFIX_MODE="${SMOKE_SCHEMA_PREFIX_MODE:-schema-scaffold}"
SMOKE_GENERATION_MODE="${SMOKE_GENERATION_MODE:-full-json}"
SMOKE_FIELD_MAX_NEW_TOKENS="${SMOKE_FIELD_MAX_NEW_TOKENS:-384}"
SMOKE_FIELD_TARGET_MODE="${SMOKE_FIELD_TARGET_MODE:-compact}"
SMOKE_FIELD_TARGET_LIMIT="${SMOKE_FIELD_TARGET_LIMIT:-0}"
SMOKE_FIELD_TARGET_MAX_CHUNKS="${SMOKE_FIELD_TARGET_MAX_CHUNKS:-1}"
SMOKE_FIELD_TARGET_CHUNK_FIELDS="${SMOKE_FIELD_TARGET_CHUNK_FIELDS:-}"
SMOKE_FIELD_OBJECT_PREFIX_MODE="${SMOKE_FIELD_OBJECT_PREFIX_MODE:-off}"
SMOKE_FAITHFULNESS_REPAIR_MODE="${SMOKE_FAITHFULNESS_REPAIR_MODE:-schema-only}"
SFT_SEED="${SFT_SEED:-42}"
DPO_SEED="${DPO_SEED:-43}"
RL_SEED="${RL_SEED:-44}"

mkdir -p "$SMOKE_OUT/samples"
PYTHONPATH=src python -m semantic_mirror.cli train validate training --out "$SMOKE_OUT/validation_report.json" > "$SMOKE_OUT/validate_summary.json"
PYTHONPATH=src python -m semantic_mirror.cli train audit training --out "$SMOKE_OUT/audit.json" > "$SMOKE_OUT/audit_summary.json"

python training/run_unsloth_sft.py \
  --training-dir training \
  --output-dir "$SMOKE_OUT/semantic-mirror-sft" \
  --max-steps "$SFT_SMOKE_STEPS" \
  --seed "$SFT_SEED"

generate_and_inspect() {
  local stage="$1"
  local model_path="$2"
  python training/generate_sir_candidates.py \
    --training-dir training \
    --model-name-or-path "$model_path" \
    --out "$SMOKE_OUT/samples/${stage}_repaired_candidates.jsonl" \
    --raw-out "$SMOKE_OUT/samples/${stage}_raw_candidates.jsonl" \
    --repaired-out "$SMOKE_OUT/samples/${stage}_repaired_candidates.jsonl" \
    --max-prompts "$SMOKE_MAX_PROMPTS" \
    --generation-mode "$SMOKE_GENERATION_MODE" \
    --field-max-new-tokens "$SMOKE_FIELD_MAX_NEW_TOKENS" \
    --field-target-mode "$SMOKE_FIELD_TARGET_MODE" \
    --field-target-limit "$SMOKE_FIELD_TARGET_LIMIT" \
    --field-target-max-chunks "$SMOKE_FIELD_TARGET_MAX_CHUNKS" \
    --field-target-chunk-fields "$SMOKE_FIELD_TARGET_CHUNK_FIELDS" \
    --field-object-prefix-mode "$SMOKE_FIELD_OBJECT_PREFIX_MODE" \
    --faithfulness-repair-mode "$SMOKE_FAITHFULNESS_REPAIR_MODE" \
    --schema-prefix-mode "$SMOKE_SCHEMA_PREFIX_MODE"
  PYTHONPATH=src python -m semantic_mirror.cli train inspect-samples "$HELD_OUT_DATASET" \
    --raw-candidates "$SMOKE_OUT/samples/${stage}_raw_candidates.jsonl" \
    --repaired-candidates "$SMOKE_OUT/samples/${stage}_repaired_candidates.jsonl" \
    --out "$SMOKE_OUT/samples/${stage}" \
    --model-name "semantic-mirror-${stage}-smoke" \
    --model-or-adapter-path "$model_path"
}

generate_and_inspect sft "$SMOKE_OUT/semantic-mirror-sft"

python training/run_preference_dpo.py \
  --training-dir training \
  --model-name-or-path "$SMOKE_OUT/semantic-mirror-sft" \
  --output-dir "$SMOKE_OUT/semantic-mirror-dpo" \
  --max-steps "$DPO_SMOKE_STEPS" \
  --seed "$DPO_SEED"
generate_and_inspect dpo "$SMOKE_OUT/semantic-mirror-dpo"

python training/run_reward_rl.py \
  --training-dir training \
  --model-name-or-path "$SMOKE_OUT/semantic-mirror-dpo" \
  --output-dir "$SMOKE_OUT/semantic-mirror-rl" \
  --max-steps "$RL_SMOKE_STEPS" \
  --seed "$RL_SEED" \
  --schema-prefix-mode "$SMOKE_SCHEMA_PREFIX_MODE"
generate_and_inspect rl "$SMOKE_OUT/semantic-mirror-rl"

PYTHONPATH=src python -m semantic_mirror.cli train report "$SMOKE_OUT" --out "$SMOKE_OUT/diagnostics"

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["SMOKE_OUT"])
stages = ["sft", "dpo", "rl"]

def rate(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator

def sample_rollup(path):
    if not path.exists():
        return {"exists": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_candidate_count = data.get("raw_candidate_count")
    raw_parseability_rate = rate(data.get("raw_parseability_count"), raw_candidate_count)
    raw_schema_validity_rate = rate(data.get("raw_schema_validity_count"), raw_candidate_count)
    raw_repair_free_rate = rate(data.get("raw_repair_free_contract_count"), raw_candidate_count)
    repaired_candidate_count = data.get("repaired_candidate_count")
    repaired_schema_validity_rate = rate(
        data.get("repaired_schema_validity_count"),
        repaired_candidate_count,
    )
    return {
        "exists": True,
        "raw_candidate_count": raw_candidate_count,
        "raw_parseability_count": data.get("raw_parseability_count"),
        "raw_parseability_rate": raw_parseability_rate,
        "raw_parseability_gate_passed": (
            raw_parseability_rate is not None and raw_parseability_rate >= 0.8
        ),
        "raw_schema_validity_count": data.get("raw_schema_validity_count"),
        "raw_schema_validity_rate": raw_schema_validity_rate,
        "raw_schema_validity_gate_passed": (
            raw_schema_validity_rate is not None and raw_schema_validity_rate >= 0.8
        ),
        "raw_repair_free_contract_count": data.get("raw_repair_free_contract_count"),
        "raw_repair_free_contract_rate": raw_repair_free_rate,
        "raw_repair_free_contract_gate_passed": (
            raw_repair_free_rate is not None and raw_repair_free_rate >= 0.5
        ),
        "raw_generation_cap_hits": data.get("raw_generation_cap_hits"),
        "repaired_candidate_count": repaired_candidate_count,
        "repaired_schema_validity_count": data.get("repaired_schema_validity_count"),
        "repaired_schema_validity_rate": repaired_schema_validity_rate,
        "repaired_schema_validity_gate_passed": repaired_schema_validity_rate == 1.0,
        "static_faithfulness_score": data.get("static_faithfulness_score"),
        "hallucination_penalties": data.get("hallucination_penalties"),
    }

manifest = {
    "mode": "smoke_chain",
    "smoke_out": str(root),
    "training_dir": "training",
    "stages": {},
    "samples": {},
    "diagnostics": str(root / "diagnostics" / "training_summary.json"),
}
for stage in stages:
    stage_dir = root / f"semantic-mirror-{stage}"
    stage_manifest = stage_dir / "training_stage_manifest.json"
    manifest["stages"][stage] = {
        "output_dir": str(stage_dir),
        "stage_manifest": str(stage_manifest),
        "stage_manifest_exists": stage_manifest.exists(),
    }
    sample_manifest = root / "samples" / stage / "sample_manifest.json"
    manifest["samples"][stage] = {
        "raw_candidates": str(root / "samples" / f"{stage}_raw_candidates.jsonl"),
        "repaired_candidates": str(root / "samples" / f"{stage}_repaired_candidates.jsonl"),
        "sample_manifest": str(sample_manifest),
        "sample_manifest_exists": sample_manifest.exists(),
        "sample_rollup": sample_rollup(sample_manifest),
    }
diagnostics = root / "diagnostics" / "training_summary.json"
manifest["diagnostics_exists"] = diagnostics.exists()
manifest["all_stage_outputs_exist"] = all(
    item["stage_manifest_exists"] for item in manifest["stages"].values()
)
manifest["all_sample_manifests_exist"] = all(
    item["sample_manifest_exists"] for item in manifest["samples"].values()
)
manifest["all_repaired_samples_schema_valid"] = all(
    item["sample_rollup"].get("repaired_candidate_count")
    == item["sample_rollup"].get("repaired_schema_validity_count")
    for item in manifest["samples"].values()
)
(root / "smoke_chain_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
PY
""",
        "run_wsl_smoke_chain.ps1": """param(
  [Parameter(Mandatory = $true)]
  [string]$HeldOutDataset,
  [string]$Distro = "Ubuntu",
  [string]$Python = "python3.12",
  [string]$VenvPath = ".venv",
  [string]$SmokeOut = "outputs/smoke-chain-wsl",
  [int]$SftSteps = 1,
  [int]$DpoSteps = 1,
  [int]$RlSteps = 1,
  [int]$MaxPrompts = 1
)

$ErrorActionPreference = "Stop"
$windowsPath = (Resolve-Path ".").Path
$datasetPath = (Resolve-Path $HeldOutDataset).Path
$windowsPathForWsl = $windowsPath -replace '\\\\', '/'
$datasetPathForWsl = $datasetPath -replace '\\\\', '/'
$wslPath = (wsl.exe -d $Distro -- wslpath -a "$windowsPathForWsl").Trim()
$heldOutWsl = (wsl.exe -d $Distro -- wslpath -a "$datasetPathForWsl").Trim()
$scriptPath = Join-Path $windowsPath "launch\\.run_wsl_smoke_chain.generated.sh"

$bashScript = @'
set -euo pipefail
cd '__WSL_PATH__'
VENV_PATH='__VENV_PATH__'
if [ ! -x "$VENV_PATH/bin/python" ]; then
  if [ "$VENV_PATH" != ".venv" ]; then
    echo "Requested VenvPath '$VENV_PATH' does not contain bin/python." >&2
    exit 1
  fi
  PYTHON_BIN='__PYTHON__' bash setup/bootstrap_linux_cuda.sh
fi
mkdir -p '__SMOKE_OUT__'
source "$VENV_PATH/bin/activate"
python - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

info = {
    "mode": "wsl_smoke_chain_environment",
    "wsl_repo_path": "__WSL_PATH__",
    "held_out_dataset": "__HELD_OUT_WSL__",
    "python_executable": sys.executable,
    "venv_path": "__VENV_PATH__",
    "output_path": "__SMOKE_OUT__",
}
try:
    import torch
    info["torch_version"] = getattr(torch, "__version__", None)
    info["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_device_count"] = int(torch.cuda.device_count())
    info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as exc:
    info["cuda_available"] = False
    info["cuda_error"] = str(exc)
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        encoding="utf-8",
    )
    info["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
except Exception as exc:
    info["nvidia_smi_error"] = str(exc)
Path("__SMOKE_OUT__").mkdir(parents=True, exist_ok=True)
Path("__SMOKE_OUT__/wsl_smoke_environment.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(info, indent=2, sort_keys=True))
PY
HELD_OUT_DATASET='__HELD_OUT_WSL__' SMOKE_OUT='__SMOKE_OUT__' SFT_SMOKE_STEPS='__SFT_STEPS__' DPO_SMOKE_STEPS='__DPO_STEPS__' RL_SMOKE_STEPS='__RL_STEPS__' SMOKE_MAX_PROMPTS='__MAX_PROMPTS__' bash launch/run_smoke_chain.sh
'@

$bashScript = $bashScript.Replace("__WSL_PATH__", $wslPath)
$bashScript = $bashScript.Replace("__HELD_OUT_WSL__", $heldOutWsl)
$bashScript = $bashScript.Replace("__VENV_PATH__", $VenvPath)
$bashScript = $bashScript.Replace("__PYTHON__", $Python)
$bashScript = $bashScript.Replace("__SMOKE_OUT__", $SmokeOut)
$bashScript = $bashScript.Replace("__SFT_STEPS__", [string]$SftSteps)
$bashScript = $bashScript.Replace("__DPO_STEPS__", [string]$DpoSteps)
$bashScript = $bashScript.Replace("__RL_STEPS__", [string]$RlSteps)
$bashScript = $bashScript.Replace("__MAX_PROMPTS__", [string]$MaxPrompts)
$bashScript = $bashScript -replace "`r`n", "`n"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($scriptPath, $bashScript, $utf8NoBom)
$scriptPathForWsl = $scriptPath -replace '\\\\', '/'
$scriptWslPath = (wsl.exe -d $Distro -- wslpath -a "$scriptPathForWsl").Trim()
try {
  wsl.exe -d $Distro -- bash "$scriptWslPath"
} finally {
  Remove-Item -LiteralPath $scriptPath -Force -ErrorAction SilentlyContinue
}
""",
        "run_full_training_eval.sh": """#!/usr/bin/env bash
set -euo pipefail

: "${HELD_OUT_DATASET:?set HELD_OUT_DATASET to a dataset directory containing manifest.json}"
: "${BASELINE_CANDIDATES:?set BASELINE_CANDIDATES to baseline candidate JSONL for the held-out dataset, for example teacher_results/teacher_candidates.jsonl}"

mkdir -p outputs/samples outputs/logs
SFT_MAX_STEPS="${SFT_MAX_STEPS:-2000}"
DPO_MAX_STEPS="${DPO_MAX_STEPS:-800}"
RL_MAX_STEPS="${RL_MAX_STEPS:-1000}"
SFT_SEED="${SFT_SEED:-42}"
DPO_SEED="${DPO_SEED:-43}"
RL_SEED="${RL_SEED:-44}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-10}"
DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-10}"
SFT_SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-3}"
DPO_SAVE_TOTAL_LIMIT="${DPO_SAVE_TOTAL_LIMIT:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
EVAL_MAX_PROMPTS="${EVAL_MAX_PROMPTS:-}"
SCHEMA_PREFIX_MODE="${SCHEMA_PREFIX_MODE:-schema-scaffold}"
GENERATION_MODE="${GENERATION_MODE:-full-json}"
FIELD_MAX_NEW_TOKENS="${FIELD_MAX_NEW_TOKENS:-384}"
FIELD_TARGET_MODE="${FIELD_TARGET_MODE:-compact}"
FIELD_TARGET_LIMIT="${FIELD_TARGET_LIMIT:-0}"
FIELD_TARGET_MAX_CHUNKS="${FIELD_TARGET_MAX_CHUNKS:-1}"
FIELD_TARGET_CHUNK_FIELDS="${FIELD_TARGET_CHUNK_FIELDS:-}"
FIELD_OBJECT_PREFIX_MODE="${FIELD_OBJECT_PREFIX_MODE:-off}"
FAITHFULNESS_REPAIR_MODE="${FAITHFULNESS_REPAIR_MODE:-schema-only}"
REUSE_STAGE_OUTPUTS="${REUSE_STAGE_OUTPUTS:-0}"
SFT_RESUME_FROM_CHECKPOINT="${SFT_RESUME_FROM_CHECKPOINT:-}"
DPO_RESUME_FROM_CHECKPOINT="${DPO_RESUME_FROM_CHECKPOINT:-}"
PACKAGE_SOURCE_FRESHNESS="${PACKAGE_SOURCE_FRESHNESS:-source_freshness.json}"
WINDOWS_AUDIT="${WINDOWS_AUDIT:-audit/current_environment.json}"
WSL_SMOKE_MANIFEST="${WSL_SMOKE_MANIFEST:-outputs/smoke-chain-wsl/smoke_chain_manifest.json}"

if [[ -n "${SOURCE_FRESHNESS_REPO_ROOT:-}" ]]; then
  PYTHONPATH=src python -m semantic_mirror.cli train source-freshness . \
    --repo-root "$SOURCE_FRESHNESS_REPO_ROOT" \
    --out "$PACKAGE_SOURCE_FRESHNESS" \
    --markdown-out "${PACKAGE_SOURCE_FRESHNESS%.json}.md"
fi

PYTHONPATH=src python -m semantic_mirror.cli train validate training --out outputs/validation_report.json > outputs/validate_summary.json
PYTHONPATH=src python -m semantic_mirror.cli train audit training --out outputs/audit.json > outputs/audit_summary.json

python - "$HELD_OUT_DATASET" "$BASELINE_CANDIDATES" "outputs/heldout_eval_dataset" "outputs/baseline_candidates_eval.jsonl" "$EVAL_MAX_PROMPTS" <<'PY'
import json
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
baseline = Path(sys.argv[2])
out = Path(sys.argv[3])
baseline_out = Path(sys.argv[4])
max_prompts = int(sys.argv[5]) if sys.argv[5] else None
manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
baseline_rows = [
    json.loads(line)
    for line in baseline.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
candidate_ids = []
seen = set()
for row in baseline_rows:
    for key in ("dataset_record_id", "record_id", "unit_id"):
        value = row.get(key)
        if value and value not in seen:
            candidate_ids.append(value)
            seen.add(value)
            break
if max_prompts is not None:
    candidate_ids = candidate_ids[:max_prompts]
candidate_id_set = set(candidate_ids)

def read_jsonl(name):
    path = dataset / manifest["files"][name]
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

gold = read_jsonl("gold")
silver = read_jsonl("silver")
hard_negative = read_jsonl("hard_negative")

def record_matches(record):
    return record.get("record_id") in candidate_id_set or record.get("unit_id") in candidate_id_set

filtered_gold = [record for record in gold if record_matches(record)]
filtered_silver = [record for record in silver if record_matches(record)]
selected_unit_ids = {record["unit_id"] for record in [*filtered_gold, *filtered_silver]}
filtered_hard_negative = [
    record for record in hard_negative if record.get("positive_unit_id") in selected_unit_ids
]
filtered_baseline = [
    row
    for row in baseline_rows
    if row.get("dataset_record_id") in candidate_id_set
    or row.get("record_id") in candidate_id_set
    or row.get("unit_id") in selected_unit_ids
]

out.mkdir(parents=True, exist_ok=True)
baseline_out.parent.mkdir(parents=True, exist_ok=True)
for name, rows in {
    "gold.jsonl": filtered_gold,
    "silver.jsonl": filtered_silver,
    "hard_negative.jsonl": filtered_hard_negative,
}.items():
    (out / name).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\\n" for row in rows),
        encoding="utf-8",
    )
(out / "review_queue.jsonl").write_text("", encoding="utf-8")
baseline_out.write_text(
    "".join(json.dumps(row, sort_keys=True) + "\\n" for row in filtered_baseline),
    encoding="utf-8",
)
subset_manifest = dict(manifest)
subset_manifest["mode"] = "heldout_eval_subset"
subset_manifest["source_dataset"] = str(dataset)
subset_manifest["baseline_candidates"] = str(baseline)
subset_manifest["files"] = {
    "gold": "gold.jsonl",
    "silver": "silver.jsonl",
    "hard_negative": "hard_negative.jsonl",
    "review_queue": "review_queue.jsonl",
}
subset_manifest["counts"] = {
    **manifest.get("counts", {}),
    "gold_records": len(filtered_gold),
    "silver_records": len(filtered_silver),
    "hard_negative_records": len(filtered_hard_negative),
    "baseline_candidate_records": len(filtered_baseline),
    "expected_units": len(selected_unit_ids),
}
(out / "manifest.json").write_text(
    json.dumps(subset_manifest, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
if not selected_unit_ids:
    raise SystemExit("No held-out records matched BASELINE_CANDIDATES.")
PY

PYTHONPATH=src python -m semantic_mirror.cli train prepare \
  outputs/heldout_eval_dataset \
  --out outputs/heldout_eval_training

generate_and_inspect() {
  local stage="$1"
  local model_path="$2"
  local seed="$3"
  local max_prompts_value="${EVAL_MAX_PROMPTS:-all}"
  local prompt_args=()
  if [[ -n "$EVAL_MAX_PROMPTS" ]]; then
    prompt_args=(--max-prompts "$EVAL_MAX_PROMPTS")
  fi
  local generation_config
  generation_config=$(python - "$stage" "$seed" "$MAX_NEW_TOKENS" "$max_prompts_value" "$SCHEMA_PREFIX_MODE" "$GENERATION_MODE" "$FIELD_MAX_NEW_TOKENS" "$FIELD_TARGET_MODE" "$FIELD_TARGET_LIMIT" "$FIELD_TARGET_MAX_CHUNKS" "$FIELD_TARGET_CHUNK_FIELDS" "$FIELD_OBJECT_PREFIX_MODE" "$FAITHFULNESS_REPAIR_MODE" <<'PY'
import json
import sys

print(json.dumps({
    "stage": sys.argv[1],
    "seed": int(sys.argv[2]),
    "max_new_tokens": int(sys.argv[3]),
    "max_prompts": sys.argv[4],
    "schema_prefix_mode": sys.argv[5],
    "generation_mode": sys.argv[6],
    "field_max_new_tokens": int(sys.argv[7]),
    "field_target_mode": sys.argv[8],
    "field_target_limit": int(sys.argv[9]),
    "field_target_max_chunks": int(sys.argv[10]),
    "field_target_chunk_fields": [item for item in sys.argv[11].split(",") if item],
    "field_object_prefix_mode": sys.argv[12],
    "faithfulness_repair_mode": sys.argv[13],
}))
PY
)
  python training/generate_sir_candidates.py --training-dir training \
    --prompt-file outputs/heldout_eval_training/rl_prompts.jsonl \
    --model-name-or-path "$model_path" \
    --out "outputs/samples/${stage}_repaired_candidates.jsonl" \
    --raw-out "outputs/samples/${stage}_raw_candidates.jsonl" \
    --repaired-out "outputs/samples/${stage}_repaired_candidates.jsonl" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --seed "$seed" \
    --generation-mode "$GENERATION_MODE" \
    --field-max-new-tokens "$FIELD_MAX_NEW_TOKENS" \
    --field-target-mode "$FIELD_TARGET_MODE" \
    --field-target-limit "$FIELD_TARGET_LIMIT" \
    --field-target-max-chunks "$FIELD_TARGET_MAX_CHUNKS" \
    --field-target-chunk-fields "$FIELD_TARGET_CHUNK_FIELDS" \
    --field-object-prefix-mode "$FIELD_OBJECT_PREFIX_MODE" \
    --faithfulness-repair-mode "$FAITHFULNESS_REPAIR_MODE" \
    --schema-prefix-mode "$SCHEMA_PREFIX_MODE" \
    "${prompt_args[@]}"
  PYTHONPATH=src python -m semantic_mirror.cli train inspect-samples outputs/heldout_eval_dataset \
    --raw-candidates "outputs/samples/${stage}_raw_candidates.jsonl" \
    --repaired-candidates "outputs/samples/${stage}_repaired_candidates.jsonl" \
    --out "outputs/samples/${stage}" \
    --model-name "semantic-mirror-${stage}" \
    --model-or-adapter-path "$model_path" \
    --generation-config-json "$generation_config"
  cp "outputs/samples/${stage}/raw_eval.json" "outputs/${stage}_raw_eval.json"
  cp "outputs/samples/${stage}/repaired_eval.json" "outputs/${stage}_eval.json"
}

stage_ready() {
  local stage_dir="$1"
  local requested_steps="$2"
  if [[ "$REUSE_STAGE_OUTPUTS" != "1" || ! -f "${stage_dir}/training_stage_manifest.json" ]]; then
    return 1
  fi
  python - "$stage_dir/training_stage_manifest.json" "$requested_steps" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if manifest.get("max_steps") == int(sys.argv[2]) else 1)
PY
}

if stage_ready outputs/semantic-mirror-sft "$SFT_MAX_STEPS"; then
  echo "Reusing existing SFT stage output at outputs/semantic-mirror-sft"
else
  sft_resume_args=()
  if [[ -n "$SFT_RESUME_FROM_CHECKPOINT" ]]; then
    sft_resume_args=(--resume-from-checkpoint "$SFT_RESUME_FROM_CHECKPOINT")
  fi
  python training/run_unsloth_sft.py --training-dir training --output-dir outputs/semantic-mirror-sft --max-steps "$SFT_MAX_STEPS" --seed "$SFT_SEED" --save-steps "$SFT_SAVE_STEPS" --save-total-limit "$SFT_SAVE_TOTAL_LIMIT" "${sft_resume_args[@]}"
fi
PYTHONPATH=src python -m semantic_mirror.cli eval candidates outputs/heldout_eval_dataset \
  --candidates outputs/baseline_candidates_eval.jsonl \
  --model-name baseline \
  --out outputs/baseline_eval.json || true

generate_and_inspect sft outputs/semantic-mirror-sft "$SFT_SEED"
PYTHONPATH=src python -m semantic_mirror.cli eval model-compare \
  outputs/baseline_eval.json outputs/sft_eval.json \
  --stage sft \
  --out outputs/sft_vs_baseline.json || true
PYTHONPATH=src python -m semantic_mirror.cli eval model-compare \
  outputs/baseline_eval.json outputs/sft_raw_eval.json \
  --stage sft \
  --out outputs/sft_raw_vs_baseline.json || true

if stage_ready outputs/semantic-mirror-dpo "$DPO_MAX_STEPS"; then
  echo "Reusing existing DPO stage output at outputs/semantic-mirror-dpo"
else
  dpo_resume_args=()
  if [[ -n "$DPO_RESUME_FROM_CHECKPOINT" ]]; then
    dpo_resume_args=(--resume-from-checkpoint "$DPO_RESUME_FROM_CHECKPOINT")
  fi
  python training/run_preference_dpo.py --training-dir training \
  --model-name-or-path outputs/semantic-mirror-sft \
  --output-dir outputs/semantic-mirror-dpo \
  --max-steps "$DPO_MAX_STEPS" \
  --seed "$DPO_SEED" \
  --save-steps "$DPO_SAVE_STEPS" \
  --save-total-limit "$DPO_SAVE_TOTAL_LIMIT" \
  "${dpo_resume_args[@]}"
fi
generate_and_inspect dpo outputs/semantic-mirror-dpo "$DPO_SEED"
PYTHONPATH=src python -m semantic_mirror.cli eval model-compare \
  outputs/sft_eval.json outputs/dpo_eval.json \
  --stage dpo \
  --out outputs/dpo_vs_sft.json || true
PYTHONPATH=src python -m semantic_mirror.cli eval model-compare \
  outputs/sft_raw_eval.json outputs/dpo_raw_eval.json \
  --stage dpo \
  --out outputs/dpo_raw_vs_sft.json || true
if stage_ready outputs/semantic-mirror-rl "$RL_MAX_STEPS"; then
  echo "Reusing existing RL stage output at outputs/semantic-mirror-rl"
else
  python training/run_reward_rl.py --training-dir training \
  --model-name-or-path outputs/semantic-mirror-dpo \
  --output-dir outputs/semantic-mirror-rl \
  --max-steps "$RL_MAX_STEPS" \
  --seed "$RL_SEED" \
  --schema-prefix-mode "$SCHEMA_PREFIX_MODE"
fi
generate_and_inspect rl outputs/semantic-mirror-rl "$RL_SEED"
PYTHONPATH=src python -m semantic_mirror.cli eval model-compare \
  outputs/sft_eval.json outputs/rl_eval.json \
  --stage rl \
  --out outputs/rl_vs_sft.json || true
PYTHONPATH=src python -m semantic_mirror.cli eval model-compare \
  outputs/sft_raw_eval.json outputs/rl_raw_eval.json \
  --stage rl \
  --out outputs/rl_raw_vs_sft.json || true
PYTHONPATH=src python -m semantic_mirror.cli train report outputs --out outputs/diagnostics

python - "$SFT_MAX_STEPS" "$DPO_MAX_STEPS" "$RL_MAX_STEPS" "$SFT_SEED" "$DPO_SEED" "$RL_SEED" "$EVAL_MAX_PROMPTS" "$MAX_NEW_TOKENS" "$SCHEMA_PREFIX_MODE" "$GENERATION_MODE" "$FIELD_MAX_NEW_TOKENS" "$FIELD_TARGET_MODE" "$FIELD_TARGET_LIMIT" "$FIELD_TARGET_MAX_CHUNKS" "$FIELD_TARGET_CHUNK_FIELDS" "$FIELD_OBJECT_PREFIX_MODE" "$FAITHFULNESS_REPAIR_MODE" "$REUSE_STAGE_OUTPUTS" "$SFT_RESUME_FROM_CHECKPOINT" "$DPO_RESUME_FROM_CHECKPOINT" "$SFT_SAVE_STEPS" "$DPO_SAVE_STEPS" "$SFT_SAVE_TOTAL_LIMIT" "$DPO_SAVE_TOTAL_LIMIT" <<'PY'
import json
import sys
from pathlib import Path

required_reports = {
    "sft_eval": "outputs/sft_eval.json",
    "sft_vs_baseline": "outputs/sft_vs_baseline.json",
    "dpo_eval": "outputs/dpo_eval.json",
    "dpo_vs_sft": "outputs/dpo_vs_sft.json",
    "rl_eval": "outputs/rl_eval.json",
    "rl_vs_sft": "outputs/rl_vs_sft.json",
}
diagnostic_reports = {
    "validation_report": "outputs/validation_report.json",
    "audit_report": "outputs/audit.json",
    "baseline_eval": "outputs/baseline_eval.json",
    "sft_raw_eval": "outputs/sft_raw_eval.json",
    "sft_raw_vs_baseline": "outputs/sft_raw_vs_baseline.json",
    "dpo_raw_eval": "outputs/dpo_raw_eval.json",
    "dpo_raw_vs_sft": "outputs/dpo_raw_vs_sft.json",
    "rl_raw_eval": "outputs/rl_raw_eval.json",
    "rl_raw_vs_sft": "outputs/rl_raw_vs_sft.json",
    "sft_sample_manifest": "outputs/samples/sft/sample_manifest.json",
    "dpo_sample_manifest": "outputs/samples/dpo/sample_manifest.json",
    "rl_sample_manifest": "outputs/samples/rl/sample_manifest.json",
    "sft_stage_manifest": "outputs/semantic-mirror-sft/training_stage_manifest.json",
    "dpo_stage_manifest": "outputs/semantic-mirror-dpo/training_stage_manifest.json",
    "rl_stage_manifest": "outputs/semantic-mirror-rl/training_stage_manifest.json",
    "diagnostics": "outputs/diagnostics/training_summary.json",
}

def summarize(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "passed": data.get("passed"),
        "mode": data.get("mode"),
        "metrics": data.get("metrics", {}),
        "deltas": data.get("deltas", {}),
        "gates": data.get("gates", []),
        "raw_candidate_count": data.get("raw_candidate_count"),
        "raw_parseability_count": data.get("raw_parseability_count"),
        "raw_generation_cap_hits": data.get("raw_generation_cap_hits"),
        "raw_schema_validity_count": data.get("raw_schema_validity_count"),
        "raw_repair_free_contract_count": data.get("raw_repair_free_contract_count"),
        "raw_exact_identity_count": data.get("raw_exact_identity_count"),
        "raw_top_level_key_validity_count": data.get("raw_top_level_key_validity_count"),
        "raw_compact_shape_count": data.get("raw_compact_shape_count"),
        "repaired_schema_validity_count": data.get("repaired_schema_validity_count"),
        "raw_static_faithfulness_score": data.get("raw_static_faithfulness_score"),
        "static_faithfulness_score": data.get("static_faithfulness_score"),
        "missing_metrics": data.get("missing_metrics"),
        "max_steps": data.get("max_steps"),
        "seed": data.get("seed"),
        "dataset_records": data.get("dataset_records"),
        "output_dir": data.get("output_dir"),
    }

def rate(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator

def raw_gate_stage_summary(stage, compare_name):
    sample = summary["diagnostic_reports"].get(f"{stage}_sample_manifest", {})
    raw_eval = summary["diagnostic_reports"].get(f"{stage}_raw_eval", {})
    raw_compare = summary["diagnostic_reports"].get(compare_name, {})
    raw_candidate_count = sample.get("raw_candidate_count")
    raw_parseability_rate = rate(sample.get("raw_parseability_count"), raw_candidate_count)
    raw_schema_validity_rate = rate(sample.get("raw_schema_validity_count"), raw_candidate_count)
    raw_repair_free_rate = rate(sample.get("raw_repair_free_contract_count"), raw_candidate_count)
    return {
        "raw_eval_report": f"outputs/{stage}_raw_eval.json",
        "raw_compare_report": raw_compare.get("path") or diagnostic_reports[compare_name],
        "raw_candidate_count": raw_candidate_count,
        "raw_parseability_count": sample.get("raw_parseability_count"),
        "raw_parseability_rate": raw_parseability_rate,
        "raw_parseability_gate_passed": (
            raw_parseability_rate is not None and raw_parseability_rate >= 0.8
        ),
        "raw_schema_validity_count": sample.get("raw_schema_validity_count"),
        "raw_schema_validity_rate": raw_schema_validity_rate,
        "raw_schema_validity_gate_passed": (
            raw_schema_validity_rate is not None and raw_schema_validity_rate >= 0.8
        ),
        "raw_repair_free_contract_count": sample.get("raw_repair_free_contract_count"),
        "raw_repair_free_contract_rate": raw_repair_free_rate,
        "raw_repair_free_contract_gate_passed": (
            raw_repair_free_rate is not None and raw_repair_free_rate >= 0.5
        ),
        "raw_generation_cap_hits": sample.get("raw_generation_cap_hits"),
        "raw_eval_passed": raw_eval.get("passed"),
        "raw_eval_metrics": raw_eval.get("metrics", {}),
        "raw_compare_passed": raw_compare.get("passed"),
        "raw_compare_deltas": raw_compare.get("deltas", {}),
        "raw_compare_gates": raw_compare.get("gates", []),
    }

summary = {
    "mode": "training_eval_summary",
    "eval_run_config": {
        "requested_max_steps": {
            "sft": int(sys.argv[1]),
            "dpo": int(sys.argv[2]),
            "rl": int(sys.argv[3]),
        },
        "seeds": {
            "sft": int(sys.argv[4]),
            "dpo": int(sys.argv[5]),
            "rl": int(sys.argv[6]),
        },
        "eval_max_prompts": None if sys.argv[7] == "" else int(sys.argv[7]),
        "max_new_tokens": int(sys.argv[8]),
        "schema_prefix_mode": sys.argv[9],
        "generation_mode": sys.argv[10],
        "field_max_new_tokens": int(sys.argv[11]),
        "field_target_mode": sys.argv[12],
        "field_target_limit": int(sys.argv[13]),
        "field_target_max_chunks": int(sys.argv[14]),
        "field_target_chunk_fields": [item for item in sys.argv[15].split(",") if item],
        "field_object_prefix_mode": sys.argv[16],
        "faithfulness_repair_mode": sys.argv[17],
        "reuse_stage_outputs": sys.argv[18] == "1",
        "sft_resume_from_checkpoint": sys.argv[19] or None,
        "dpo_resume_from_checkpoint": sys.argv[20] or None,
        "checkpoint_policy": {
            "sft_save_steps": int(sys.argv[21]),
            "dpo_save_steps": int(sys.argv[22]),
            "sft_save_total_limit": int(sys.argv[23]),
            "dpo_save_total_limit": int(sys.argv[24]),
        },
    },
    "gate_policy": {
        "required_reports": "must_pass",
        "diagnostic_reports": "must_exist_but_may_fail",
        "raw_comparison_reports": "diagnostic_non_blocking",
    },
    "required_reports": {},
    "diagnostic_reports": {},
}
for name, path in required_reports.items():
    summary["required_reports"][name] = summarize(path)
for name, path in diagnostic_reports.items():
    report_path = Path(path)
    summary["diagnostic_reports"][name] = {
        "exists": report_path.exists(),
        **(summarize(path) if report_path.exists() else {}),
    }
summary["raw_gate_summary"] = {
    "sft": raw_gate_stage_summary("sft", "sft_raw_vs_baseline"),
    "dpo": raw_gate_stage_summary("dpo", "dpo_raw_vs_sft"),
    "rl": raw_gate_stage_summary("rl", "rl_raw_vs_sft"),
}
summary["stage_execution_summary"] = {
    stage: {
        "requested_max_steps": summary["eval_run_config"]["requested_max_steps"][stage],
        "manifest_max_steps": summary["diagnostic_reports"].get(f"{stage}_stage_manifest", {}).get("max_steps"),
        "seed": summary["diagnostic_reports"].get(f"{stage}_stage_manifest", {}).get("seed"),
        "dataset_records": summary["diagnostic_reports"].get(f"{stage}_stage_manifest", {}).get("dataset_records"),
        "output_dir": summary["diagnostic_reports"].get(f"{stage}_stage_manifest", {}).get("output_dir"),
        "resume_from_checkpoint": summary["diagnostic_reports"].get(f"{stage}_stage_manifest", {}).get("resume_from_checkpoint"),
        "reuse_mode_enabled": summary["eval_run_config"]["reuse_stage_outputs"],
        "manifest_matches_requested_max_steps": (
            summary["diagnostic_reports"].get(f"{stage}_stage_manifest", {}).get("max_steps")
            == summary["eval_run_config"]["requested_max_steps"][stage]
        ),
    }
    for stage in ("sft", "dpo", "rl")
}
summary["final_eval_gate_summary"] = {
    "heldout_unit_coverage_1_0": all(
        item.get("metrics", {}).get("heldout_unit_coverage") == 1.0
        for item in [
            summary["required_reports"].get("sft_eval", {}),
            summary["required_reports"].get("dpo_eval", {}),
            summary["required_reports"].get("rl_eval", {}),
        ]
    ),
    "repaired_schema_validity_1_0": all(
        item.get("metrics", {}).get("schema_validity") == 1.0
        for item in [
            summary["required_reports"].get("sft_eval", {}),
            summary["required_reports"].get("dpo_eval", {}),
            summary["required_reports"].get("rl_eval", {}),
        ]
    ),
    "sft_vs_baseline_passed": summary["required_reports"].get("sft_vs_baseline", {}).get("passed"),
    "dpo_vs_sft_passed": summary["required_reports"].get("dpo_vs_sft", {}).get("passed"),
    "rl_vs_sft_passed": summary["required_reports"].get("rl_vs_sft", {}).get("passed"),
    "rl_raw_hallucination_not_worse_than_sft": (
        summary["raw_gate_summary"].get("rl", {})
        .get("raw_compare_deltas", {})
        .get("hallucination_penalties", 1)
        <= 0
    ),
    "rl_raw_static_faithfulness_not_worse_than_sft": (
        summary["raw_gate_summary"].get("rl", {})
        .get("raw_compare_deltas", {})
        .get("average_static_faithfulness_score", -1)
        >= 0
    ),
    "raw_parseability_stretch_passed": all(
        item.get("raw_parseability_gate_passed") for item in summary["raw_gate_summary"].values()
    ),
    "raw_schema_validity_stretch_passed": all(
        item.get("raw_schema_validity_gate_passed") for item in summary["raw_gate_summary"].values()
    ),
    "raw_repair_free_contract_stretch_passed": all(
        item.get("raw_repair_free_contract_gate_passed")
        for item in summary["raw_gate_summary"].values()
    ),
}
summary["final_eval_gate_summary"]["all_final_eval_gates_passed"] = all(
    value is True for value in summary["final_eval_gate_summary"].values()
)
summary["passed"] = all(
    item.get("passed") for item in summary["required_reports"].values()
) and all(item.get("exists") for item in summary["diagnostic_reports"].values())
summary["gate_counts"] = {
    "required_total": len(summary["required_reports"]),
    "required_passed": sum(1 for item in summary["required_reports"].values() if item.get("passed")),
    "diagnostic_total": len(summary["diagnostic_reports"]),
    "diagnostic_existing": sum(1 for item in summary["diagnostic_reports"].values() if item.get("exists")),
    "diagnostic_failed": sum(
        1
        for item in summary["diagnostic_reports"].values()
        if item.get("exists") and item.get("passed") is False
    ),
}
summary["required_total"] = summary["gate_counts"]["required_total"]
summary["required_passed"] = summary["gate_counts"]["required_passed"]
summary["diagnostic_total"] = summary["gate_counts"]["diagnostic_total"]
summary["diagnostic_existing"] = summary["gate_counts"]["diagnostic_existing"]
summary["diagnostic_failed"] = summary["gate_counts"]["diagnostic_failed"]
summary["all_final_eval_gates_passed"] = summary["final_eval_gate_summary"][
    "all_final_eval_gates_passed"
]
Path("outputs/training_eval_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
PY

contract_status_args=(
  train contract-status outputs
  --sft-steps "$SFT_MAX_STEPS"
  --dpo-steps "$DPO_MAX_STEPS"
  --rl-steps "$RL_MAX_STEPS"
  --out outputs/contract_status.json
  --markdown-out outputs/contract_status.md
)
if [[ -f "$PACKAGE_SOURCE_FRESHNESS" ]]; then
  contract_status_args+=(--package-source-freshness "$PACKAGE_SOURCE_FRESHNESS")
fi
if [[ -f "$WINDOWS_AUDIT" ]]; then
  contract_status_args+=(--windows-audit "$WINDOWS_AUDIT")
fi
if [[ -f "$WSL_SMOKE_MANIFEST" ]]; then
  contract_status_args+=(--wsl-smoke-manifest "$WSL_SMOKE_MANIFEST")
fi
PYTHONPATH=src python -m semantic_mirror.cli "${contract_status_args[@]}"
""",
        "inspect_full_training_eval_resume.sh": """#!/usr/bin/env bash
set -euo pipefail

SFT_MAX_STEPS="${SFT_MAX_STEPS:-2000}"
DPO_MAX_STEPS="${DPO_MAX_STEPS:-800}"
RL_MAX_STEPS="${RL_MAX_STEPS:-1000}"
REUSE_STAGE_OUTPUTS="${REUSE_STAGE_OUTPUTS:-0}"
SFT_RESUME_FROM_CHECKPOINT="${SFT_RESUME_FROM_CHECKPOINT:-}"
DPO_RESUME_FROM_CHECKPOINT="${DPO_RESUME_FROM_CHECKPOINT:-}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-10}"
DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-10}"
SFT_SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-3}"
DPO_SAVE_TOTAL_LIMIT="${DPO_SAVE_TOTAL_LIMIT:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Neither $PYTHON_BIN nor python3 was found." >&2
    exit 1
  fi
fi

mkdir -p outputs
inspect_args=(
  train inspect-resume outputs
  --sft-steps "$SFT_MAX_STEPS"
  --dpo-steps "$DPO_MAX_STEPS"
  --rl-steps "$RL_MAX_STEPS"
  --sft-save-steps "$SFT_SAVE_STEPS"
  --dpo-save-steps "$DPO_SAVE_STEPS"
  --sft-save-total-limit "$SFT_SAVE_TOTAL_LIMIT"
  --dpo-save-total-limit "$DPO_SAVE_TOTAL_LIMIT"
  --out outputs/full_training_eval_resume_inspection.json
  --markdown-out outputs/full_training_eval_resume_inspection.md
)
if [[ "$REUSE_STAGE_OUTPUTS" == "1" ]]; then
  inspect_args+=(--reuse-stage-outputs)
fi
if [[ -n "$SFT_RESUME_FROM_CHECKPOINT" ]]; then
  inspect_args+=(--sft-resume-from-checkpoint "$SFT_RESUME_FROM_CHECKPOINT")
fi
if [[ -n "$DPO_RESUME_FROM_CHECKPOINT" ]]; then
  inspect_args+=(--dpo-resume-from-checkpoint "$DPO_RESUME_FROM_CHECKPOINT")
fi
PYTHONPATH=src "$PYTHON_BIN" -m semantic_mirror.cli "${inspect_args[@]}"
""",
    }
    for name, content in scripts.items():
        path = launch_target / name
        path.write_text(content, encoding="utf-8", newline="\n")
        path.chmod(0o755)


def _write_bootstrap_scripts(setup_target: Path) -> None:
    scripts = {
        "bootstrap_linux_cuda.sh": f"""#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${{PYTHON_BIN:-python3.12}}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN was not found. Install Python 3.11, 3.12, or 3.13 before running this script." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
version = sys.version_info[:2]
if not ((3, 11) <= version < (3, 14)):
    raise SystemExit(f"Python {{sys.version.split()[0]}} is unsupported; expected {UNSLOTH_PYTHON_RANGE}.")
PY

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-training.txt
python - <<'PY'
import importlib.util
missing = [
    name
    for name in (
        "torch",
        "unsloth",
        "trl",
        "datasets",
        "transformers",
        "bitsandbytes",
        "mergekit",
        "llm_blender",
        "weave",
    )
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit("Missing training modules after install: " + ", ".join(missing))
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch imports, but CUDA is not available.")
import transformers.utils.hub as transformers_hub
if not hasattr(transformers_hub, "TRANSFORMERS_CACHE"):
    from pathlib import Path
    transformers_hub.TRANSFORMERS_CACHE = str(Path.home() / ".cache" / "huggingface" / "hub")
from trl import DPOTrainer, SFTTrainer  # noqa: F401
print("Semantic Mirror training environment ready:", torch.__version__, torch.version.cuda)
PY
""",
        "bootstrap_wsl_ubuntu.ps1": """param(
  [string]$Distro = "Ubuntu",
  [string]$Python = "python3.12"
)

$ErrorActionPreference = "Stop"
$windowsPath = (Get-Location).Path
$windowsPathForWsl = $windowsPath -replace '\\\\', '/'
$wslPath = (wsl.exe -d $Distro -- wslpath -a "$windowsPathForWsl").Trim()
wsl.exe -d $Distro -- bash -lc "cd '$wslPath' && PYTHON_BIN=$Python bash setup/bootstrap_linux_cuda.sh"
""",
    }
    for name, content in scripts.items():
        path = setup_target / name
        newline = "\n" if name.endswith(".sh") else None
        path.write_text(content, encoding="utf-8", newline=newline)
        if name.endswith(".sh"):
            path.chmod(0o755)


def _training_package_readme(audit: dict[str, Any]) -> str:
    ready = "yes" if audit["ready_to_launch"] else "no"
    return f"""# Semantic Mirror Training Bundle

This directory is generated by `semantic-mirror train package`. It contains a
validated training batch, generated Unsloth/TRL scripts, the Semantic Mirror
runtime source needed for candidate scoring, launch helpers, and a sanitized
audit of the packaging machine.

Current-machine ready to launch: {ready}

Run on a CUDA Linux or WSL machine. The bootstrap script creates `.venv`, checks
for Python {UNSLOTH_PYTHON_RANGE}, installs CUDA training dependencies, and
verifies PyTorch CUDA importability:

```bash
export SFT_MAX_STEPS="${{SFT_MAX_STEPS:-2000}}"
export DPO_MAX_STEPS="${{DPO_MAX_STEPS:-800}}"
export RL_MAX_STEPS="${{RL_MAX_STEPS:-1000}}"
bash setup/bootstrap_linux_cuda.sh
source .venv/bin/activate
bash launch/run_sft.sh
bash launch/run_dpo.sh
bash launch/run_rl.sh
```

Run the bounded smoke chain before longer training. This validates and audits
the batch, runs capped SFT/DPO/RL stages, generates raw and repaired candidates
for each stage, writes sample inspection artifacts, creates diagnostics, and
records `outputs/smoke-chain/smoke_chain_manifest.json`:

```bash
HELD_OUT_DATASET=/path/to/heldout_dataset \
SFT_SMOKE_STEPS=1 DPO_SMOKE_STEPS=1 RL_SMOKE_STEPS=1 \
bash launch/run_smoke_chain.sh
```

From Windows PowerShell, the WSL smoke launcher converts the bundle and dataset
paths with `wslpath`, bootstraps `.venv` if needed, records the mounted repo
path, Python executable, CUDA device, and smoke output path in
`outputs/smoke-chain-wsl/wsl_smoke_environment.json`, and then runs the same
bounded smoke chain:

```powershell
powershell -ExecutionPolicy Bypass -File launch/run_wsl_smoke_chain.ps1 -HeldOutDataset C:\\path\\to\\heldout_dataset
```

Generate and score held-out candidates after training:

```bash
bash launch/generate_candidates.sh
HELD_OUT_REPO=/path/to/held-out/repo bash launch/score_candidates.sh
```

Run the full training and model-gate sequence when a held-out dataset and a
baseline candidate JSONL are available. `semantic-mirror teacher ingest` writes
`teacher_candidates.jsonl`, which can be used as `BASELINE_CANDIDATES` when the
teacher run covered the same held-out units:

```bash
HELD_OUT_DATASET=/path/to/heldout_dataset \
BASELINE_CANDIDATES=/path/to/teacher_results/teacher_candidates.jsonl \
bash launch/run_full_training_eval.sh
```

For long full-eval runs, set `REUSE_STAGE_OUTPUTS=1` to reuse an existing
SFT/DPO/RL output directory only when its `training_stage_manifest.json`
`max_steps` matches the requested stage cap.
Use `SFT_RESUME_FROM_CHECKPOINT=/path/to/checkpoint` or
`DPO_RESUME_FROM_CHECKPOINT=/path/to/checkpoint` to resume those trainer-backed
stages. The trainer-backed stages default to `SFT_SAVE_STEPS=10`,
`DPO_SAVE_STEPS=10`, `SFT_SAVE_TOTAL_LIMIT=3`, and
`DPO_SAVE_TOTAL_LIMIT=3` so interrupted bounded runs keep recent checkpoints
without retaining every checkpoint. RL currently records
`resume_supported=false`, so the wrapper only reuses a completed RL output
directory.

Before launching a resumed full-eval run, inspect the same reuse and resume
decisions without starting training. The shell helper calls the same
`train inspect-resume` CLI command that can be run directly by automation:

```bash
SFT_MAX_STEPS=300 DPO_MAX_STEPS=120 RL_MAX_STEPS=120 \
REUSE_STAGE_OUTPUTS=1 \
DPO_RESUME_FROM_CHECKPOINT=outputs/semantic-mirror-dpo/checkpoint-10 \
bash launch/inspect_full_training_eval_resume.sh

PYTHONPATH=src python -m semantic_mirror.cli train inspect-resume outputs \
  --sft-steps 300 \
  --dpo-steps 120 \
  --rl-steps 120 \
  --reuse-stage-outputs \
  --dpo-resume-from-checkpoint outputs/semantic-mirror-dpo/checkpoint-10 \
  --out outputs/full_training_eval_resume_inspection.json \
  --markdown-out outputs/full_training_eval_resume_inspection.md
```

The inspector writes `outputs/full_training_eval_resume_inspection.json` and
`outputs/full_training_eval_resume_inspection.md`, and reports whether each
stage will be reused, resumed, rerun, or started fresh.
For automation, `launch/commands_manifest.json` classifies every packaged
command by category and includes a `launches_training` boolean so inspection,
status, and diagnostic commands can be selected without accidentally starting a
training run. The older `launch/commands.json` remains a flat command lookup for
backward compatibility.

After packaging or refreshing bundled source, generate source freshness evidence
from the package root:

```bash
PYTHONPATH=src python -m semantic_mirror.cli train source-freshness . \
  --repo-root /path/to/repo \
  --out source_freshness.json \
  --markdown-out source_freshness.md
```

When regenerating contract status outside `run_full_training_eval.sh`, include
the package source freshness report, native audit report, and WSL smoke-chain
manifest so Windows readiness evidence is not dropped:

```bash
PYTHONPATH=src python -m semantic_mirror.cli train contract-status outputs \
  --sft-steps "$SFT_MAX_STEPS" \
  --dpo-steps "$DPO_MAX_STEPS" \
  --rl-steps "$RL_MAX_STEPS" \
  --windows-audit audit/current_environment.json \
  --wsl-smoke-manifest outputs/smoke-chain-wsl/smoke_chain_manifest.json \
  --package-source-freshness source_freshness.json \
  --out outputs/contract_status.json \
  --markdown-out outputs/contract_status.md
```

This writes `outputs/baseline_eval.json`, raw and repaired eval reports for
SFT/DPO/RL, `outputs/sft_vs_baseline.json`, `outputs/dpo_vs_sft.json`,
`outputs/rl_vs_sft.json`, per-stage sample inspection folders,
`outputs/diagnostics/`, `outputs/training_eval_summary.json`,
`outputs/contract_status.json`, and `outputs/contract_status.md`. The contract
status JSON includes `remaining_recovery_plan`, and the Markdown includes a
`Recovery Plan` table mapping each failed gate to the required action, whether
training is required, blocking stages, and target artifacts. The
recovery plan distinguishes stale stage-derived eval and sample artifacts from
missing reports: eval rows use `generate_eval_report_after_stage`, sample rows
use `generate_sample_inspection_after_stage`, and `blocked_by_stages` names the
stage that must be completed before those artifacts are current.
`outputs/contract_status.json` and `train contract-status` stdout are automation
surfaces: both include `contract_scorecard_summary`, `repo_hygiene_summary`,
`windows_readiness_summary`, `package_source_summary`,
`package_command_manifest_summary`, `package_metadata_summary`,
`human_usefulness_summary`, `stage_recovery_summary`, `remaining_by_area`,
`remaining_recovery_plan`, and compact `next_actions`, including
current-versus-expected failed-gate evidence, DPO/RL resume decisions, Phase 6
failed gates, real timed-answer counts, package source freshness,
command-manifest safety checks, and package Python metadata.
Checked package evidence failures surface as package-area gates with non-training
recovery actions for source freshness, command manifest, and Python metadata.
Sample manifests and the summary include raw parseability, cap hits, repair-free
contract counts, exact identity counts, top-level key validity, and compact shape
validity.

The bundle does not include `.env` or secret values. Use `.env.training.example`
as a template on the target machine.
"""


def _unsloth_sft_script() -> str:
    return '''"""Run Unsloth LoRA SFT for Semantic Mirror data.

Install runtime dependencies in a CUDA environment before running:
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" trl datasets
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
os.environ.setdefault("UNSLOTH_ENABLE_CCE", "0")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", default=".")
    parser.add_argument("--output-dir", default="outputs/semantic-mirror-sft")
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.training_dir)
    config = json.loads((root / "unsloth_sft_config.json").read_text(encoding="utf-8"))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model"],
        max_seq_length=config["training"]["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        load_in_16bit=config.get("load_in_16bit", False),
    )
    tokenizer.truncation_side = "left"
    dataset = load_dataset("json", data_files=str(root / config["inputs"]["sft_jsonl"]), split="train")
    dataset = dataset.map(lambda row: {"text": _messages_to_text(row["messages"], tokenizer)})
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=args.output_dir,
            dataset_text_field="text",
            max_length=config["training"]["max_seq_length"],
            packing=config["training"]["packing"],
            per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            warmup_ratio=config["training"]["warmup_ratio"],
            num_train_epochs=args.num_train_epochs or config["training"]["num_train_epochs"],
            max_steps=args.max_steps or -1,
            learning_rate=config["training"]["learning_rate"],
            logging_steps=1,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            optim="adamw_8bit",
            seed=args.seed,
            report_to="none",
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    _write_stage_manifest(
        output,
        {
            "stage": "sft",
            "training_dir": str(root),
            "output_dir": str(output),
            "model_name_or_path": config["base_model"],
            "dataset_records": len(dataset),
            "max_steps": args.max_steps,
            "num_train_epochs": args.num_train_epochs or config["training"]["num_train_epochs"],
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "seed": args.seed,
        },
    )
    return 0


def _write_stage_manifest(output: Path, payload: dict) -> None:
    payload = dict(payload)
    payload["artifact_files"] = sorted(path.name for path in output.iterdir() if path.is_file())
    (output / "training_stage_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )


def _messages_to_text(messages: list[dict[str, str]], tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    parts = []
    for message in messages:
        role = message["role"].upper()
        parts.append(f"<|{role}|>\\n{message['content']}")
    parts.append("<|ASSISTANT_END|>")
    return "\\n\\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _preference_dpo_script() -> str:
    return '''"""Run preference/RL-style DPO training for Semantic Mirror.

Install runtime dependencies in a CUDA environment before running:
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" trl datasets
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
os.environ.setdefault("UNSLOTH_ENABLE_CCE", "0")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
from unsloth import FastLanguageModel
import transformers.utils.hub as transformers_hub

if not hasattr(transformers_hub, "TRANSFORMERS_CACHE"):
    transformers_hub.TRANSFORMERS_CACHE = str(Path.home() / ".cache" / "huggingface" / "hub")

from trl import DPOConfig, DPOTrainer


def _has_peft_adapters(model) -> bool:
    return bool(getattr(model, "peft_config", None) or getattr(model, "active_adapter", None))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", default=".")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--output-dir", default="outputs/semantic-mirror-dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.training_dir)
    sft_config = json.loads((root / "unsloth_sft_config.json").read_text(encoding="utf-8"))
    reward_config = json.loads((root / "rl_reward_config.json").read_text(encoding="utf-8"))
    dataset = load_dataset(
        "json",
        data_files=str(root / reward_config["inputs"]["preference_pairs_jsonl"]),
        split="train",
    )
    if "images" not in dataset.column_names:
        dataset = dataset.add_column("images", [None] * len(dataset))

    model_name = args.model_name_or_path or sft_config["base_model"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=sft_config["training"]["max_seq_length"],
        load_in_4bit=sft_config["load_in_4bit"],
        load_in_16bit=sft_config.get("load_in_16bit", False),
    )
    tokenizer.truncation_side = "left"
    dpo_processing_class = getattr(tokenizer, "tokenizer", tokenizer)
    dpo_processing_class.truncation_side = "left"
    if not _has_peft_adapters(model):
        model = FastLanguageModel.get_peft_model(
            model,
            r=sft_config["lora"]["r"],
            lora_alpha=sft_config["lora"]["alpha"],
            lora_dropout=sft_config["lora"]["dropout"],
            target_modules=sft_config["lora"]["target_modules"],
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
    try:
        model.warnings_issued
    except AttributeError:
        model.warnings_issued = {}
    original_model_type = getattr(model.config, "model_type", None)
    if original_model_type in {"qwen3_5", "qwen3_5_moe"}:
        model.config.model_type = "qwen3_text"
    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(
            output_dir=args.output_dir,
            beta=args.beta,
            per_device_train_batch_size=sft_config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=sft_config["training"]["gradient_accumulation_steps"],
            learning_rate=1e-6,
            num_train_epochs=args.num_train_epochs or 1,
            max_steps=args.max_steps or -1,
            max_length=sft_config["training"]["max_seq_length"],
            max_prompt_length=min(4096, sft_config["training"]["max_seq_length"] // 2),
            logging_steps=1,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=dataset,
        processing_class=dpo_processing_class,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if original_model_type is not None:
        model.config.model_type = original_model_type
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    _write_stage_manifest(
        output,
        {
            "stage": "dpo",
            "training_dir": str(root),
            "output_dir": str(output),
            "model_name_or_path": model_name,
            "dataset_records": len(dataset),
            "beta": args.beta,
            "max_steps": args.max_steps,
            "num_train_epochs": args.num_train_epochs or 1,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "seed": args.seed,
        },
    )
    return 0


def _write_stage_manifest(output: Path, payload: dict) -> None:
    payload = dict(payload)
    payload["artifact_files"] = sorted(path.name for path in output.iterdir() if path.is_file())
    (output / "training_stage_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _reward_rl_script() -> str:
    return '''"""Run deterministic-reward policy optimization for Semantic Mirror.

This is the RL stage after SFT/DPO. It samples SIR completions from the policy,
scores them against static facts and preference pairs, and applies a simple
REINFORCE-style update. It is intentionally conservative: faithfulness rewards
come before compactness, and missing/invented facts produce negative advantage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
os.environ.setdefault("UNSLOTH_ENABLE_CCE", "0")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
from unsloth import FastLanguageModel
from transformers import StoppingCriteria, StoppingCriteriaList


REWARD_FIELDS = (
    "calls",
    "control_flow",
    "side_effects",
    "returns",
    "writes",
    "state_mutations",
    "failure_modes",
)


class _JsonObjectStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        completion = input_ids[0, self.prompt_len :]
        if completion.numel() == 0:
            return False
        text = self.tokenizer.decode(completion, skip_special_tokens=True)
        return _has_complete_json_object(text)


def _has_peft_adapters(model) -> bool:
    return bool(getattr(model, "peft_config", None) or getattr(model, "active_adapter", None))


def _schema_prefix(record: dict, mode: str) -> str:
    if mode == "off":
        return ""
    target = record.get("reward_reference", {}).get("compact_target", {})
    if not isinstance(target, dict):
        return ""
    identity_fields = (
        "unit_id",
        "source_spans",
        "language",
        "symbol_type",
        "name",
        "qualified_name",
    )
    if any(field not in target for field in identity_fields):
        return ""
    if mode == "schema-scaffold":
        algorithm = target.get("algorithm", {})
        if not isinstance(algorithm, dict):
            algorithm = {
                "claim": "",
                "confidence": 0.5,
                "source_spans": target.get("source_spans", []),
            }
        scaffold = {
            "unit_id": target["unit_id"],
            "source_spans": target["source_spans"],
            "language": target["language"],
            "symbol_type": target["symbol_type"],
            "name": target["name"],
            "qualified_name": target["qualified_name"],
            "algorithm": algorithm,
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
            "confidence": target.get("confidence", algorithm.get("confidence", 0.5)),
        }
        return json.dumps(scaffold, separators=(",", ":"))[:-1]
    prefix = json.dumps(
        {field: target[field] for field in identity_fields},
        separators=(",", ":"),
    )[:-1]
    if mode == "identity":
        return prefix + ',"algorithm":'
    if mode == "identity-algorithm":
        algorithm = target.get("algorithm", {})
        return (
            prefix
            + ',"algorithm":'
            + json.dumps(algorithm, separators=(",", ":"))
            + ',"control_flow":'
        )
    return ""


def _encode_generation_inputs(
    text_tokenizer,
    formatted_prompt: str,
    schema_prefix: str,
    max_prompt_tokens: int,
    device,
):
    prefix_ids = None
    prefix_attention = None
    prefix_tokens = 0
    prompt_limit = max_prompt_tokens
    if schema_prefix:
        prefix_encoded = text_tokenizer(
            schema_prefix,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prefix_ids = prefix_encoded["input_ids"]
        prefix_attention = prefix_encoded.get("attention_mask")
        if prefix_attention is None:
            prefix_attention = torch.ones_like(prefix_ids)
        prefix_tokens = int(prefix_ids.shape[1])
        prompt_limit = max(128, max_prompt_tokens - prefix_tokens)
    encoded = text_tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_limit,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    prompt_len = int(input_ids.shape[1])
    if prefix_ids is not None:
        input_ids = torch.cat([input_ids, prefix_ids], dim=1)
        attention_mask = torch.cat([attention_mask, prefix_attention], dim=1)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return inputs, prompt_len, prefix_tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", default=".")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--output-dir", default="outputs/semantic-mirror-rl")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--faithfulness-repair-mode",
        choices=["schema-only", "compact-target", "full-static"],
        default="schema-only",
    )
    parser.add_argument(
        "--schema-prefix-mode",
        choices=["off", "identity", "identity-algorithm", "schema-scaffold"],
        default="schema-scaffold",
    )
    args = parser.parse_args()

    root = Path(args.training_dir)
    sft_config = json.loads((root / "unsloth_sft_config.json").read_text(encoding="utf-8"))
    reward_config = json.loads((root / "rl_reward_config.json").read_text(encoding="utf-8"))
    prompts = _read_jsonl(root / reward_config["inputs"]["rl_prompts_jsonl"])
    preferences = _preferences_by_prompt(root, reward_config["inputs"]["preference_pairs_jsonl"])
    if not prompts:
        raise ValueError("rl_prompts.jsonl is empty")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name = args.model_name_or_path or sft_config["base_model"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=sft_config["training"]["max_seq_length"],
        load_in_4bit=sft_config["load_in_4bit"],
        load_in_16bit=sft_config.get("load_in_16bit", False),
    )
    tokenizer.truncation_side = "left"
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    text_tokenizer.truncation_side = "left"
    if not _has_peft_adapters(model):
        model = FastLanguageModel.get_peft_model(
            model,
            r=sft_config["lora"]["r"],
            lora_alpha=sft_config["lora"]["alpha"],
            lora_dropout=sft_config["lora"]["dropout"],
            target_modules=sft_config["lora"]["target_modules"],
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    device = next(model.parameters()).device
    generation_tokens = min(
        args.max_new_tokens,
        max(sft_config["training"]["max_seq_length"] - 128, 1),
    )
    max_prompt_tokens = max(128, sft_config["training"]["max_seq_length"] - generation_tokens)
    max_steps = args.max_steps or len(prompts)
    moving_baseline = 0.0
    history = []

    for step in range(max_steps):
        record = prompts[step % len(prompts)]
        formatted_prompt = _format_generation_prompt(record["prompt"], tokenizer)
        schema_prefix = _schema_prefix(record, args.schema_prefix_mode)
        encoded, prompt_len, schema_prefix_tokens = _encode_generation_inputs(
            text_tokenizer,
            formatted_prompt,
            schema_prefix,
            max_prompt_tokens,
            device,
        )
        input_len = int(encoded["input_ids"].shape[1])
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=generation_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                min_new_tokens=8,
                stopping_criteria=StoppingCriteriaList([
                    _JsonObjectStoppingCriteria(text_tokenizer, prompt_len)
                ]),
        )
        output_ids = output_ids.detach().clone().to(device)
        completion = output_ids[:, prompt_len:]
        generated_tokens = int(output_ids.shape[1] - input_len)
        completion_tokens = int(completion.shape[1])
        text = text_tokenizer.decode(completion[0], skip_special_tokens=True)
        raw_sir_unit = _extract_json_object(text)
        repair_input = json.loads(json.dumps(raw_sir_unit))
        sir_unit = _repair_sir_unit(repair_input, record["metadata"], record["reward_reference"])
        sir_unit = _apply_faithfulness_repair(
            sir_unit,
            record["reward_reference"].get("static_facts", {}),
            record["reward_reference"].get("compact_target", {}),
            args.faithfulness_repair_mode,
        )
        reward = _semantic_reward(sir_unit, record["reward_reference"], reward_config)
        reward += _preference_bonus(sir_unit, preferences.get(record["prompt"]))
        reward += _raw_generation_bonus(
            raw_sir_unit,
            record["reward_reference"],
            int(completion.shape[1]),
        )
        moving_baseline = 0.9 * moving_baseline + 0.1 * reward
        advantage = reward - moving_baseline
        sequence_logprob = _sequence_logprob(model, output_ids, prompt_len)
        length_penalty = args.kl_coef * (completion.shape[1] / max(args.max_new_tokens, 1))
        loss = -(advantage * args.reward_scale) * sequence_logprob + length_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(
            {
                "step": step,
                "record_id": record["record_id"],
                "unit_id": record["reward_reference"]["unit_id"],
                "reward": round(float(reward), 4),
                "raw_parseable": not bool(raw_sir_unit.get("raw_error")),
                "raw_parse_error": raw_sir_unit.get("raw_error"),
                "hit_generation_cap": generated_tokens >= generation_tokens,
                "advantage": round(float(advantage), 4),
                "loss": round(float(loss.detach().cpu()), 6),
                "completion_tokens": completion_tokens,
                "generated_tokens": generated_tokens,
                "schema_prefix_mode": args.schema_prefix_mode,
                "schema_prefix_applied": bool(schema_prefix),
                "schema_prefix_tokens": schema_prefix_tokens,
            }
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "rl_training_report.json").write_text(
        json.dumps(
            {
                "stage": "rl",
                "model_name_or_path": model_name,
                "steps": len(history),
                "seed": args.seed,
                "average_reward": round(sum(item["reward"] for item in history) / len(history), 6),
                "history": history,
            },
            indent=2,
            sort_keys=True,
        )
        + "\\n",
        encoding="utf-8",
    )
    _write_stage_manifest(
        output,
        {
            "stage": "rl",
            "training_dir": str(root),
            "output_dir": str(output),
            "model_name_or_path": model_name,
            "dataset_records": len(prompts),
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "kl_coef": args.kl_coef,
            "reward_scale": args.reward_scale,
            "schema_prefix_mode": args.schema_prefix_mode,
            "resume_supported": False,
            "seed": args.seed,
            "metrics": {
                "steps": len(history),
                "average_reward": round(sum(item["reward"] for item in history) / len(history), 6),
                "raw_parseable": sum(1 for item in history if item["raw_parseable"]),
            },
        },
    )
    return 0


def _write_stage_manifest(output: Path, payload: dict) -> None:
    payload = dict(payload)
    payload["artifact_files"] = sorted(path.name for path in output.iterdir() if path.is_file())
    (output / "training_stage_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )


def _sequence_logprob(model, output_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    attention_mask = torch.ones_like(output_ids)
    logits = model(input_ids=output_ids, attention_mask=attention_mask).logits[:, :-1, :]
    labels = output_ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    completion_mask = torch.zeros_like(labels, dtype=torch.bool)
    completion_mask[:, max(prompt_len - 1, 0) :] = True
    denom = completion_mask.sum().clamp_min(1)
    return (token_log_probs * completion_mask).sum() / denom


def _semantic_reward(
    sir_unit: dict,
    reference: dict,
    reward_config: dict,
) -> float:
    if not isinstance(sir_unit, dict) or sir_unit.get("raw_error"):
        return -5.0
    static_facts = reference["static_facts"]
    positive = reward_config["positive_rewards"]
    penalties = reward_config["penalties"]
    reward = 0.0
    for field in REWARD_FIELDS:
        expected = _claim_keys(static_facts.get(field, []), field)
        observed = _claim_keys(sir_unit.get(field, []), field)
        reward += len(expected & observed) * _positive_value(field, positive)
        reward += len(expected - observed) * penalties["missing_required_static_fact"]
        reward += len(observed - expected) * penalties["invented_call_write_error_or_behavior"]
    for category, expected_claims in static_facts.get("data_ml_details", {}).items():
        expected = _claim_keys(expected_claims, category)
        observed = _claim_keys(sir_unit.get("data_ml_details", {}).get(category, []), category)
        reward += len(expected & observed) * positive["preserved_data_ml_detail"]
        reward += len(expected - observed) * penalties["missing_required_static_fact"]
        reward += len(observed - expected) * penalties["invented_call_write_error_or_behavior"]
    for claim in _iter_claims(sir_unit):
        if not claim.get("source_spans"):
            reward += penalties["claim_without_valid_source_evidence"]
    return reward


DATA_ML_DETAIL_CATEGORIES = (
    "losses",
    "model_architecture",
    "tensor_shapes",
    "training_loops",
    "optimizer_scheduler",
    "metrics",
    "checkpointing",
)
LIST_FIELDS = (
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
)


def _repair_sir_unit(unit: dict, metadata: dict, reference: dict) -> dict:
    if not isinstance(unit, dict):
        unit = {}
    elif unit.get("raw_error"):
        unit = {}
    source_spans = reference.get("source_spans") or unit.get("source_spans") or []
    source_path = metadata.get("source_path") or reference.get("source_path") or "<unknown>"
    unit["unit_id"] = metadata.get("unit_id") or reference.get("unit_id") or unit.get("unit_id")
    unit["source_spans"] = source_spans
    unit["language"] = metadata.get("language") or unit.get("language") or "python"
    unit["symbol_type"] = metadata.get("symbol_type") or unit.get("symbol_type") or "module"
    unit["name"] = unit.get("name") or metadata.get("qualified_name") or source_path
    unit["qualified_name"] = metadata.get("qualified_name") or unit.get("qualified_name") or unit["name"]
    if not isinstance(unit.get("algorithm"), dict):
        unit["algorithm"] = {
            "claim": f"Semantic IR unit for {unit['qualified_name']}.",
            "confidence": 0.5,
            "source_spans": source_spans,
        }
    unit["algorithm"].setdefault("source_spans", source_spans)
    unit["algorithm"].setdefault("confidence", 0.5)
    unit["algorithm"].setdefault("claim", f"Semantic IR unit for {unit['qualified_name']}.")
    for field in LIST_FIELDS:
        if not isinstance(unit.get(field), list):
            unit[field] = []
    if not isinstance(unit.get("data_ml_details"), dict):
        unit["data_ml_details"] = {}
    for category in DATA_ML_DETAIL_CATEGORIES:
        if not isinstance(unit["data_ml_details"].get(category), list):
            unit["data_ml_details"][category] = []
    unit["confidence"] = unit.get("confidence", unit["algorithm"].get("confidence", 0.5))
    return unit


def _apply_faithfulness_repair(
    unit: dict,
    static_facts: dict,
    compact_target: dict | None = None,
    repair_mode: str = "schema-only",
) -> dict:
    if not isinstance(unit, dict) or unit.get("raw_error"):
        return unit
    if repair_mode == "schema-only":
        repair_source = {}
    elif repair_mode == "full-static":
        repair_source = static_facts if isinstance(static_facts, dict) else {}
    elif repair_mode == "compact-target":
        repair_source = compact_target if isinstance(compact_target, dict) else {}
    else:
        raise ValueError("repair_mode must be 'schema-only', 'compact-target', or 'full-static'")
    if isinstance(repair_source.get("algorithm"), dict):
        unit["algorithm"] = repair_source["algorithm"]
    for field in LIST_FIELDS:
        if isinstance(repair_source.get(field), list):
            unit[field] = repair_source[field]
    data_ml_details = repair_source.get("data_ml_details", {})
    if isinstance(data_ml_details, dict):
        unit["data_ml_details"] = {
            category: data_ml_details.get(category, [])
            if isinstance(data_ml_details.get(category), list)
            else []
            for category in DATA_ML_DETAIL_CATEGORIES
        }
    unit["confidence"] = unit.get("confidence", unit.get("algorithm", {}).get("confidence", 0.7))
    return unit


def _raw_generation_bonus(raw_sir_unit: dict, reference: dict, completion_tokens: int) -> float:
    if not isinstance(raw_sir_unit, dict) or raw_sir_unit.get("raw_error"):
        return -80.0
    reference = reference if isinstance(reference, dict) else {}
    reward = 1.0
    if completion_tokens <= 8:
        reward -= 2.0
    compact_target = reference.get("compact_target", {})
    compact_target = compact_target if isinstance(compact_target, dict) else {}
    compact_token_budget = max(
        64,
        len(json.dumps(compact_target, separators=(",", ":"))) // 3 + 64,
    )
    if completion_tokens > compact_token_budget:
        reward -= min((completion_tokens - compact_token_budget) / 16.0, 24.0)
    allowed_keys = {
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
    if compact_target:
        allowed_keys = set(compact_target)
    missing_keys = allowed_keys - set(raw_sir_unit)
    if missing_keys:
        reward -= min(len(missing_keys) * 4.0, 80.0)
    extra_keys = set(raw_sir_unit) - allowed_keys
    if extra_keys:
        reward -= min(len(extra_keys) * 5.0, 40.0)
    for field in ("unit_id", "language", "symbol_type", "name", "qualified_name"):
        expected = compact_target.get(field, reference.get(field))
        if not expected:
            continue
        if raw_sir_unit.get(field) == expected:
            reward += 1.0
        else:
            reward -= 8.0
    expected_spans = compact_target.get("source_spans", reference.get("source_spans"))
    if expected_spans:
        if raw_sir_unit.get("source_spans") == expected_spans:
            reward += 1.0
        else:
            reward -= 6.0
    for key in ("unit_id", "algorithm", "data_ml_details"):
        if key in raw_sir_unit:
            reward += 0.5
    if compact_target and raw_sir_unit == compact_target:
        reward += 20.0
    expected_counts = reference.get("compact_expected_counts", {})
    expected_counts = expected_counts if isinstance(expected_counts, dict) else {}
    for field in LIST_FIELDS:
        expected_len = expected_counts.get(field)
        if expected_len is None and compact_target:
            expected_value = compact_target.get(field, [])
            expected_len = len(expected_value) if isinstance(expected_value, list) else 0
        if expected_len is None:
            continue
        observed = raw_sir_unit.get(field)
        observed_len = len(observed) if isinstance(observed, list) else 0
        if observed_len == expected_len:
            reward += 0.2
        elif observed_len > expected_len:
            reward -= min((observed_len - expected_len) * 0.5, 5.0)
        else:
            reward -= min((expected_len - observed_len) * 0.25, 2.0)
    expected_detail_counts = expected_counts.get("data_ml_details", {})
    expected_detail_counts = (
        expected_detail_counts if isinstance(expected_detail_counts, dict) else {}
    )
    raw_details = raw_sir_unit.get("data_ml_details", {})
    target_details = compact_target.get("data_ml_details", {})
    for category in DATA_ML_DETAIL_CATEGORIES:
        expected_len = expected_detail_counts.get(category)
        if expected_len is None and isinstance(target_details, dict):
            expected_value = target_details.get(category, [])
            expected_len = len(expected_value) if isinstance(expected_value, list) else 0
        if expected_len is None:
            continue
        observed = raw_details.get(category, []) if isinstance(raw_details, dict) else []
        observed_len = len(observed) if isinstance(observed, list) else 0
        if observed_len == expected_len:
            reward += 0.2
        elif observed_len > expected_len:
            reward -= min((observed_len - expected_len) * 0.5, 5.0)
        else:
            reward -= min((expected_len - observed_len) * 0.25, 2.0)
    static_facts = reference.get("static_facts", {})
    static_facts = static_facts if isinstance(static_facts, dict) else {}
    expected_algorithm = static_facts.get("algorithm", {})
    if isinstance(expected_algorithm, dict) and expected_algorithm.get("claim"):
        raw_algorithm = raw_sir_unit.get("algorithm", {})
        if isinstance(raw_algorithm, dict) and raw_algorithm.get("claim"):
            reward += 0.5
        else:
            reward -= 0.5
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
        expected = static_facts.get(field, [])
        if not isinstance(expected, list) or not expected:
            continue
        observed = raw_sir_unit.get(field)
        if isinstance(observed, list) and observed:
            reward += min(len(observed), len(expected)) * 0.25
            if len(observed) > len(expected):
                reward -= min((len(observed) - len(expected)) * 0.25, 4.0)
        else:
            reward -= 0.5
    expected_details = static_facts.get("data_ml_details", {})
    raw_details = raw_sir_unit.get("data_ml_details", {})
    if isinstance(expected_details, dict):
        for category, expected in expected_details.items():
            if not isinstance(expected, list) or not expected:
                continue
            observed = raw_details.get(category, []) if isinstance(raw_details, dict) else []
            if isinstance(observed, list) and observed:
                reward += min(len(observed), len(expected)) * 0.25
                if len(observed) > len(expected):
                    reward -= min((len(observed) - len(expected)) * 0.25, 4.0)
            else:
                reward -= 0.5
    if any(key in raw_sir_unit for key in ("calls", "writes", "returns", "state_mutations")):
        reward += 1.0
    return reward


_GENERATION_SYSTEM_PROMPT = (
    "You generate one valid Semantic Mirror SIR JSON unit. Use every required top-level "
    "schema key exactly once. Preserve only source-backed static facts supplied in the "
    "prompt. Do not invent behavior. Return minified JSON only: begin with {, end with }, "
    "and do not wrap the object in Markdown fences."
)


def _format_generation_prompt(user_prompt: str, tokenizer=None) -> str:
    messages = [
        {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{user_prompt}\\n\\n"
                "Return only one complete minified SIR JSON object. Copy the final SIR JSON "
                "object between FINAL_SIR_JSON_START and FINAL_SIR_JSON_END exactly: same top-level keys, source-backed values, "
                "compact list lengths, and exact identity fields. Do not shorten unit_id or "
                "qualified_name. Do not add safety_report, summary, code_analysis, analysis, "
                "output_template, FINAL_SIR_JSON_START, FINAL_SIR_JSON_END, or any key outside "
                "the final SIR JSON. The answer "
                "must start with {\\\"unit_id\\\". Do not continue the input JSON or use Markdown "
                "fences."
            ),
        },
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        except Exception:
            pass
    return (
        f"<|SYSTEM|>\\n{_GENERATION_SYSTEM_PROMPT}\\n\\n"
        f"<|USER|>\\n{user_prompt}\\n\\n"
        "Return only one complete minified SIR JSON object. Copy the final SIR JSON object between "
        "FINAL_SIR_JSON_START and FINAL_SIR_JSON_END exactly: same top-level keys, "
        "source-backed values, compact list lengths, "
        "and exact identity fields. Do not shorten unit_id or qualified_name. "
        "Do not add safety_report, summary, code_analysis, analysis, output_template, "
        "FINAL_SIR_JSON_START, FINAL_SIR_JSON_END, or any key outside the final SIR JSON. "
        "The answer must start with {\\\"unit_id\\\". Do not continue "
        "the input JSON or use Markdown fences.\\n\\n"
        "<|ASSISTANT|>\\n"
    )


def _positive_value(field: str, positive: dict) -> int:
    if field == "calls":
        return positive["preserved_call"]
    if field == "returns":
        return positive["preserved_return_variant"]
    if field in {"writes", "state_mutations"}:
        return positive["preserved_write_or_state_mutation"]
    if field == "failure_modes":
        return positive["preserved_source_backed_failure_mode"]
    if field == "control_flow":
        return positive["preserved_control_flow"]
    return 1


def _preference_bonus(sir_unit: dict, preference: dict | None) -> float:
    if preference is None or not isinstance(sir_unit, dict):
        return 0.0
    chosen = _parse_json_object(preference.get("chosen"))
    rejected = _parse_json_object(preference.get("rejected"))
    candidate_keys = _unit_signature(sir_unit)
    chosen_overlap = len(candidate_keys & _unit_signature(chosen))
    rejected_overlap = len(candidate_keys & _unit_signature(rejected))
    if chosen_overlap > rejected_overlap:
        return 2.0
    if rejected_overlap > chosen_overlap:
        return -2.0
    return 0.0


def _unit_signature(unit: dict) -> set[str]:
    keys = set()
    if not isinstance(unit, dict):
        return keys
    for field in REWARD_FIELDS:
        keys.update(f"{field}:{key}" for key in _claim_keys(unit.get(field, []), field))
    for category, claims in unit.get("data_ml_details", {}).items():
        keys.update(f"data_ml_{category}:{key}" for key in _claim_keys(claims, category))
    return keys


def _claim_keys(claims: list[dict], field: str) -> set[str]:
    keys = set()
    for claim in claims or []:
        if not isinstance(claim, dict):
            continue
        if "name" in claim:
            keys.add(str(claim["name"]))
        elif "kind" in claim:
            keys.add(f"{field}:{claim['kind']}:{claim.get('claim', '')}")
        else:
            keys.add(str(claim.get("claim", "")))
    return keys


def _iter_claims(unit: dict):
    if not isinstance(unit, dict):
        return
    algorithm = unit.get("algorithm")
    if isinstance(algorithm, dict):
        yield algorithm
    for field in (
        *REWARD_FIELDS,
        "reads",
        "external_dependencies",
        "hazards",
        "uncertainty",
    ):
        for claim in unit.get(field, []) or []:
            if isinstance(claim, dict):
                yield claim
    for claims in unit.get("data_ml_details", {}).values():
        for claim in claims or []:
            if isinstance(claim, dict):
                yield claim


def _preferences_by_prompt(root: Path, rel_path: str) -> dict[str, dict]:
    path = root / rel_path
    if not path.exists():
        return {}
    return {record["prompt"]: record for record in _read_jsonl(path) if "prompt" in record}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _extract_json_object(text: str) -> dict:
    text = _assistant_completion_region(text)
    decoder = json.JSONDecoder()
    last_error = "no JSON object found"
    first_json = text.find("{")
    indices = [first_json] if first_json >= 0 and not text[:first_json].strip() else [
        index for index, char in enumerate(text) if char == "{"
    ]
    for index in indices:
        if index < 0:
            continue
        char = text[index]
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if isinstance(parsed, dict) and "unit_id" in parsed:
            return parsed
    return {"unit_id": "<unparseable>", "raw_error": f"no SIR JSON object found: {last_error}"}


def _has_complete_json_object(text: str) -> bool:
    start = text.find("{")
    if start < 0:
        return False
    depth = 0
    in_string = False
    escape = False
    for char in text[start:]:
        if in_string:
            if escape:
                escape = False
            elif char == "\\\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return True
            if depth < 0:
                return False
    return False


def _assistant_completion_region(text: str) -> str:
    for marker in ("<|ASSISTANT|>", "<|assistant|>"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
    return text.strip()


def _parse_json_object(text: str | None) -> dict:
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"unit_id": "<unparseable>", "raw_error": str(exc)}
    return parsed if isinstance(parsed, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _generate_candidates_script() -> str:
    return '''"""Generate Semantic IR candidate JSONL from a trained model or adapter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
os.environ.setdefault("UNSLOTH_ENABLE_CCE", "0")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
from unsloth import FastLanguageModel
from transformers import StoppingCriteria, StoppingCriteriaList


class _JsonObjectStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        completion = input_ids[0, self.prompt_len :]
        if completion.numel() == 0:
            return False
        text = self.tokenizer.decode(completion, skip_special_tokens=True)
        return _has_complete_json_object(text)


def _schema_prefix(record: dict, mode: str) -> str:
    if mode == "off":
        return ""
    target = record.get("reward_reference", {}).get("compact_target", {})
    if not isinstance(target, dict):
        return ""
    identity_fields = (
        "unit_id",
        "source_spans",
        "language",
        "symbol_type",
        "name",
        "qualified_name",
    )
    if any(field not in target for field in identity_fields):
        return ""
    if mode == "schema-scaffold":
        algorithm = target.get("algorithm", {})
        if not isinstance(algorithm, dict):
            algorithm = {
                "claim": "",
                "confidence": 0.5,
                "source_spans": target.get("source_spans", []),
            }
        scaffold = {
            "unit_id": target["unit_id"],
            "source_spans": target["source_spans"],
            "language": target["language"],
            "symbol_type": target["symbol_type"],
            "name": target["name"],
            "qualified_name": target["qualified_name"],
            "algorithm": algorithm,
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
            "confidence": target.get("confidence", algorithm.get("confidence", 0.5)),
        }
        return json.dumps(scaffold, separators=(",", ":"))[:-1]
    prefix = json.dumps(
        {field: target[field] for field in identity_fields},
        separators=(",", ":"),
    )[:-1]
    if mode == "identity":
        return prefix + ',"algorithm":'
    if mode == "identity-algorithm":
        algorithm = target.get("algorithm", {})
        return (
            prefix
            + ',"algorithm":'
            + json.dumps(algorithm, separators=(",", ":"))
            + ',"control_flow":'
        )
    return ""


def _encode_generation_inputs(
    text_tokenizer,
    formatted_prompt: str,
    schema_prefix: str,
    max_prompt_tokens: int,
    device,
):
    prefix_ids = None
    prefix_attention = None
    prefix_tokens = 0
    prompt_limit = max_prompt_tokens
    if schema_prefix:
        prefix_encoded = text_tokenizer(
            schema_prefix,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prefix_ids = prefix_encoded["input_ids"]
        prefix_attention = prefix_encoded.get("attention_mask")
        if prefix_attention is None:
            prefix_attention = torch.ones_like(prefix_ids)
        prefix_tokens = int(prefix_ids.shape[1])
        prompt_limit = max(128, max_prompt_tokens - prefix_tokens)
    encoded = text_tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_limit,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    prompt_len = int(input_ids.shape[1])
    if prefix_ids is not None:
        input_ids = torch.cat([input_ids, prefix_ids], dim=1)
        attention_mask = torch.cat([attention_mask, prefix_attention], dim=1)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    return inputs, prompt_len, prefix_tokens


FIELD_WISE_LIST_FIELDS = (
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
)


FIELD_WISE_DATA_ML_CATEGORIES = (
    "losses",
    "model_architecture",
    "tensor_shapes",
    "training_loops",
    "optimizer_scheduler",
    "metrics",
    "checkpointing",
)


_FIELD_SYSTEM_PROMPT = (
    "You generate one source-backed Semantic Mirror SIR field. Return only one "
    "minified JSON object with the requested field name. The first character "
    "must be { and the last character must be }. Do not add Markdown, prose, "
    "analysis, tables, or unrelated keys."
)


FIELD_STATIC_HINT_DEFAULT_LIMITS = {
    "control_flow": 2,
    "reads": 2,
    "writes": 2,
    "calls": 2,
    "returns": 2,
    "side_effects": 2,
    "failure_modes": 2,
    "state_mutations": 2,
    "external_dependencies": 1,
    "hazards": 2,
    "uncertainty": 1,
}


FIELD_STATIC_HINT_KEYS = (
    "claim",
    "confidence",
    "source_spans",
    "name",
    "call",
    "kind",
    "target",
    "module",
    "imported",
    "alias",
    "predicate",
    "condition",
    "expression",
    "value",
    "parameter",
    "symbol",
    "metric",
    "loss",
    "shape",
)


def _compact_generation_claim(claim):
    if not isinstance(claim, dict):
        return {"claim": "", "confidence": 0.0, "source_spans": []}
    compact = {
        key: claim[key]
        for key in FIELD_STATIC_HINT_KEYS
        if key in claim and claim[key] not in (None, [], {})
    }
    if "claim" not in compact:
        compact["claim"] = ""
    compact["confidence"] = compact.get("confidence", claim.get("confidence", 0.7))
    spans = compact.get("source_spans", [])
    compact["source_spans"] = spans[:1] if isinstance(spans, list) else []
    return compact


def _hint_limit(field: str, requested_limit: int, compact_value) -> int:
    if requested_limit > 0:
        return requested_limit
    if field == "data_ml_details":
        return 2
    compact_count = len(compact_value) if isinstance(compact_value, list) else 0
    return max(compact_count, FIELD_STATIC_HINT_DEFAULT_LIMITS.get(field, 2))


def _compact_static_hint_value(field: str, value, limit: int):
    return _compact_static_hint_chunk(field, value, limit, 0)


def _compact_static_hint_chunk(field: str, value, limit: int, chunk_index: int):
    offset = max(chunk_index, 0) * max(limit, 1)
    if field == "data_ml_details":
        value = value if isinstance(value, dict) else {}
        details = {}
        for category in FIELD_WISE_DATA_ML_CATEGORIES:
            category_value = value.get(category, [])
            if not isinstance(category_value, list):
                category_value = []
            details[category] = [
                _compact_generation_claim(claim)
                for claim in category_value[offset : offset + limit]
            ]
        return details
    if isinstance(value, list):
        return [
            _compact_generation_claim(claim)
            for claim in value[offset : offset + limit]
        ]
    return [] if field != "data_ml_details" else {}


def _hint_count(field: str, value) -> int:
    if field == "data_ml_details":
        value = value if isinstance(value, dict) else {}
        return max(
            (
                len(items)
                for items in value.values()
                if isinstance(items, list)
            ),
            default=0,
        )
    return len(value) if isinstance(value, list) else 0


def _merge_field_value(field: str, existing, new_value):
    if field == "data_ml_details":
        merged = {
            category: list(existing.get(category, []))
            if isinstance(existing, dict) and isinstance(existing.get(category), list)
            else []
            for category in FIELD_WISE_DATA_ML_CATEGORIES
        }
        new_value = new_value if isinstance(new_value, dict) else {}
        for category in FIELD_WISE_DATA_ML_CATEGORIES:
            items = new_value.get(category, [])
            if isinstance(items, list):
                merged[category].extend(items)
        return merged
    merged = list(existing) if isinstance(existing, list) else []
    if isinstance(new_value, list):
        merged.extend(new_value)
    return merged


def _parse_field_set(value: str) -> set[str]:
    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


def _format_field_prompt(
    base_prompt: str,
    field: str,
    target_value,
    field_target_mode: str,
    tokenizer=None,
) -> str:
    field_type = "object" if field == "data_ml_details" else "array"
    if field_target_mode == "static-facts":
        target_instruction = (
            "Use the full source-backed static_facts value as the target coverage. "
            "Do not exceed that target and do not invent facts."
        )
        target_label = "Full static_facts target"
    elif field_target_mode == "static-hints":
        target_instruction = (
            "Use static_hints as the source-backed fact pool. Return only facts copied "
            "from or directly supported by static_hints, and keep the output at or "
            "below output_budget. The compact_target shows the existing compact field "
            "shape; static_hints may include additional source-backed facts for this "
            "diagnostic."
        )
        target_label = "Static hints target"
        target_payload = (
            f"Output budget for `{field}`: {target_value.get('output_budget', 0)}. "
            f"Existing compact `{field}` value: "
            f"{json.dumps(target_value.get('compact_target'), separators=(',', ':'))}. "
            f"Allowed static hints for `{field}`: "
            f"{json.dumps(target_value.get('static_hints'), separators=(',', ':'))}."
        )
    else:
        target_instruction = (
            "Keep the compact target value as an upper bound and do not invent facts."
        )
        target_label = "Compact target"
        target_payload = (
            f"{target_label} for `{field}`: "
            f"{json.dumps(target_value, separators=(',', ':'))}"
        )
    if field_target_mode == "static-facts":
        target_payload = (
            f"{target_label} for `{field}`: "
            f"{json.dumps(target_value, separators=(',', ':'))}"
        )
    messages = [
        {"role": "system", "content": _FIELD_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{base_prompt}\\n\\n"
                f"Generate only the `{field}` field as a JSON {field_type}. "
                "Use source-backed facts from static_facts and the final SIR JSON. "
                f"{target_instruction} "
                "Your first output character must be `{`; output no prose before it. "
                "Return exactly one minified object shaped like "
                f"{{\\\"{field}\\\":<json_{field_type}>}}. "
                f"{target_payload}"
            ),
        },
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        except Exception:
            pass
    return (
        f"<|SYSTEM|>\\n{_FIELD_SYSTEM_PROMPT}\\n\\n"
        f"<|USER|>\\n{messages[1]['content']}\\n\\n"
        "<|ASSISTANT|>\\n"
    )


def _generate_object_completion(
    model,
    text_tokenizer,
    formatted_prompt: str,
    generation_tokens: int,
    max_prompt_tokens: int,
    device,
    object_prefix: str = "",
):
    prefix_ids = None
    prefix_attention = None
    prefix_tokens = 0
    prompt_limit = max_prompt_tokens
    if object_prefix:
        prefix_encoded = text_tokenizer(
            object_prefix,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prefix_ids = prefix_encoded["input_ids"]
        prefix_attention = prefix_encoded.get("attention_mask")
        if prefix_attention is None:
            prefix_attention = torch.ones_like(prefix_ids)
        prefix_tokens = int(prefix_ids.shape[1])
        prompt_limit = max(128, max_prompt_tokens - prefix_tokens)
    inputs = text_tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_limit,
    )
    prompt_len = inputs["input_ids"].shape[1]
    if prefix_ids is not None:
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs["input_ids"])
        inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix_ids], dim=1)
        inputs["attention_mask"] = torch.cat([attention_mask, prefix_attention], dim=1)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=generation_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        stopping_criteria=StoppingCriteriaList([
            _JsonObjectStoppingCriteria(text_tokenizer, prompt_len)
        ]),
    )
    completion_ids = output_ids[:, prompt_len:]
    completion_tokens = int(completion_ids.shape[1])
    generated_tokens = int(output_ids.shape[1] - input_len)
    text = text_tokenizer.decode(completion_ids[0], skip_special_tokens=True)
    return {
        "text": text,
        "completion_tokens": completion_tokens,
        "generated_tokens": generated_tokens,
        "object_prefix_tokens": prefix_tokens,
        "hit_generation_cap": generated_tokens >= generation_tokens,
    }


def _extract_any_json_object(text: str) -> dict:
    text = _assistant_completion_region(text)
    decoder = json.JSONDecoder()
    last_error = "no JSON object found"
    first_json = text.find("{")
    indices = [first_json] if first_json >= 0 and not text[:first_json].strip() else [
        index for index, char in enumerate(text) if char == "{"
    ]
    for index in indices:
        if index < 0:
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if isinstance(parsed, dict):
            return parsed
    return {"raw_error": f"no JSON object found: {last_error}"}


def _empty_sir_unit(target: dict, reference: dict | None = None, metadata: dict | None = None) -> dict:
    reference = reference if isinstance(reference, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    source_spans = (
        target.get("source_spans")
        or reference.get("source_spans")
        or metadata.get("source_spans")
        or []
    )
    unit_id = target.get("unit_id") or reference.get("unit_id") or metadata.get("unit_id")
    source_path = (
        target.get("name")
        or target.get("qualified_name")
        or reference.get("source_path")
        or metadata.get("source_path")
        or unit_id
    )
    algorithm = target.get("algorithm", {})
    if not isinstance(algorithm, dict) or not algorithm.get("claim"):
        static_algorithm = reference.get("static_facts", {}).get("algorithm", {})
        if isinstance(static_algorithm, dict) and static_algorithm.get("claim"):
            algorithm = static_algorithm
        else:
            algorithm = {
                "claim": f"Semantic IR unit for {source_path}.",
                "confidence": 0.5,
                "source_spans": source_spans,
            }
    if not isinstance(algorithm, dict):
        algorithm = {
            "claim": f"Semantic IR unit for {source_path}.",
            "confidence": 0.5,
            "source_spans": source_spans,
        }
    algorithm.setdefault("source_spans", source_spans)
    algorithm.setdefault("claim", f"Semantic IR unit for {source_path}.")
    algorithm.setdefault("confidence", 0.5)
    return {
        "unit_id": unit_id,
        "source_spans": source_spans,
        "language": target.get("language") or reference.get("language") or metadata.get("language") or "python",
        "symbol_type": (
            target.get("symbol_type")
            or reference.get("symbol_type")
            or metadata.get("symbol_type")
            or "module"
        ),
        "name": target.get("name") or target.get("qualified_name") or source_path,
        "qualified_name": target.get("qualified_name") or target.get("name") or source_path,
        "algorithm": algorithm,
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
            category: [] for category in FIELD_WISE_DATA_ML_CATEGORIES
        },
        "hazards": [],
        "uncertainty": [],
        "confidence": target.get("confidence", algorithm.get("confidence", 0.5)),
    }


def _normalize_field_value(field: str, value):
    if field == "data_ml_details":
        value = value if isinstance(value, dict) else {}
        return {
            category: value.get(category, [])
            if isinstance(value.get(category), list)
            else []
            for category in FIELD_WISE_DATA_ML_CATEGORIES
        }
    return value if isinstance(value, list) else []


def _limit_field_target(field: str, value, limit: int):
    if limit <= 0:
        return value
    if field == "data_ml_details":
        value = value if isinstance(value, dict) else {}
        return {
            category: value.get(category, [])[:limit]
            if isinstance(value.get(category), list)
            else []
            for category in FIELD_WISE_DATA_ML_CATEGORIES
        }
    if isinstance(value, list):
        return value[:limit]
    return value


def _generate_field_wise_candidate(
    model,
    tokenizer,
    text_tokenizer,
    prompt: dict,
    field_generation_tokens: int,
    max_field_prompt_tokens: int,
    field_target_mode: str,
    field_target_limit: int,
    field_target_max_chunks: int,
    field_target_chunk_fields: set[str],
    field_object_prefix_mode: str,
    device,
):
    target = prompt.get("reward_reference", {}).get("compact_target", {})
    if not isinstance(target, dict):
        return (
            {"unit_id": "<unparseable>", "raw_error": "missing compact_target"},
            [],
            0,
            0,
            False,
        )
    sir_unit = _empty_sir_unit(
        target,
        prompt.get("reward_reference", {}),
        prompt.get("metadata", {}),
    )
    field_reports = []
    total_completion_tokens = 0
    total_generated_tokens = 0
    any_cap_hit = False
    static_facts = prompt.get("reward_reference", {}).get("static_facts", {})
    static_facts = static_facts if isinstance(static_facts, dict) else {}
    for field in (*FIELD_WISE_LIST_FIELDS, "data_ml_details"):
        compact_value = target.get(field, {} if field == "data_ml_details" else [])
        static_value = static_facts.get(field, {} if field == "data_ml_details" else [])
        if field_target_mode == "static-facts":
            target_values = [static_value]
            chunk_count = 1
        elif field_target_mode == "static-hints":
            hint_limit = _hint_limit(field, field_target_limit, compact_value)
            available_hints = _hint_count(field, static_value)
            max_chunks_for_field = (
                field_target_max_chunks
                if not field_target_chunk_fields or field in field_target_chunk_fields
                else 1
            )
            chunk_count = max(1, min(max_chunks_for_field, (available_hints + hint_limit - 1) // hint_limit))
            target_values = [
                {
                    "output_budget": hint_limit,
                    "chunk_index": chunk_index,
                    "chunk_count": chunk_count,
                    "compact_target": compact_value if chunk_index == 0 else ([] if field != "data_ml_details" else {}),
                    "static_hints": _compact_static_hint_chunk(
                        field,
                        static_value,
                        hint_limit,
                        chunk_index,
                    ),
                }
                for chunk_index in range(chunk_count)
            ]
        else:
            target_values = [compact_value]
            chunk_count = 1
        if field_target_mode != "static-hints":
            hint_limit = field_target_limit
            target_values = [
                _limit_field_target(field, target_value, field_target_limit)
                for target_value in target_values
            ]
        merged_value = {} if field == "data_ml_details" else []
        for chunk_index, target_value in enumerate(target_values):
            formatted = _format_field_prompt(
                prompt["prompt"],
                field,
                target_value,
                field_target_mode,
                tokenizer,
            )
            result = _generate_object_completion(
                model,
                text_tokenizer,
                formatted,
                field_generation_tokens,
                max_field_prompt_tokens,
                device,
                object_prefix=f'{{"{field}":' if field_object_prefix_mode == "object" else "",
            )
            parsed = _extract_any_json_object(result["text"])
            raw_value = parsed.get(field) if isinstance(parsed, dict) else None
            value = _normalize_field_value(field, raw_value)
            if isinstance(parsed, dict) and not parsed.get("raw_error"):
                merged_value = _merge_field_value(field, merged_value, value)
            total_completion_tokens += result["completion_tokens"]
            total_generated_tokens += result["generated_tokens"]
            any_cap_hit = any_cap_hit or result["hit_generation_cap"]
            empty = (
                not any(value.values())
                if isinstance(value, dict)
                else not bool(value)
            )
            field_reports.append(
                {
                    "field": field,
                    "field_target_mode": field_target_mode,
                    "field_target_limit": field_target_limit,
                    "effective_field_target_limit": hint_limit,
                    "field_target_chunk_index": chunk_index,
                    "field_target_chunk_count": chunk_count,
                    "field_target_chunk_enabled": (
                        not field_target_chunk_fields or field in field_target_chunk_fields
                    ),
                    "field_object_prefix_mode": field_object_prefix_mode,
                    "parseable": isinstance(parsed, dict) and not parsed.get("raw_error"),
                    "raw_error": parsed.get("raw_error") if isinstance(parsed, dict) else "not_object",
                    "empty": empty,
                    "completion_tokens": result["completion_tokens"],
                    "generated_tokens": result["generated_tokens"],
                    "object_prefix_tokens": result.get("object_prefix_tokens", 0),
                    "hit_generation_cap": result["hit_generation_cap"],
                    "raw_text": result["text"][:1000],
                }
            )
        sir_unit[field] = merged_value
    return sir_unit, field_reports, total_completion_tokens, total_generated_tokens, any_cap_hit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", default=".")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-out")
    parser.add_argument("--repaired-out")
    parser.add_argument("--prompt-file")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--record-id", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-faithfulness-repair", action="store_true")
    parser.add_argument(
        "--faithfulness-repair-mode",
        choices=["schema-only", "compact-target", "full-static"],
        default="schema-only",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["full-json", "field-wise"],
        default="full-json",
    )
    parser.add_argument("--field-max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--field-target-mode",
        choices=["compact", "static-facts", "static-hints"],
        default="compact",
    )
    parser.add_argument("--field-target-limit", type=int, default=0)
    parser.add_argument("--field-target-max-chunks", type=int, default=1)
    parser.add_argument("--field-target-chunk-fields", default="")
    parser.add_argument(
        "--field-object-prefix-mode",
        choices=["off", "object"],
        default="off",
    )
    parser.add_argument(
        "--schema-prefix-mode",
        choices=["off", "identity", "identity-algorithm", "schema-scaffold"],
        default="schema-scaffold",
    )
    args = parser.parse_args()

    root = Path(args.training_dir)
    config = json.loads((root / "unsloth_sft_config.json").read_text(encoding="utf-8"))
    prompts_path = Path(args.prompt_file) if args.prompt_file else root / "rl_prompts.jsonl"
    prompts = _read_jsonl(prompts_path)
    if args.record_id:
        requested = set(args.record_id)
        prompts = [
            prompt for prompt in prompts
            if prompt["record_id"] in requested
            or prompt.get("metadata", {}).get("record_id") in requested
            or prompt.get("metadata", {}).get("unit_id") in requested
        ]
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=config["training"]["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        load_in_16bit=config.get("load_in_16bit", False),
    )
    FastLanguageModel.for_inference(model)
    tokenizer.truncation_side = "left"
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    text_tokenizer.truncation_side = "left"
    generation_tokens = min(
        args.max_new_tokens,
        max(config["training"]["max_seq_length"] - 128, 1),
    )
    max_prompt_tokens = max(128, config["training"]["max_seq_length"] - generation_tokens)
    field_generation_tokens = min(
        args.field_max_new_tokens,
        max(config["training"]["max_seq_length"] - 128, 1),
    )
    max_field_prompt_tokens = max(
        128,
        config["training"]["max_seq_length"] - field_generation_tokens,
    )
    raw_rows = []
    repaired_rows = []
    for prompt in prompts:
        device = "cuda" if torch.cuda.is_available() else None
        metadata = prompt["metadata"]
        field_generation_reports = []
        if args.generation_mode == "field-wise":
            (
                raw_sir_unit,
                field_generation_reports,
                completion_tokens,
                generated_tokens,
                hit_generation_cap,
            ) = _generate_field_wise_candidate(
                model,
                tokenizer,
                text_tokenizer,
                prompt,
                field_generation_tokens,
                max_field_prompt_tokens,
                args.field_target_mode,
                args.field_target_limit,
                max(args.field_target_max_chunks, 1),
                _parse_field_set(args.field_target_chunk_fields),
                args.field_object_prefix_mode,
                device,
            )
            schema_prefix = ""
            schema_prefix_tokens = 0
            text = json.dumps(raw_sir_unit, separators=(",", ":"))
        else:
            formatted_prompt = _format_generation_prompt(prompt["prompt"], tokenizer)
            schema_prefix = _schema_prefix(prompt, args.schema_prefix_mode)
            inputs, prompt_len, schema_prefix_tokens = _encode_generation_inputs(
                text_tokenizer,
                formatted_prompt,
                schema_prefix,
                max_prompt_tokens,
                device,
            )
            input_len = int(inputs["input_ids"].shape[1])
            output_ids = model.generate(
                **inputs,
                max_new_tokens=generation_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                stopping_criteria=StoppingCriteriaList([
                    _JsonObjectStoppingCriteria(text_tokenizer, prompt_len)
                ]),
            )
            completion_ids = output_ids[:, prompt_len:]
            generated_tokens = int(output_ids.shape[1] - input_len)
            completion_tokens = int(completion_ids.shape[1])
            text = text_tokenizer.decode(completion_ids[0], skip_special_tokens=True)
            hit_generation_cap = generated_tokens >= generation_tokens
            raw_sir_unit = _extract_json_object(text)
        raw_parse_error = raw_sir_unit.get("raw_error")
        repair_input = json.loads(json.dumps(raw_sir_unit))
        sir_unit = _repair_sir_unit(
            repair_input,
            metadata,
            prompt.get("reward_reference", {}),
        )
        if not args.no_faithfulness_repair:
            sir_unit = _apply_faithfulness_repair(
                sir_unit,
                prompt.get("reward_reference", {}).get("static_facts", {}),
                prompt.get("reward_reference", {}).get("compact_target", {}),
                args.faithfulness_repair_mode,
            )
        base_row = {
            "record_id": prompt["record_id"],
            "dataset_record_id": metadata.get("record_id") or metadata.get("unit_id"),
            "unit_id": metadata.get("unit_id"),
            "source_path": metadata.get("source_path"),
            "source_repo_id": metadata.get("source_repo_id"),
            "source_repo_path": metadata.get("source_repo_path"),
            "source_repo_location": metadata.get("source_repo_location"),
            "language": metadata.get("language", "python"),
            "profile": metadata.get("profile"),
            "zoom": metadata.get("zoom"),
            "raw_text": text,
            "completion_tokens": completion_tokens,
            "generated_tokens": generated_tokens,
            "hit_generation_cap": hit_generation_cap,
            "raw_parse_error": raw_parse_error,
            "generation_config": {
                "generation_mode": args.generation_mode,
                "field_target_mode": args.field_target_mode,
                "field_target_limit": args.field_target_limit,
                "field_target_max_chunks": max(args.field_target_max_chunks, 1),
                "field_target_chunk_fields": sorted(_parse_field_set(args.field_target_chunk_fields)),
                "field_object_prefix_mode": args.field_object_prefix_mode,
                "max_new_tokens": args.max_new_tokens,
                "effective_max_new_tokens": generation_tokens,
                "field_max_new_tokens": args.field_max_new_tokens,
                "effective_field_max_new_tokens": field_generation_tokens,
                "seed": args.seed,
                "faithfulness_repair": not args.no_faithfulness_repair,
                "faithfulness_repair_mode": args.faithfulness_repair_mode,
                "schema_prefix_mode": args.schema_prefix_mode,
                "schema_prefix_applied": bool(schema_prefix),
                "schema_prefix_tokens": schema_prefix_tokens,
            },
        }
        raw_rows.append(
            {
                **base_row,
                "raw_parseable": not bool(raw_sir_unit.get("raw_error")),
                "repair_applied": False,
                "sir_unit": raw_sir_unit,
                "raw_sir_unit": raw_sir_unit,
                "field_generation_reports": field_generation_reports,
            }
        )
        repaired_rows.append(
            {
                **base_row,
                "raw_parseable": not bool(raw_sir_unit.get("raw_error")),
                "repair_applied": True,
                "raw_sir_unit": raw_sir_unit,
                "sir_unit": sir_unit,
            }
        )
    _write_jsonl(Path(args.out), repaired_rows)
    if args.raw_out:
        _write_jsonl(Path(args.raw_out), raw_rows)
    if args.repaired_out:
        _write_jsonl(Path(args.repaired_out), repaired_rows)
    return 0


DATA_ML_DETAIL_CATEGORIES = (
    "losses",
    "model_architecture",
    "tensor_shapes",
    "training_loops",
    "optimizer_scheduler",
    "metrics",
    "checkpointing",
)
LIST_FIELDS = (
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
)


def _repair_sir_unit(unit: dict, metadata: dict, reference: dict) -> dict:
    if not isinstance(unit, dict):
        unit = {}
    elif unit.get("raw_error"):
        unit = {}
    source_spans = reference.get("source_spans") or unit.get("source_spans") or []
    source_path = metadata.get("source_path") or reference.get("source_path") or "<unknown>"
    unit["unit_id"] = metadata.get("unit_id") or reference.get("unit_id") or unit.get("unit_id")
    unit["source_spans"] = source_spans
    unit["language"] = metadata.get("language") or unit.get("language") or "python"
    unit["symbol_type"] = metadata.get("symbol_type") or unit.get("symbol_type") or "module"
    unit["name"] = unit.get("name") or metadata.get("qualified_name") or source_path
    unit["qualified_name"] = metadata.get("qualified_name") or unit.get("qualified_name") or unit["name"]
    if not isinstance(unit.get("algorithm"), dict):
        unit["algorithm"] = {
            "claim": f"Semantic IR unit for {unit['qualified_name']}.",
            "confidence": 0.5,
            "source_spans": source_spans,
        }
    unit["algorithm"].setdefault("source_spans", source_spans)
    unit["algorithm"].setdefault("confidence", 0.5)
    unit["algorithm"].setdefault("claim", f"Semantic IR unit for {unit['qualified_name']}.")
    for field in LIST_FIELDS:
        if not isinstance(unit.get(field), list):
            unit[field] = []
    if not isinstance(unit.get("data_ml_details"), dict):
        unit["data_ml_details"] = {}
    for category in DATA_ML_DETAIL_CATEGORIES:
        if not isinstance(unit["data_ml_details"].get(category), list):
            unit["data_ml_details"][category] = []
    unit["confidence"] = unit.get("confidence", unit["algorithm"].get("confidence", 0.5))
    return unit


def _apply_faithfulness_repair(
    unit: dict,
    static_facts: dict,
    compact_target: dict | None = None,
    repair_mode: str = "schema-only",
) -> dict:
    if not isinstance(unit, dict) or unit.get("raw_error"):
        return unit
    if repair_mode == "schema-only":
        repair_source = {}
    elif repair_mode == "full-static":
        repair_source = static_facts if isinstance(static_facts, dict) else {}
    elif repair_mode == "compact-target":
        repair_source = compact_target if isinstance(compact_target, dict) else {}
    else:
        raise ValueError("repair_mode must be 'schema-only', 'compact-target', or 'full-static'")
    if isinstance(repair_source.get("algorithm"), dict):
        unit["algorithm"] = repair_source["algorithm"]
    for field in LIST_FIELDS:
        if isinstance(repair_source.get(field), list):
            unit[field] = repair_source[field]
    data_ml_details = repair_source.get("data_ml_details", {})
    if isinstance(data_ml_details, dict):
        unit["data_ml_details"] = {
            category: data_ml_details.get(category, [])
            if isinstance(data_ml_details.get(category), list)
            else []
            for category in DATA_ML_DETAIL_CATEGORIES
        }
    unit["confidence"] = unit.get("confidence", unit.get("algorithm", {}).get("confidence", 0.7))
    return unit


_GENERATION_SYSTEM_PROMPT = (
    "You generate one valid Semantic Mirror SIR JSON unit. Use every required top-level "
    "schema key exactly once. Preserve only source-backed static facts supplied in the "
    "prompt. Do not invent behavior. Return minified JSON only: begin with {, end with }, "
    "and do not wrap the object in Markdown fences."
)


def _format_generation_prompt(user_prompt: str, tokenizer=None) -> str:
    messages = [
        {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{user_prompt}\\n\\n"
                "Return only one complete minified SIR JSON object. Copy the final SIR JSON "
                "object between FINAL_SIR_JSON_START and FINAL_SIR_JSON_END exactly: same top-level keys, source-backed values, "
                "compact list lengths, and exact identity fields. Do not shorten unit_id or "
                "qualified_name. Do not add safety_report, summary, code_analysis, analysis, "
                "output_template, FINAL_SIR_JSON_START, FINAL_SIR_JSON_END, or any key outside "
                "the final SIR JSON. The answer "
                "must start with {\\\"unit_id\\\". Do not continue the input JSON or use Markdown "
                "fences."
            ),
        },
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        except Exception:
            pass
    return (
        f"<|SYSTEM|>\\n{_GENERATION_SYSTEM_PROMPT}\\n\\n"
        f"<|USER|>\\n{user_prompt}\\n\\n"
        "Return only one complete minified SIR JSON object. Copy the final SIR JSON object between "
        "FINAL_SIR_JSON_START and FINAL_SIR_JSON_END exactly: same top-level keys, "
        "source-backed values, compact list lengths, "
        "and exact identity fields. Do not shorten unit_id or qualified_name. "
        "Do not add safety_report, summary, code_analysis, analysis, output_template, "
        "FINAL_SIR_JSON_START, FINAL_SIR_JSON_END, or any key outside the final SIR JSON. "
        "The answer must start with {\\\"unit_id\\\". Do not continue "
        "the input JSON or use Markdown fences.\\n\\n"
        "<|ASSISTANT|>\\n"
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\\n" for row in rows),
        encoding="utf-8",
    )


def _extract_json_object(text: str) -> dict:
    text = _assistant_completion_region(text)
    decoder = json.JSONDecoder()
    last_error = "no JSON object found"
    first_json = text.find("{")
    indices = [first_json] if first_json >= 0 and not text[:first_json].strip() else [
        index for index, char in enumerate(text) if char == "{"
    ]
    for index in indices:
        if index < 0:
            continue
        char = text[index]
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if isinstance(parsed, dict) and "unit_id" in parsed:
            return parsed
    return {"unit_id": "<unparseable>", "raw_error": f"no SIR JSON object found: {last_error}"}


def _has_complete_json_object(text: str) -> bool:
    start = text.find("{")
    if start < 0:
        return False
    depth = 0
    in_string = False
    escape = False
    for char in text[start:]:
        if in_string:
            if escape:
                escape = False
            elif char == "\\\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return True
            if depth < 0:
                return False
    return False


def _assistant_completion_region(text: str) -> str:
    for marker in ("<|ASSISTANT|>", "<|assistant|>"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
    return text.strip()


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _score_candidates_script() -> str:
    return '''"""Score generated Semantic IR candidates using deterministic reward rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_mirror.rewards import score_document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    repo = Path(args.repo)
    outputs = []
    for line in Path(args.candidates).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        document = {
            "schema_version": "0.1.0",
            "source_path": record["source_path"],
            "language": record.get("language", "python"),
            "profile": record["profile"],
            "zoom": record["zoom"],
            "units": [record["sir_unit"]],
            "unsupported_reasons": [],
        }
        outputs.append({"record_id": record.get("record_id"), "score": score_document(document, repo_path=repo)})
    Path(args.out).write_text(
        "".join(json.dumps(record, sort_keys=True) + "\\n" for record in outputs),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _validate_jsonl(path: Path, kind: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines and kind != "preference_pairs":
        issues.append({"kind": "empty_jsonl", "path": path.name})
    for index, line in enumerate(lines, start=1):
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(
                {
                    "kind": "invalid_jsonl",
                    "path": path.name,
                    "line": index,
                    "error": str(exc),
                }
            )
    return issues


def _validate_sft_config(path: Path) -> list[dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    issues: list[dict[str, Any]] = []
    for key in ("base_model", "method", "load_in_4bit", "lora", "training", "inputs"):
        if key not in config:
            issues.append({"kind": "missing_sft_config_key", "key": key})
    if config.get("method") not in {"QLoRA", "bf16 LoRA"}:
        issues.append({"kind": "unsupported_sft_method", "actual": config.get("method")})
    if config.get("load_in_4bit") and config.get("load_in_16bit"):
        issues.append({"kind": "conflicting_precision_flags"})
    if config.get("training", {}).get("max_seq_length", 0) <= 0:
        issues.append({"kind": "invalid_max_seq_length"})
    return issues


def _validate_reward_config(path: Path) -> list[dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    issues: list[dict[str, Any]] = []
    if config.get("objective") != "faithfulness_first_compactness_second":
        issues.append({"kind": "invalid_reward_objective"})
    for key in ("positive_rewards", "penalties", "guardrails"):
        if key not in config:
            issues.append({"kind": "missing_reward_config_key", "key": key})
    return issues


def _validate_python_syntax(path: Path) -> list[dict[str, Any]]:
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [{"kind": "python_syntax_error", "path": path.name, "error": str(exc)}]
    return []


def _training_validation_report(root: Path, issues: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mode": "training_validate",
        "training_dir": str(root),
        "passed": not issues,
        "issues": issues,
    }


def _check(
    name: str,
    passed: bool,
    *,
    required: bool,
    actual: Any,
    detail: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "required": required,
        "actual": actual,
        "detail": detail,
    }


def _module_detail(module: str, *, module_probe: ModuleProbe | None) -> dict[str, Any]:
    if module_probe is not None:
        return {"importable": bool(module_probe(module))}
    try:
        if importlib.util.find_spec(module) is None:
            return {"importable": False, "error": "module spec not found"}
        imported = importlib.import_module(module)
        return {
            "importable": True,
            "version": getattr(imported, "__version__", None),
        }
    except (ImportError, AttributeError, ValueError):
        return {"importable": False, "error": "module import failed"}
    except Exception as exc:
        return {"importable": False, "error": str(exc)}


def _module_available(module: str, *, module_probe: ModuleProbe | None) -> bool:
    return bool(_module_detail(module, module_probe=module_probe).get("importable"))


def _probe_python_runtime(
    python_executable: str,
    required_modules: tuple[str, ...],
) -> dict[str, Any]:
    script = r'''
from __future__ import annotations

import importlib
import importlib.util
import json
import sys

modules = json.loads(sys.argv[1])
module_details = {}
for module in modules:
    try:
        if importlib.util.find_spec(module) is None:
            module_details[module] = {"importable": False, "error": "module spec not found"}
            continue
        imported = importlib.import_module(module)
        module_details[module] = {
            "importable": True,
            "version": getattr(imported, "__version__", None),
        }
    except Exception as exc:
        module_details[module] = {"importable": False, "error": str(exc)}

torch_info = {"importable": False}
try:
    import torch

    torch_info = {
        "importable": True,
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "bf16_supported": None,
        "device_count": 0,
        "devices": [],
    }
    if torch_info["cuda_available"]:
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_bf16_supported):
            torch_info["bf16_supported"] = bool(is_bf16_supported())
        torch_info["device_count"] = torch.cuda.device_count()
        devices = []
        for index in range(torch_info["device_count"]):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": [props.major, props.minor],
                }
            )
        torch_info["devices"] = devices
except Exception as exc:
    torch_info = {"importable": False, "error": str(exc)}

print(json.dumps({
    "ok": True,
    "python_executable": sys.executable,
    "python_version": ".".join(str(part) for part in sys.version_info[:3]),
    "module_details": module_details,
    "torch": torch_info,
}))
'''
    try:
        result = subprocess.run(
            [python_executable, "-c", script, json.dumps(list(required_modules))],
            check=False,
            capture_output=True,
            encoding="utf-8",
            timeout=60,
        )
    except Exception as exc:
        return _failed_python_runtime_probe(python_executable, required_modules, str(exc))
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        return _failed_python_runtime_probe(python_executable, required_modules, error)
    json_lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
    try:
        payload = json.loads(json_lines[-1] if json_lines else result.stdout)
    except (IndexError, json.JSONDecodeError) as exc:
        return _failed_python_runtime_probe(
            python_executable,
            required_modules,
            f"runtime probe did not return JSON: {exc}",
        )
    payload.setdefault("ok", True)
    payload.setdefault("python_executable", python_executable)
    payload.setdefault("module_details", {})
    payload.setdefault("torch", {"importable": False, "error": "missing torch probe"})
    return payload


def _failed_python_runtime_probe(
    python_executable: str,
    required_modules: tuple[str, ...],
    error: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "python_executable": python_executable,
        "python_version": "unknown",
        "probe_error": error,
        "module_details": {
            module: {"importable": False, "error": "runtime probe failed"}
            for module in required_modules
        },
        "torch": {"importable": False, "error": error},
    }


def _python_version_supported_for_unsloth(version: str) -> bool:
    parts = version.split(".")
    if len(parts) < 2:
        return False
    major = _parse_int(parts[0])
    minor = _parse_int(parts[1])
    if major is None or minor is None:
        return False
    parsed = (major, minor)
    return UNSLOTH_PYTHON_MIN <= parsed < UNSLOTH_PYTHON_MAX_EXCLUSIVE


def _probe_torch(*, torch_probe: TorchProbe | None) -> dict[str, Any]:
    if torch_probe is not None:
        return torch_probe()
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:
        return {"importable": False, "error": str(exc)}

    info: dict[str, Any] = {
        "importable": True,
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "bf16_supported": None,
        "device_count": 0,
        "devices": [],
    }
    if info["cuda_available"]:
        try:
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(is_bf16_supported):
                info["bf16_supported"] = bool(is_bf16_supported())
            info["device_count"] = torch.cuda.device_count()
            info["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(index).total_memory / (1024**3),
                        2,
                    ),
                    "compute_capability": [
                        torch.cuda.get_device_properties(index).major,
                        torch.cuda.get_device_properties(index).minor,
                    ],
                }
                for index in range(info["device_count"])
            ]
        except Exception as exc:
            info["device_probe_error"] = str(exc)
    return info


def _probe_nvidia_smi(*, nvidia_smi_runner: NvidiaSmiRunner | None) -> dict[str, Any]:
    if nvidia_smi_runner is None and shutil.which("nvidia-smi") is None:
        return {"available": False, "error": "nvidia-smi not found on PATH"}
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = (
            nvidia_smi_runner(command)
            if nvidia_smi_runner is not None
            else subprocess.run(command, check=False, capture_output=True, encoding="utf-8")
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if result.returncode != 0:
        return {
            "available": False,
            "error": result.stderr.strip() or result.stdout.strip(),
            "returncode": result.returncode,
        }
    devices = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3:
            devices.append(
                {
                    "name": fields[0],
                    "memory_total_mb": _parse_int(fields[1]),
                    "driver_version": fields[2],
                }
            )
    return {"available": True, "devices": devices}


def _training_launch_command(
    root: Path,
    *,
    stage: str,
    output_dir: Path,
    python_executable: str | None,
    model_name_or_path: str | None,
    beta: float,
    max_steps: int | None,
    kl_coef: float,
    schema_prefix_mode: str,
    resume_from_checkpoint: str | None,
    seed: int | None,
) -> list[str]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    script_key = {
        "sft": "sft_script",
        "dpo": "preference_script",
        "rl": "rl_script",
    }[stage]
    script = root / manifest["files"][script_key]
    command = [
        python_executable or sys.executable,
        str(script),
        "--training-dir",
        str(root),
        "--output-dir",
        str(output_dir),
    ]
    if stage == "dpo":
        if model_name_or_path:
            command.extend(["--model-name-or-path", model_name_or_path])
        command.extend(["--beta", str(beta)])
        if max_steps is not None:
            command.extend(["--max-steps", str(max_steps)])
        if resume_from_checkpoint:
            command.extend(["--resume-from-checkpoint", resume_from_checkpoint])
    if stage == "rl":
        if model_name_or_path:
            command.extend(["--model-name-or-path", model_name_or_path])
        command.extend(["--kl-coef", str(kl_coef)])
        command.extend(["--schema-prefix-mode", schema_prefix_mode])
    if stage in {"sft", "rl"} and max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    if stage == "sft" and resume_from_checkpoint:
        command.extend(["--resume-from-checkpoint", resume_from_checkpoint])
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command


def _run_training_subprocess(
    command: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr:
        result = subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr, check=False)
    return result.returncode


def _write_optional_report(report: dict[str, Any], report_out: Path | str | None) -> None:
    if report_out is None:
        return
    path = Path(report_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_package_manifest(out: Path, manifest: dict[str, Any]) -> None:
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _replace_directory(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _load_training_env(env_file: Path | str | None) -> tuple[dict[str, str], Path | None]:
    candidates: list[Path] = []
    if env_file is not None:
        candidates.append(Path(env_file))
    candidates.append(Path.cwd() / ".env")
    for path in candidates:
        if not path.exists():
            continue
        values: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        return values, path.resolve()
    return {}, None


def _training_command_hints(
    missing_modules: list[str],
    *,
    python_supported: bool,
    require_gpu: bool,
) -> list[str]:
    hints = []
    if not python_supported:
        hints.append(
            f"Create the training environment with Python {UNSLOTH_PYTHON_RANGE}; "
            "the packaging Python is not a supported Unsloth runtime."
        )
    if missing_modules:
        hints.append(
            "Install CUDA-compatible training dependencies: unsloth, trl, datasets, "
            "transformers, torch, bitsandbytes, and peft."
        )
    if require_gpu:
        hints.append("Run on a CUDA machine before launching the default Qwen3-family LoRA scripts.")
    return hints


def _training_audit_repro_command(
    training_dir: Path,
    *,
    env_file: Path | None,
    require_gpu: bool,
    require_hf_token: bool,
    python_executable: str | None,
) -> list[str]:
    command = ["uv", "run", "semantic-mirror", "train", "audit", str(training_dir)]
    if env_file is not None:
        command.extend(["--env-file", str(env_file)])
    if not require_gpu:
        command.append("--allow-cpu")
    if require_hf_token:
        command.append("--require-hf-token")
    if python_executable is not None:
        command.extend(["--python-executable", python_executable])
    return command


def _training_audit_blocker_summary(
    failed_required_checks: list[str],
    *,
    python_supported: bool,
    missing_modules: list[str],
    torch_info: dict[str, Any],
    require_gpu: bool,
) -> list[str]:
    summary = []
    if "python_version_supported_for_unsloth" in failed_required_checks and not python_supported:
        summary.append(f"Python runtime is outside the supported {UNSLOTH_PYTHON_RANGE} range.")
    if "required_training_modules" in failed_required_checks and missing_modules:
        summary.append("Missing required training modules: " + ", ".join(missing_modules) + ".")
    if "torch_importable" in failed_required_checks:
        error = torch_info.get("error")
        summary.append(
            "PyTorch is not importable"
            + (f": {error}." if error else ".")
        )
    if "torch_cuda_available" in failed_required_checks and require_gpu:
        summary.append("PyTorch CUDA is not available for the audited runtime.")
    return summary


def _collect_training_metrics(run: Path) -> dict[str, Any]:
    series: dict[str, list[dict[str, Any]]] = {name: [] for name in DIAGNOSTIC_PLOT_SPECS}
    sources: dict[str, dict[str, Any]] = {}
    for log_path in sorted(run.rglob("*.log")):
        log_records = _parse_training_log_records(log_path)
        if not log_records:
            continue
        stage = _stage_from_path(log_path)
        sources[str(log_path)] = {"kind": "training_log", "records": len(log_records)}
        for step, record in enumerate(log_records, start=1):
            if "loss" in record and stage in {"sft", "dpo", "unknown"}:
                loss_stage = "sft" if stage == "unknown" else stage
                series[f"{loss_stage}_loss"].append(
                    _point(step, record["loss"], log_path, label=loss_stage)
                )
            if stage == "dpo" and "reward_accuracy" in record:
                series["dpo_reward_accuracy"].append(
                    _point(step, record["reward_accuracy"], log_path, label=stage)
                )
    for json_path in sorted(run.rglob("*.json")):
        payload = _read_json_file(json_path)
        if payload is None:
            continue
        if payload.get("stage") == "rl" and isinstance(payload.get("history"), list):
            history = payload["history"]
            sources[str(json_path)] = {"kind": "rl_training_report", "records": len(history)}
            for index, row in enumerate(history, start=1):
                if "reward" in row:
                    series["rl_reward"].append(_point(index, row["reward"], json_path, label="rl"))
                if "raw_parseable" in row:
                    series["rl_parseability"].append(
                        _point(index, 1.0 if row["raw_parseable"] else 0.0, json_path, label="rl")
                    )
                if "loss" in row:
                    series["eval_metrics"].append(
                        _point(index, row["loss"], json_path, label="rl_loss")
                    )
        if isinstance(payload.get("log_history"), list):
            log_history = payload["log_history"]
            stage = _stage_from_path(json_path)
            sources[str(json_path)] = {
                "kind": "trainer_state",
                "records": len(log_history),
            }
            for index, row in enumerate(log_history, start=1):
                if "loss" in row and stage in {"sft", "dpo", "unknown"}:
                    loss_stage = "sft" if stage == "unknown" else stage
                    series[f"{loss_stage}_loss"].append(
                        _point(index, row["loss"], json_path, label=loss_stage)
                    )
                if stage == "dpo" and "rewards/accuracies" in row:
                    series["dpo_reward_accuracy"].append(
                        _point(index, row["rewards/accuracies"], json_path, label="dpo")
                    )
        if isinstance(payload.get("metrics"), dict):
            metrics = payload["metrics"]
            sources[str(json_path)] = {"kind": "evaluation_report", "records": len(metrics)}
            metric_index = len(series["eval_metrics"]) + 1
            for key in (
                "average_static_faithfulness_score",
                "hallucination_penalties",
                "total_score",
            ):
                if key in metrics:
                    series["eval_metrics"].append(
                        _point(metric_index, metrics[key], json_path, label=key)
                    )
                    metric_index += 1
            for key in ("schema_validity", "heldout_unit_coverage"):
                if key in metrics:
                    series["schema_coverage"].append(
                        _point(len(series["schema_coverage"]) + 1, metrics[key], json_path, label=key)
                    )
    for jsonl_path in sorted(run.rglob("*candidates*.jsonl")):
        rows = _read_jsonl(jsonl_path)
        if not rows:
            continue
        sources[str(jsonl_path)] = {"kind": "candidate_jsonl", "records": len(rows)}
        for index, row in enumerate(rows, start=1):
            raw_text = str(row.get("raw_text") or row.get("output") or "")
            if raw_text:
                series["generation_lengths"].append(
                    _point(index, len(raw_text), jsonl_path, label="raw_text")
                )
    return {"series": series, "sources": sources}


def _parse_training_log_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _stage_from_path(path: Path) -> str:
    lowered = str(path).lower()
    for stage in ("sft", "dpo", "rl"):
        if stage in lowered:
            return stage
    return "unknown"


def _point(index: int, value: Any, source: Path, *, label: str) -> dict[str, Any]:
    numeric = _coerce_float(value)
    return {
        "x": index,
        "y": 0.0 if numeric is None else numeric,
        "label": label,
        "source": str(source),
        "missing": numeric is None,
    }


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _contract_gate(
    name: str,
    passed: bool,
    *,
    actual: Any = None,
    expected: Any = True,
    evidence: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual if actual is not None else bool(passed),
        "expected": expected,
        "evidence": evidence,
    }


def _full_eval_contract_status_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Semantic Mirror Full-Eval Contract Status",
        "",
        f"- Run directory: `{report['run_dir']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Passed: `{report['passed']}`",
        "",
        "## Requested Steps",
        "",
        "| Stage | Requested | Manifest | Matches |",
        "| --- | ---: | ---: | --- |",
    ]
    for stage in ("sft", "dpo", "rl"):
        stage_status = report["stage_status"][stage]
        lines.append(
            f"| `{stage}` | {stage_status['requested_max_steps']} | "
            f"{stage_status['manifest_max_steps']} | "
            f"{stage_status['manifest_matches_requested_max_steps']} |"
        )
    lines.extend(
        [
            "",
            "## Stage Evidence",
            "",
            "| Stage | Manifest Current | Eval Current | Compare Current | Sample Current |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for stage in ("sft", "dpo", "rl"):
        evidence = report["stage_evidence_summary"][stage]
        lines.append(
            f"| `{stage}` | `{evidence['manifest_current']}` | "
            f"`{evidence['eval_current']}` | `{evidence['compare_current']}` | "
            f"`{evidence['sample_current']}` |"
        )
    recovery_status = report.get("stage_recovery_status") or {}
    if recovery_status:
        lines.extend(
            [
                "",
                "## Stage Recovery",
                "",
                "| Stage | Action | Latest Checkpoint | Missing Current Artifacts |",
                "| --- | --- | --- | --- |",
            ]
        )
        for stage in ("sft", "dpo", "rl"):
            recovery = recovery_status.get(stage, {})
            missing = recovery.get("missing_current_artifacts") or []
            checkpoint = recovery.get("latest_checkpoint_relative") or recovery.get(
                "latest_checkpoint"
            )
            lines.append(
                f"| `{stage}` | `{recovery.get('action')}` | "
                f"`{checkpoint}` | "
                f"{', '.join(f'`{item}`' for item in missing)} |"
            )
    summary_status = report["training_eval_summary_status"]
    lines.extend(
        [
            "",
            "## Training Eval Summary",
            "",
            f"- Requested-step match: `{summary_status['requested_max_steps_match']}`",
            f"- Stage-manifest-step match: `{summary_status['stage_manifest_max_steps_match']}`",
            f"- Actual summary steps: `{json.dumps(summary_status['actual'], sort_keys=True)}`",
        ]
    )
    repo_hygiene = report.get("repo_hygiene_status") or {}
    if repo_hygiene.get("checked"):
        lines.extend(
            [
                "",
                "## Repo Hygiene",
                "",
                f"- Repo root: `{repo_hygiene.get('repo_root')}`",
                f"- Branch: `{repo_hygiene.get('branch')}`",
                f"- Passed: `{repo_hygiene.get('passed')}`",
                f"- Summary: {repo_hygiene.get('summary')}",
                f"- Tracked changes: `{len(repo_hygiene.get('tracked_changes', []))}`",
                f"- Untracked paths: `{len(repo_hygiene.get('untracked', []))}`",
                f"- Allowed ignored local-only paths: `{len(repo_hygiene.get('ignored_allowed', []))}`",
                f"- Unexpected ignored paths: `{len(repo_hygiene.get('ignored_unexpected', []))}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Repo Hygiene",
                "",
                f"- Checked: `{repo_hygiene.get('checked', False)}`",
                f"- Summary: {repo_hygiene.get('summary')}",
            ]
        )
    windows_readiness = report.get("windows_readiness_status") or {}
    lines.extend(
        [
            "",
            "## Windows Readiness",
            "",
            f"- Checked: `{windows_readiness.get('checked', False)}`",
            f"- Passed: `{windows_readiness.get('passed')}`",
            f"- Summary: {windows_readiness.get('summary')}",
            f"- Native audit path: `{windows_readiness.get('windows_audit_path')}`",
            f"- Native passed: `{windows_readiness.get('native_passed')}`",
            f"- Native blocked: `{windows_readiness.get('native_blocked')}`",
            f"- WSL smoke manifest path: `{windows_readiness.get('wsl_smoke_manifest_path')}`",
            f"- WSL smoke manifest mode: `{windows_readiness.get('wsl_smoke_manifest_mode')}`",
            f"- WSL smoke complete: `{windows_readiness.get('wsl_smoke_complete')}`",
            f"- WSL failed checks: `{', '.join(windows_readiness.get('wsl_failed_checks') or []) or 'None'}`",
            f"- WSL missing stage manifests: `{', '.join(windows_readiness.get('wsl_missing_stage_manifests') or []) or 'None'}`",
            f"- WSL missing sample manifests: `{', '.join(windows_readiness.get('wsl_missing_sample_manifests') or []) or 'None'}`",
            f"- WSL diagnostics exists: `{windows_readiness.get('wsl_diagnostics_exists')}`",
        ]
    )
    package_source = report.get("package_source_status") or {}
    lines.extend(
        [
            "",
            "## Package Source Freshness",
            "",
            f"- Checked: `{package_source.get('checked', False)}`",
            f"- Passed: `{package_source.get('passed')}`",
            f"- Summary: {package_source.get('summary')}",
            f"- Evidence path: `{package_source.get('path')}`",
            f"- Freshness commit: `{package_source.get('git_commit')}`",
            f"- Current repo commit: `{package_source.get('current_repo_commit')}`",
            f"- Commit matches repo: `{package_source.get('git_commit_matches_repo')}`",
            f"- Compared scope: `{package_source.get('compared_scope')}`",
            f"- Compared files: `{package_source.get('compared_file_count')}`",
            f"- All compared files match: `{package_source.get('all_compared_files_match')}`",
        ]
    )
    mismatched_files = package_source.get("mismatched_files") or []
    if mismatched_files:
        lines.extend(["", "Mismatched files:"])
        lines.extend(f"- `{path}`" for path in mismatched_files)
    command_manifest = report.get("package_command_manifest_status") or {}
    lines.extend(
        [
            "",
            "## Package Command Manifest",
            "",
            f"- Checked: `{command_manifest.get('checked', False)}`",
            f"- Passed: `{command_manifest.get('passed')}`",
            f"- Summary: {command_manifest.get('summary')}",
            f"- Evidence path: `{command_manifest.get('path')}`",
            f"- Command count: `{command_manifest.get('command_count')}`",
            f"- Training command count: `{command_manifest.get('training_command_count')}`",
            f"- Non-training command count: `{command_manifest.get('non_training_command_count')}`",
        ]
    )
    if command_manifest.get("training_commands"):
        lines.append(
            "- Training commands: "
            + ", ".join(
                f"`{command}`" for command in command_manifest["training_commands"]
            )
        )
    if command_manifest.get("failed_checks"):
        lines.extend(["", "Failed command-manifest checks:"])
        lines.extend(f"- `{check}`" for check in command_manifest["failed_checks"])
    package_metadata = report.get("package_metadata_status") or {}
    lines.extend(
        [
            "",
            "## Package Python Metadata",
            "",
            f"- Checked: `{package_metadata.get('checked', False)}`",
            f"- Passed: `{package_metadata.get('passed')}`",
            f"- Summary: {package_metadata.get('summary')}",
            f"- Evidence path: `{package_metadata.get('path')}`",
            f"- Requires Python: `{package_metadata.get('requires_python')}`",
            f"- Expected training range: `{package_metadata.get('expected_requires_python')}`",
            f"- Excludes Python 3.14: `{package_metadata.get('excludes_python_3_14')}`",
        ]
    )
    human_usefulness = report.get("human_usefulness_status") or {}
    lines.extend(
        [
            "",
            "## Human Usefulness",
            "",
            f"- Checked: `{human_usefulness.get('checked', False)}`",
            f"- Passed: `{human_usefulness.get('passed')}`",
            f"- Summary: {human_usefulness.get('summary')}",
            f"- Suite report path: `{human_usefulness.get('path')}`",
        ]
    )
    coverage_reports = human_usefulness.get("coverage_reports") or []
    if coverage_reports:
        lines.extend(
            [
                "",
                "### Answer Coverage",
                "",
                "| Coverage Report | Passed | Pending | Real Timed Answers | Failed Gates |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for coverage in coverage_reports:
            failed = coverage.get("failed_gates") or []
            lines.append(
                f"| `{coverage.get('path')}` | `{coverage.get('passed')}` | "
                f"{coverage.get('pending_task_count')} | "
                f"{coverage.get('real_timed_answer_records')} | "
                f"{', '.join(f'`{gate}`' for gate in failed)} |"
            )
    collection_plan = human_usefulness.get("collection_plan_status") or {}
    if collection_plan.get("checked"):
        lines.extend(
            [
                "",
                "### Collection Plan",
                "",
                f"- Passed: `{collection_plan.get('passed')}`",
                f"- Summary: {collection_plan.get('summary')}",
                f"- Plan path: `{collection_plan.get('path')}`",
                f"- Answer records: `{collection_plan.get('answer_record_count')}/{collection_plan.get('required_total_answer_records')}`",
            ]
        )
        studies = collection_plan.get("studies") or {}
        if studies:
            lines.extend(
                [
                    "",
                    "| Study | Answer Target Exists | Answer Records | Required Records |",
                    "| --- | --- | ---: | ---: |",
                ]
            )
            for label, study in studies.items():
                lines.append(
                    f"| `{label}` | `{study.get('answer_target_exists')}` | "
                    f"{study.get('answer_records')} | {study.get('required_answer_records')} |"
                )
    lines.extend(
        [
            "",
            "## Contract Scorecard",
            "",
            "| Area | Required | Max Reward | Earned | Passed | Evidence |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in report["contract_scorecard"]:
        lines.append(
            f"| `{row['area']}` | `{row['required']}` | "
            f"{row['max_reward']} | {row['earned_reward']} | `{row['passed']}` | "
            f"{row['evidence']} |"
        )
    reward = report["contract_reward_summary"]
    lines.extend(
        [
            "",
            "## Reward Summary",
            "",
            f"- Required reward: `{reward['required_reward_earned']}/{reward['required_reward_possible']}`",
            f"- Optional reward: `{reward['optional_reward_earned']}/{reward['optional_reward_possible']}`",
            f"- Minimum acceptable required reward: `{reward['minimum_acceptable_required_reward']}`",
            f"- Required reward threshold met: `{reward['required_reward_threshold_met']}`",
            f"- Zero failed required areas: `{reward['zero_failed_required_areas']}`",
            f"- Completion eligible: `{reward['completion_eligible']}`",
        ]
    )
    lines.extend(
        [
            "",
            "## Remaining Items",
            "",
        ]
    )
    if report["remaining_items"]:
        lines.extend(
            [
                "### By Area",
                "",
                "| Area | Count | Gates |",
                "| --- | ---: | --- |",
            ]
        )
        for area, gates in report["remaining_by_area"].items():
            lines.append(
                f"| `{area}` | {len(gates)} | "
                f"{', '.join(f'`{gate}`' for gate in gates)} |"
            )
        lines.extend(["", "### Gate Details", ""])
        lines.extend(
            [
                "| Gate | Actual | Expected | Evidence |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in report["remaining_items"]:
            lines.append(
                f"| `{item['gate']}` | `{json.dumps(item['actual'], sort_keys=True)}` | "
                f"`{json.dumps(item['expected'], sort_keys=True)}` | `{item['evidence']}` |"
            )
        recovery_plan = report.get("remaining_recovery_plan") or []
        if recovery_plan:
            lines.extend(
                [
                    "",
                    "### Recovery Plan",
                    "",
                    "| Gate | Action | Requires Training | Blocked By | Artifacts |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for plan_item in recovery_plan:
                blocked_by = plan_item.get("blocked_by_stages") or []
                artifacts = plan_item.get("artifacts") or []
                lines.append(
                    f"| `{plan_item['gate']}` | `{plan_item['required_action']}` | "
                    f"`{plan_item['requires_training']}` | "
                    f"{', '.join(f'`{stage}`' for stage in blocked_by) or '`None`'} | "
                    f"{', '.join(f'`{artifact}`' for artifact in artifacts)} |"
                )
    else:
        lines.append("All full-eval contract gates are currently proven.")
    resume_status = report.get("resume_inspection_status")
    if resume_status and resume_status.get("exists"):
        lines.extend(
            [
                "",
                "## Resume Inspection",
                "",
                f"- Evidence: `{resume_status['path']}`",
                f"- Reuse enabled: `{resume_status['reuse_stage_outputs_enabled']}`",
                "",
                "| Stage | Action | Requested | Manifest | Checkpoint | Reason |",
                "| --- | --- | ---: | ---: | --- | --- |",
            ]
        )
        for stage in ("sft", "dpo", "rl"):
            decision = resume_status["decisions"].get(stage, {})
            checkpoint = decision.get("resume_from_checkpoint") or {}
            lines.append(
                f"| `{stage}` | `{decision.get('action')}` | "
                f"{decision.get('requested_max_steps')} | "
                f"{decision.get('manifest_max_steps')} | "
                f"`{checkpoint.get('path')}` | `{decision.get('reason')}` |"
            )
    if report.get("next_actions"):
        lines.extend(["", "## Next Actions", ""])
        for action in report["next_actions"]:
            lines.extend(
                [
                    f"### {action['title']}",
                    "",
                    f"- Category: `{action.get('category', 'unspecified')}`",
                    f"- Launches training: `{action.get('launches_training', False)}`",
                    "",
                    action["reason"],
                    "",
                    "```bash",
                    action["command"],
                    "```",
                    "",
                ]
            )
            if action.get("windows_powershell_command"):
                lines.extend(
                    [
                        "```powershell",
                        action["windows_powershell_command"],
                        "```",
                        "",
                    ]
                )
    lines.extend(
        [
            "",
            "## Gate Details",
            "",
            "| Gate | Passed | Evidence |",
            "| --- | --- | --- |",
        ]
    )
    for gate in report["gates"]:
        lines.append(f"| `{gate['name']}` | `{gate['passed']}` | `{gate.get('evidence')}` |")
    lines.append("")
    return "\n".join(lines)


def _full_eval_next_actions(
    run: Path,
    stage_status: dict[str, dict[str, Any]],
    report_status: dict[str, dict[str, Any]],
    sample_status: dict[str, dict[str, Any]],
    diagnostics_status: dict[str, Any],
    repo_hygiene_status: dict[str, Any],
    windows_readiness_status: dict[str, Any],
    package_source_status: dict[str, Any],
    human_usefulness_status: dict[str, Any],
) -> list[dict[str, Any]]:
    package_root = run.parent if run.name == "outputs" else run
    actions: list[dict[str, Any]] = []
    sft_steps = stage_status["sft"]["requested_max_steps"]
    dpo_steps = stage_status["dpo"]["requested_max_steps"]
    rl_steps = stage_status["rl"]["requested_max_steps"]
    env_parts = [
        f"SFT_MAX_STEPS={sft_steps}" if sft_steps is not None else "",
        f"DPO_MAX_STEPS={dpo_steps}" if dpo_steps is not None else "",
        f"RL_MAX_STEPS={rl_steps}" if rl_steps is not None else "",
        "REUSE_STAGE_OUTPUTS=1",
    ]
    if repo_hygiene_status.get("checked") and repo_hygiene_status.get("repo_root"):
        source_repo = _posix_relpath(Path(repo_hygiene_status["repo_root"]), package_root)
        env_parts.append(f"SOURCE_FRESHNESS_REPO_ROOT={_shell_single_quoted(source_repo)}")
    dpo_checkpoint = _latest_checkpoint(run / "semantic-mirror-dpo")
    if (
        dpo_checkpoint is not None
        and not stage_status["dpo"]["manifest_matches_requested_max_steps"]
    ):
        env_parts.append(
            f"DPO_RESUME_FROM_CHECKPOINT={_posix_relpath(dpo_checkpoint, package_root)}"
        )
    base_env = " ".join(part for part in env_parts if part)
    inspect_command = f"{base_env} bash launch/inspect_full_training_eval_resume.sh"
    run_command = f"{base_env} bash launch/run_full_training_eval.sh"
    status_evidence_flags = _contract_status_evidence_flags(
        package_root=package_root,
        repo_hygiene_status=repo_hygiene_status,
        windows_readiness_status=windows_readiness_status,
        package_source_status=package_source_status,
        human_usefulness_status=human_usefulness_status,
    )
    status_command = (
        f"{base_env} PYTHONPATH=src python -m semantic_mirror.cli "
        "train contract-status outputs "
        f"--sft-steps {sft_steps} --dpo-steps {dpo_steps} --rl-steps {rl_steps} "
        f"{status_evidence_flags}"
        "--out outputs/contract_status.json --markdown-out outputs/contract_status.md"
    )
    diagnostics_command = "PYTHONPATH=src python -m semantic_mirror.cli train report outputs --out outputs/diagnostics"
    actions.append(
        {
            "title": "Inspect resume plan",
            "category": "inspection",
            "launches_training": False,
            "reason": "Preview stage reuse and resume decisions before launching training.",
            "command": f"# from {package_root}\n{inspect_command}",
            "windows_powershell_command": _windows_wsl_command(package_root, inspect_command),
        }
    )
    if (
        windows_readiness_status.get("checked")
        and not windows_readiness_status.get("passed")
        and windows_readiness_status.get("native_blocked")
        and not windows_readiness_status.get("wsl_smoke_complete")
    ):
        wsl_failed = windows_readiness_status.get("wsl_failed_checks") or []
        wsl_reason = (
            "Windows-native audit is blocked, and WSL smoke-chain evidence is "
            "missing or incomplete."
        )
        if wsl_failed:
            wsl_reason += " Failed WSL checks: " + ", ".join(
                f"`{check}`" for check in wsl_failed
            ) + "."
        wsl_smoke_command = (
            "powershell -ExecutionPolicy Bypass -File "
            "launch/run_wsl_smoke_chain.ps1 -HeldOutDataset <windows_dataset_dir>"
        )
        wsl_smoke_powershell = "\n".join(
            [
                f"$package = {_powershell_single_quoted(str(package_root))}",
                "Set-Location $package",
                wsl_smoke_command,
            ]
        )
        actions.append(
            {
                "title": "Run Windows-hosted WSL smoke chain",
                "category": "training",
                "launches_training": True,
                "reason": wsl_reason,
                "command": f"# from {package_root}\n{wsl_smoke_command}",
                "windows_powershell_command": wsl_smoke_powershell,
            }
        )
    if not stage_status["dpo"]["manifest_matches_requested_max_steps"]:
        rl_incomplete = (
            not stage_status["rl"]["manifest_matches_requested_max_steps"]
            or not report_status["rl_eval"]["exists"]
            or not sample_status["rl"]["manifest_exists"]
        )
        actions.append(
            {
                "title": (
                    "Resume full eval through DPO and RL"
                    if rl_incomplete
                    else "Resume full eval through DPO"
                ),
                "category": "training",
                "launches_training": True,
                "reason": (
                    "DPO has not reached the requested max_steps, and RL/final eval evidence is incomplete; "
                    "resume from the newest available DPO checkpoint if present, then continue the full chain."
                    if rl_incomplete
                    else "DPO has not reached the requested max_steps; resume from the newest available DPO checkpoint if present."
                ),
                "command": f"# from {package_root}\n{run_command}",
                "windows_powershell_command": _windows_wsl_command(package_root, run_command),
            }
        )
    elif (
        not stage_status["rl"]["manifest_matches_requested_max_steps"]
        or not report_status["rl_eval"]["exists"]
        or not sample_status["rl"]["manifest_exists"]
    ):
        actions.append(
            {
                "title": "Run RL and final eval",
                "category": "training",
                "launches_training": True,
                "reason": "DPO is complete but RL/final eval evidence is missing.",
                "command": f"# from {package_root}\n{run_command}",
                "windows_powershell_command": _windows_wsl_command(package_root, run_command),
            }
        )
    if (
        not diagnostics_status["all_required_plots_exist"]
        or not diagnostics_status["sources_current_for_run"]
        or not diagnostics_status["stages_current_for_requested_steps"]
    ):
        stale_stages = diagnostics_status.get("stale_or_missing_stages") or []
        diagnostics_reason = (
            "Diagnostic plots must be regenerated from this target outputs directory "
            "after target-stage evidence is current."
        )
        if stale_stages:
            diagnostics_reason += (
                " Current diagnostics are blocked by stale stages: "
                + ", ".join(f"`{stage}`" for stage in stale_stages)
                + "."
            )
        actions.append(
            {
                "title": "Regenerate target diagnostics",
                "category": "diagnostics",
                "launches_training": False,
                "reason": diagnostics_reason,
                "command": f"# from {package_root}\n{diagnostics_command}",
                "windows_powershell_command": _windows_wsl_command(
                    package_root, diagnostics_command
                ),
            }
        )
    phase6_action = _phase6_collection_next_action(package_root, human_usefulness_status)
    if phase6_action is not None:
        actions.append(phase6_action)
    actions.append(
        {
            "title": "Regenerate contract status",
            "category": "status",
            "launches_training": False,
            "reason": "Refresh JSON and Markdown status after the next full-eval attempt.",
            "command": f"# from {package_root}\n{status_command}",
            "windows_powershell_command": _windows_wsl_command(package_root, status_command),
        }
    )
    return actions


def _phase6_collection_next_action(
    package_root: Path,
    human_usefulness_status: dict[str, Any],
) -> dict[str, Any] | None:
    if not human_usefulness_status.get("checked"):
        return None
    coverage_reports = [
        coverage
        for coverage in human_usefulness_status.get("coverage_reports") or []
        if coverage.get("study")
    ]
    failed_coverage_reports = [
        coverage for coverage in coverage_reports if coverage.get("passed") is False
    ]
    if human_usefulness_status.get("passed") and not failed_coverage_reports:
        return None
    if not coverage_reports:
        return None
    coverage_parent = None
    first_coverage_path = coverage_reports[0].get("path")
    if first_coverage_path:
        coverage_parent = Path(first_coverage_path).resolve().parent
    collection_plan = _find_phase6_collection_plan(coverage_parent)
    if collection_plan is not None:
        command_sequence = _phase6_collection_plan_commands(collection_plan)
        if command_sequence:
            command = "\n".join(command_sequence)
            plan_path = _posix_relpath(Path(collection_plan["path"]), package_root)
            return {
                "title": "Run real Phase 6 collection and eval sequence",
                "category": "human_study",
                "launches_training": False,
                "reason": (
                    "A Phase 6 collection plan already exists; run its conduct-study, "
                    "coverage, eval, and suite commands to replace template answers "
                    "with real timed reviewer logs and refreshed usefulness gates."
                ),
                "command": f"# plan: {plan_path}\n{command}",
                "windows_powershell_command": command,
            }
    answers_dir = (coverage_parent or package_root) / "phase6_real_answers"
    out_path = answers_dir / "phase6_real_collection_plan.json"
    study_flags = []
    for coverage in coverage_reports:
        label = _phase6_collection_study_label(coverage)
        study_path = _posix_relpath(Path(coverage["study"]), package_root)
        study_flags.append(f"--study {label}={_shell_single_quoted(study_path)}")
    command = (
        "PYTHONPATH=src python -m semantic_mirror.cli review study-collection-plan "
        + " ".join(study_flags)
        + f" --answers-dir {_shell_single_quoted(_posix_relpath(answers_dir, package_root))}"
        + " --reviewer 'REPLACE_WITH_REVIEWER'"
        + f" --out {_shell_single_quoted(_posix_relpath(out_path, package_root))}"
    )
    return {
        "title": "Create real Phase 6 answer collection plan",
        "category": "human_study",
        "launches_training": False,
        "reason": (
            "Human usefulness evidence is checked but failing; create a real timed "
            "reviewer-answer plan before re-running study-status and eval human-study."
        ),
        "command": f"# from {package_root}\n{command}",
        "windows_powershell_command": _windows_wsl_command(package_root, command),
    }


def _find_phase6_collection_plan(coverage_parent: Path | None) -> dict[str, Any] | None:
    if coverage_parent is None:
        return None
    for name in (
        "phase6_real_collection_plan.json",
        "phase6_collection_manifest.json",
        "phase6_collection_manifest_from_package.json",
    ):
        path = coverage_parent / name
        report = _read_json_file(path)
        if (
            isinstance(report, dict)
            and report.get("mode") == "phase6_real_human_study_collection_plan"
        ):
            report["path"] = str(path)
            return report
    return None


def _phase6_collection_plan_commands(plan: dict[str, Any]) -> list[str]:
    studies = plan.get("studies")
    if not isinstance(studies, dict):
        return []
    commands = []
    for label, study in sorted(studies.items()):
        if not isinstance(study, dict):
            continue
        for key, header in (
            ("conduct_command", "conduct"),
            ("coverage_command", "coverage"),
            ("eval_command", "eval"),
        ):
            command = study.get(key)
            if isinstance(command, str) and command:
                commands.append(f"# {label} {header}\n{command}")
    suite_command = plan.get("suite_command")
    if isinstance(suite_command, str) and suite_command:
        commands.append(f"# suite\n{suite_command}")
    return commands


def _phase6_collection_study_label(coverage: dict[str, Any]) -> str:
    path_text = " ".join(
        str(coverage.get(key) or "").lower() for key in ("path", "study", "answers")
    )
    if "diff" in path_text:
        return "diff_mode"
    if "whole" in path_text:
        return "whole_repo"
    return "study"


def _windows_wsl_command(package_root: Path, bash_command: str, *, distro: str = "Ubuntu") -> str:
    return "\n".join(
        [
            f"$package = {_powershell_single_quoted(str(package_root))}",
            "$packageForWsl = $package -replace '\\\\', '/'",
            f'$packageWsl = (wsl.exe -d {distro} -- wslpath -a "$packageForWsl").Trim()',
            f'wsl.exe -d {distro} -- bash -lc "cd \'$packageWsl\' && {bash_command}"',
        ]
    )


def _contract_status_evidence_flags(
    *,
    package_root: Path,
    repo_hygiene_status: dict[str, Any],
    windows_readiness_status: dict[str, Any],
    package_source_status: dict[str, Any],
    human_usefulness_status: dict[str, Any],
) -> str:
    flags: list[str] = []
    if repo_hygiene_status.get("checked") and repo_hygiene_status.get("repo_root"):
        flags.extend(
            [
                "--repo-root",
                _shell_single_quoted(
                    _posix_relpath(Path(repo_hygiene_status["repo_root"]), package_root)
                ),
            ]
        )
    if windows_readiness_status.get("windows_audit_path"):
        flags.extend(
            [
                "--windows-audit",
                _shell_single_quoted(
                    _posix_relpath(
                        Path(windows_readiness_status["windows_audit_path"]),
                        package_root,
                    )
                ),
            ]
        )
    if windows_readiness_status.get("wsl_smoke_manifest_path"):
        flags.extend(
            [
                "--wsl-smoke-manifest",
                _shell_single_quoted(
                    _posix_relpath(
                        Path(windows_readiness_status["wsl_smoke_manifest_path"]),
                        package_root,
                    )
                ),
            ]
        )
    if package_source_status.get("path"):
        flags.extend(
            [
                "--package-source-freshness",
                _shell_single_quoted(
                    _posix_relpath(Path(package_source_status["path"]), package_root)
                ),
            ]
        )
    if human_usefulness_status.get("path"):
        flags.extend(
            [
                "--human-study-suite",
                _shell_single_quoted(
                    _posix_relpath(Path(human_usefulness_status["path"]), package_root)
                ),
            ]
        )
    for coverage in human_usefulness_status.get("coverage_reports") or []:
        if coverage.get("path"):
            flags.extend(
                [
                    "--human-study-coverage",
                    _shell_single_quoted(
                        _posix_relpath(Path(coverage["path"]), package_root)
                    ),
                ]
            )
    return " ".join(flags) + (" " if flags else "")


def _shell_single_quoted(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _powershell_single_quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _latest_checkpoint(stage_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    if not stage_dir.exists():
        return None
    for path in stage_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    return max(checkpoints, default=(0, None))[1]


def _posix_relpath(path: Path, start: Path) -> str:
    try:
        rel = Path(os.path.relpath(path, start))
    except ValueError:
        rel = path
    return rel.as_posix()


def _stage_contract_status(
    run: Path,
    stage: str,
    requested_steps: int | None,
) -> dict[str, Any]:
    stage_dir = run / f"semantic-mirror-{stage}"
    manifest_path = stage_dir / "training_stage_manifest.json"
    manifest = _read_json_file(manifest_path)
    return {
        "stage": stage,
        "stage_dir": str(stage_dir),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest is not None,
        "manifest_max_steps": manifest.get("max_steps") if manifest else None,
        "requested_max_steps": requested_steps,
        "manifest_matches_requested_max_steps": (
            manifest is not None
            and requested_steps is not None
            and manifest.get("max_steps") == requested_steps
        ),
        "resume_from_checkpoint": manifest.get("resume_from_checkpoint") if manifest else None,
        "output_dir": manifest.get("output_dir") if manifest else None,
    }


def _stage_recovery_contract_status(
    *,
    run: Path,
    stage_status: dict[str, dict[str, Any]],
    report_status: dict[str, dict[str, Any]],
    sample_status: dict[str, dict[str, Any]],
    resume_inspection_status: dict[str, Any],
) -> dict[str, Any]:
    report_keys = {
        "sft": ("sft_eval", "sft_vs_baseline"),
        "dpo": ("dpo_eval", "dpo_vs_sft"),
        "rl": ("rl_eval", "rl_vs_sft"),
    }
    recovery: dict[str, Any] = {}
    resume_decisions = resume_inspection_status.get("decisions", {})
    for stage in ("sft", "dpo", "rl"):
        latest_checkpoint = _latest_checkpoint(run / f"semantic-mirror-{stage}")
        decision = (
            resume_decisions.get(stage, {})
            if isinstance(resume_decisions, dict)
            else {}
        )
        inferred_action = _stage_recovery_action(
            stage=stage,
            status=stage_status[stage],
            latest_checkpoint=latest_checkpoint,
        )
        missing_current_artifacts = []
        if not stage_status[stage]["manifest_matches_requested_max_steps"]:
            missing_current_artifacts.append("stage_manifest")
        eval_key, compare_key = report_keys[stage]
        if not report_status[eval_key]["current_for_requested_stage"]:
            missing_current_artifacts.append(eval_key)
        if not report_status[compare_key]["current_for_requested_stage"]:
            missing_current_artifacts.append(compare_key)
        if not sample_status[stage]["complete_for_requested_stage"]:
            missing_current_artifacts.append(f"{stage}_sample_inspection")
        recovery[stage] = {
            "action": decision.get("action") or inferred_action,
            "reason": decision.get("reason"),
            "requested_max_steps": stage_status[stage]["requested_max_steps"],
            "manifest_max_steps": stage_status[stage]["manifest_max_steps"],
            "manifest_current": stage_status[stage][
                "manifest_matches_requested_max_steps"
            ],
            "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
            "latest_checkpoint_relative": (
                _posix_relpath(latest_checkpoint, run) if latest_checkpoint else None
            ),
            "resume_supported": stage in {"sft", "dpo"},
            "missing_current_artifacts": missing_current_artifacts,
        }
    return recovery


def _stage_recovery_action(
    *,
    stage: str,
    status: dict[str, Any],
    latest_checkpoint: Path | None,
) -> str:
    if status["manifest_matches_requested_max_steps"]:
        return "reuse"
    if stage in {"sft", "dpo"} and latest_checkpoint is not None:
        return "resume"
    return "run" if not status["manifest_exists"] else "rerun"


def _eval_summary_contract_status(
    eval_summary: dict[str, Any] | None,
    requested_steps: dict[str, int | None],
) -> dict[str, Any]:
    summary_requested = {}
    summary_manifest = {}
    if eval_summary:
        configured = eval_summary.get("eval_run_config", {}).get("requested_max_steps", {})
        if isinstance(configured, dict):
            summary_requested = {
                stage: configured.get(stage) for stage in ("sft", "dpo", "rl")
            }
        execution = eval_summary.get("stage_execution_summary", {})
        if isinstance(execution, dict):
            summary_manifest = {
                stage: execution.get(stage, {}).get("manifest_max_steps")
                for stage in ("sft", "dpo", "rl")
                if isinstance(execution.get(stage), dict)
            }
    expected = {stage: requested_steps.get(stage) for stage in ("sft", "dpo", "rl")}
    actual = {
        "requested_max_steps": summary_requested or None,
        "stage_manifest_max_steps": summary_manifest or None,
    }
    requested_matches = bool(summary_requested) and all(
        summary_requested.get(stage) == expected.get(stage)
        for stage in ("sft", "dpo", "rl")
        if expected.get(stage) is not None
    )
    manifest_matches = bool(summary_manifest) and all(
        summary_manifest.get(stage) == expected.get(stage)
        for stage in ("sft", "dpo", "rl")
        if expected.get(stage) is not None
    )
    return {
        "matches_requested_steps": requested_matches and manifest_matches,
        "requested_max_steps_match": requested_matches,
        "stage_manifest_max_steps_match": manifest_matches,
        "actual": actual,
        "expected": expected,
    }


def _json_report_status(path: Path, *, require_passed: bool = False) -> dict[str, Any]:
    report = _read_json_file(path)
    return {
        "path": str(path),
        "exists": report is not None,
        "passed": report.get("passed") if report else None,
        "mode": report.get("mode") if report else None,
        "required_passed": (report is not None and (not require_passed or report.get("passed") is True)),
    }


def _sample_contract_status(sample_dir: Path) -> dict[str, Any]:
    return {
        "sample_dir": str(sample_dir),
        "manifest_exists": (sample_dir / "sample_manifest.json").exists(),
        "raw_candidates_exist": (sample_dir / "raw_candidates.jsonl").exists(),
        "repaired_candidates_exist": (sample_dir / "repaired_candidates.jsonl").exists(),
        "raw_eval_exists": (sample_dir / "raw_eval.json").exists(),
        "repaired_eval_exists": (sample_dir / "repaired_eval.json").exists(),
        "inspection_markdown_exists": (sample_dir / "sample_inspection.md").exists(),
    }


def _diagnostics_contract_status(diagnostics: Path, *, run: Path) -> dict[str, Any]:
    required = [
        "training_summary.json",
        "training_summary.md",
        "sft_loss.png",
        "dpo_loss.png",
        "dpo_reward_accuracy.png",
        "rl_reward.png",
        "rl_parseability.png",
        "generation_lengths.png",
        "eval_metrics.png",
        "schema_coverage.png",
    ]
    existing = [name for name in required if (diagnostics / name).exists()]
    summary_path = diagnostics / "training_summary.json"
    summary = _read_json_file(summary_path)
    source_files = _diagnostic_source_files(summary)
    run_tokens = _path_match_tokens(run)
    foreign_sources = [
        source
        for source in source_files
        if not any(token and token in source.replace("\\", "/") for token in run_tokens)
    ]
    return {
        "summary_path": str(summary_path),
        "summary_exists": summary is not None,
        "required_plots": required,
        "existing_required_plots": existing,
        "missing_required_plots": [name for name in required if name not in existing],
        "all_required_plots_exist": len(existing) == len(required),
        "source_files": source_files,
        "foreign_source_files": foreign_sources,
        "sources_current_for_run": bool(source_files) and not foreign_sources,
    }


def _diagnostic_source_files(summary: dict[str, Any] | None) -> list[str]:
    if not isinstance(summary, dict):
        return []
    sources: set[str] = set()
    plots = summary.get("plots", {})
    if isinstance(plots, dict):
        for plot in plots.values():
            if not isinstance(plot, dict):
                continue
            source_files = plot.get("source_files", [])
            if isinstance(source_files, list):
                sources.update(str(source) for source in source_files)
    return sorted(sources)


def _diagnostic_gate_actual(diagnostics_status: dict[str, Any]) -> dict[str, Any]:
    foreign_sources = diagnostics_status["foreign_source_files"]
    return {
        "existing_required_plot_count": len(diagnostics_status["existing_required_plots"]),
        "missing_required_plots": diagnostics_status["missing_required_plots"],
        "sources_current_for_run": diagnostics_status["sources_current_for_run"],
        "stages_current_for_requested_steps": diagnostics_status[
            "stages_current_for_requested_steps"
        ],
        "stale_or_missing_stages": diagnostics_status["stale_or_missing_stages"],
        "foreign_source_file_count": len(foreign_sources),
        "foreign_source_file_examples": foreign_sources[:5],
    }


def _stage_evidence_summary(
    stage_status: dict[str, dict[str, Any]],
    report_status: dict[str, dict[str, Any]],
    sample_status: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    report_keys = {
        "sft": ("sft_eval", "sft_vs_baseline"),
        "dpo": ("dpo_eval", "dpo_vs_sft"),
        "rl": ("rl_eval", "rl_vs_sft"),
    }
    summary: dict[str, dict[str, Any]] = {}
    for stage, (eval_key, compare_key) in report_keys.items():
        summary[stage] = {
            "manifest_current": stage_status[stage][
                "manifest_matches_requested_max_steps"
            ],
            "manifest_max_steps": stage_status[stage]["manifest_max_steps"],
            "requested_max_steps": stage_status[stage]["requested_max_steps"],
            "eval_current": report_status[eval_key]["current_for_requested_stage"],
            "eval_exists": report_status[eval_key]["exists"],
            "compare_current": report_status[compare_key][
                "current_for_requested_stage"
            ],
            "compare_exists": report_status[compare_key]["exists"],
            "sample_current": sample_status[stage]["complete_for_requested_stage"],
            "sample_exists": sample_status[stage]["manifest_exists"],
        }
    return summary


def _contract_scorecard_contract_summary(
    scorecard: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "area": row.get("area"),
            "required": row.get("required"),
            "passed": row.get("passed"),
            "earned_reward": row.get("earned_reward"),
            "max_reward": row.get("max_reward"),
            "evidence": row.get("evidence"),
        }
        for row in scorecard
    ]


def _repo_hygiene_contract_summary(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "checked": status.get("checked", False),
        "passed": status.get("passed"),
        "summary": status.get("summary"),
        "tracked_change_count": len(status.get("tracked_changes", []) or []),
        "untracked_count": len(status.get("untracked", []) or []),
        "unexpected_ignored_count": len(status.get("ignored_unexpected", []) or []),
    }


def _stage_recovery_contract_summary(
    status: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for stage, recovery in status.items():
        if not isinstance(recovery, dict):
            continue
        summary[str(stage)] = {
            "action": recovery.get("action"),
            "requested_max_steps": recovery.get("requested_max_steps"),
            "manifest_max_steps": recovery.get("manifest_max_steps"),
            "latest_checkpoint_relative": recovery.get("latest_checkpoint_relative"),
            "missing_current_artifact_count": len(
                recovery.get("missing_current_artifacts", []) or []
            ),
        }
    return summary


def _windows_readiness_contract_summary(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "checked": status.get("checked", False),
        "passed": status.get("passed"),
        "native_passed": status.get("native_passed"),
        "native_blocked": status.get("native_blocked"),
        "native_failed_required_checks": status.get(
            "native_failed_required_checks", []
        ),
        "native_recommended_fallback": status.get("native_recommended_fallback"),
        "wsl_smoke_manifest_mode": status.get("wsl_smoke_manifest_mode"),
        "wsl_smoke_complete": status.get("wsl_smoke_complete"),
        "wsl_failed_checks": status.get("wsl_failed_checks", []),
        "wsl_missing_stage_manifest_count": len(
            status.get("wsl_missing_stage_manifests", []) or []
        ),
        "wsl_missing_sample_manifest_count": len(
            status.get("wsl_missing_sample_manifests", []) or []
        ),
        "wsl_diagnostics_exists": status.get("wsl_diagnostics_exists"),
        "summary": status.get("summary"),
    }


def _package_source_contract_summary(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "checked": status.get("checked", False),
        "passed": status.get("passed"),
        "git_commit_matches_repo": status.get("git_commit_matches_repo"),
        "compared_file_count": status.get("compared_file_count"),
        "mismatched_file_count": len(status.get("mismatched_files", []) or []),
        "all_package_specific_docs_present": status.get(
            "all_package_specific_docs_present"
        ),
        "missing_package_specific_doc_count": len(
            status.get("missing_package_specific_docs", []) or []
        ),
        "summary": status.get("summary"),
    }


def _package_command_manifest_contract_summary(
    status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checked": status.get("checked", False),
        "passed": status.get("passed"),
        "command_count": status.get("command_count"),
        "training_command_count": status.get("training_command_count"),
        "non_training_command_count": status.get("non_training_command_count"),
        "failed_checks": status.get("failed_checks", []),
    }


def _package_metadata_contract_summary(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "checked": status.get("checked", False),
        "passed": status.get("passed"),
        "requires_python": status.get("requires_python"),
        "expected_requires_python": status.get("expected_requires_python"),
        "excludes_python_3_14": status.get("excludes_python_3_14"),
        "summary": status.get("summary"),
    }


def _human_usefulness_contract_summary(status: dict[str, Any]) -> dict[str, Any]:
    collection_plan = status.get("collection_plan_status")
    if not isinstance(collection_plan, dict):
        collection_plan = {}
    collection_studies = collection_plan.get("studies")
    if not isinstance(collection_studies, dict):
        collection_studies = {}
    required_phase6_gates = status.get("required_phase6_gates")
    if not isinstance(required_phase6_gates, dict):
        required_phase6_gates = {}
    metrics = status.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    coverage_reports = status.get("coverage_reports")
    if not isinstance(coverage_reports, list):
        coverage_reports = []
    answer_record_count = _contract_summary_int(
        collection_plan.get("answer_record_count")
    )
    required_total_answer_records = _contract_summary_int(
        collection_plan.get("required_total_answer_records")
    )
    remaining_total_answer_records = max(
        required_total_answer_records - answer_record_count, 0
    )
    return {
        "checked": status.get("checked", False),
        "passed": status.get("passed"),
        "summary": status.get("summary"),
        "failed_phase6_gates": [
            gate for gate, passed in required_phase6_gates.items() if not passed
        ],
        "total_real_timed_answer_records": metrics.get(
            "total_real_timed_answer_records"
        ),
        "total_valid_answer_records": metrics.get("total_valid_answer_records"),
        "coverage_reports": [
            {
                "path": report.get("path"),
                "passed": report.get("passed"),
                "pending_task_count": report.get("pending_task_count"),
                "real_timed_answer_records": report.get("real_timed_answer_records"),
                "failed_gates": report.get("failed_gates", []),
            }
            for report in coverage_reports
            if isinstance(report, dict)
        ],
        "collection_plan": {
            "checked": collection_plan.get("checked", False),
            "passed": collection_plan.get("passed"),
            "answer_record_count": answer_record_count,
            "required_total_answer_records": required_total_answer_records,
            "remaining_total_answer_records": remaining_total_answer_records,
            "complete": remaining_total_answer_records == 0,
            "missing_answer_targets": collection_plan.get("missing_answer_targets", []),
            "studies": {
                str(label): _human_usefulness_study_contract_summary(study)
                for label, study in collection_studies.items()
                if isinstance(study, dict)
            },
        },
    }


def _human_usefulness_study_contract_summary(
    study: dict[str, Any],
) -> dict[str, Any]:
    answer_records = _contract_summary_int(study.get("answer_records"))
    required_answer_records = _contract_summary_int(study.get("required_answer_records"))
    remaining_answer_records = max(required_answer_records - answer_records, 0)
    return {
        "answer_records": answer_records,
        "required_answer_records": required_answer_records,
        "remaining_answer_records": remaining_answer_records,
        "complete": remaining_answer_records == 0,
        "answer_target_exists": study.get("answer_target_exists"),
        "answer_target": study.get("answer_target"),
        "coverage_report": study.get("coverage_report"),
        "eval_report": study.get("eval_report"),
    }


def _contract_summary_int(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _remaining_by_area(remaining_items: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in remaining_items:
        gate = str(item["gate"])
        area = _remaining_area_for_gate(gate)
        grouped.setdefault(area, []).append(gate)
    return dict(sorted(grouped.items()))


def _remaining_area_for_gate(gate: str) -> str:
    if gate.startswith("sft_"):
        return "sft"
    if gate.startswith("dpo_"):
        return "dpo"
    if gate.startswith("rl_"):
        return "rl"
    if gate.startswith("diagnostic_"):
        return "diagnostics"
    if gate.startswith("windows_"):
        return "windows_unsloth_readiness"
    if gate.startswith("package_"):
        return "package"
    if gate.startswith("training_eval_summary") or gate == "all_final_eval_gates_passed":
        return "final_summary"
    return "other"


def _remaining_recovery_plan(
    remaining_items: list[dict[str, Any]],
    *,
    stage_recovery_status: dict[str, Any],
    diagnostics_status: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _remaining_gate_recovery_item(
            item,
            stage_recovery_status=stage_recovery_status,
            diagnostics_status=diagnostics_status,
        )
        for item in remaining_items
    ]


def _remaining_gate_recovery_item(
    item: dict[str, Any],
    *,
    stage_recovery_status: dict[str, Any],
    diagnostics_status: dict[str, Any],
) -> dict[str, Any]:
    gate = str(item["gate"])
    area = _remaining_area_for_gate(gate)
    stage = area if area in {"sft", "dpo", "rl"} else None
    stage_recovery = stage_recovery_status.get(stage, {}) if stage else {}
    artifacts = [str(item.get("evidence"))] if item.get("evidence") else []
    requires_training = False
    required_action = "inspect"
    blocked_by: list[str] = []
    if stage:
        required_action = str(stage_recovery.get("action") or "run")
        requires_training = required_action in {"resume", "run", "rerun"}
        current_evidence = item.get("actual")
        if (
            isinstance(current_evidence, dict)
            and current_evidence.get("stage_current_for_requested_steps") is False
        ):
            blocked_by = [stage]
            requires_training = True
        if gate.endswith("_sample_inspection_complete"):
            artifacts = [str(item.get("evidence")), f"{item.get('evidence')}/sample_manifest.json"]
            required_action = "generate_sample_inspection_after_stage"
        elif "_eval_" in gate or "_vs_" in gate:
            required_action = "generate_eval_report_after_stage"
    elif area == "diagnostics":
        required_action = "regenerate_diagnostics"
        blocked_by = list(diagnostics_status.get("stale_or_missing_stages") or [])
        requires_training = bool(blocked_by)
    elif area == "final_summary":
        required_action = "rerun_full_eval_summary"
        blocked_by = [
            stage_name
            for stage_name, recovery in stage_recovery_status.items()
            if recovery.get("missing_current_artifacts")
        ]
        requires_training = bool(blocked_by)
    elif area == "package":
        required_action = _package_recovery_action(gate)
    elif area == "windows_unsloth_readiness":
        required_action = "run_wsl_smoke_chain"
        requires_training = True
    return {
        "gate": gate,
        "area": area,
        "stage": stage,
        "required_action": required_action,
        "requires_training": requires_training,
        "blocked_by_stages": blocked_by,
        "artifacts": artifacts,
        "current_evidence": item.get("actual"),
        "expected_evidence": item.get("expected"),
    }


def _package_recovery_action(gate: str) -> str:
    if gate == "package_source_freshness_valid_when_checked":
        return "regenerate_package_source_freshness"
    if gate == "package_command_manifest_valid_when_checked":
        return "regenerate_package_command_manifest"
    if gate == "package_python_metadata_valid_when_checked":
        return "fix_package_python_metadata"
    return "inspect_package_evidence"


def _repo_hygiene_contract_status(repo_root: Path | str | None) -> dict[str, Any]:
    if repo_root is None:
        return {
            "checked": False,
            "passed": None,
            "summary": "Repo hygiene not checked; pass --repo-root to record git status evidence.",
        }
    root = Path(repo_root).resolve()
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--branch", "--ignored"],
            cwd=root,
            check=False,
            capture_output=True,
            encoding="utf-8",
        )
    except OSError as exc:
        return {
            "checked": True,
            "passed": False,
            "repo_root": str(root),
            "summary": f"git status failed: {exc}",
            "error": str(exc),
        }
    lines = [line.rstrip() for line in result.stdout.splitlines() if line.strip()]
    branch = next((line for line in lines if line.startswith("## ")), None)
    entries = [line for line in lines if not line.startswith("## ")]
    tracked_changes = [
        line for line in entries if not line.startswith("?? ") and not line.startswith("!! ")
    ]
    untracked = [line[3:] for line in entries if line.startswith("?? ")]
    ignored = [line[3:] for line in entries if line.startswith("!! ")]
    allowed_ignored = [
        path for path in ignored if _repo_hygiene_ignored_path_allowed(path)
    ]
    unexpected_ignored = [
        path for path in ignored if not _repo_hygiene_ignored_path_allowed(path)
    ]
    passed = (
        result.returncode == 0
        and not tracked_changes
        and not untracked
        and not unexpected_ignored
    )
    summary = (
        "git status clean except allowed ignored local-only artifacts"
        if passed
        else (
            f"git status has {len(tracked_changes)} tracked change(s), "
            f"{len(untracked)} untracked path(s), and "
            f"{len(unexpected_ignored)} unexpected ignored path(s)"
        )
    )
    return {
        "checked": True,
        "passed": passed,
        "repo_root": str(root),
        "git_returncode": result.returncode,
        "branch": branch,
        "summary": summary,
        "tracked_changes": tracked_changes,
        "untracked": untracked,
        "ignored_allowed": allowed_ignored,
        "ignored_unexpected": unexpected_ignored,
        "stderr": result.stderr.strip(),
    }


def _repo_hygiene_ignored_path_allowed(path: str) -> bool:
    normalized = path.replace("\\", "/").rstrip("/")
    allowed_exact = {
        ".env",
        ".pytest_cache",
        ".ruff_cache",
        ".semantic-mirror",
        ".venv",
        "SEMANTIC_MIRROR_GOAL_CONTRACT.md",
        "SEMANTIC_MIRROR_PLAN.md",
        "outputs",
    }
    allowed_suffixes = (
        "/__pycache__",
        "/.pytest_cache",
        "/.ruff_cache",
    )
    return normalized in allowed_exact or normalized.endswith(allowed_suffixes)


def _windows_readiness_contract_status(
    *,
    windows_audit_path: Path | str | None,
    wsl_smoke_manifest_path: Path | str | None,
) -> dict[str, Any]:
    audit_path = Path(windows_audit_path).resolve() if windows_audit_path else None
    smoke_path = (
        Path(wsl_smoke_manifest_path).resolve() if wsl_smoke_manifest_path else None
    )
    audit = _read_json_file(audit_path) if audit_path is not None else None
    smoke = _read_json_file(smoke_path) if smoke_path is not None else None
    native_checked = isinstance(audit, dict)
    native_passed = bool(audit and audit.get("passed") and audit.get("ready_to_launch"))
    native_blocked = bool(
        audit
        and not native_passed
        and isinstance(audit.get("blocker"), dict)
        and audit["blocker"].get("blocked")
    )
    wsl_checked = isinstance(smoke, dict)
    stages = smoke.get("stages", {}) if isinstance(smoke, dict) else {}
    samples = smoke.get("samples", {}) if isinstance(smoke, dict) else {}
    wsl_stage_manifests = {
        stage: bool((stages.get(stage) or {}).get("stage_manifest_exists"))
        for stage in ("sft", "dpo", "rl")
    }
    wsl_sample_manifests = {
        stage: bool((samples.get(stage) or {}).get("sample_manifest_exists"))
        for stage in ("sft", "dpo", "rl")
    }
    wsl_complete = (
        wsl_checked
        and smoke.get("mode") == "smoke_chain"
        and all(wsl_stage_manifests.values())
        and all(wsl_sample_manifests.values())
        and bool(smoke.get("diagnostics_exists"))
    )
    wsl_failed_checks = []
    if wsl_checked and smoke.get("mode") != "smoke_chain":
        wsl_failed_checks.append("smoke_chain_manifest_mode")
    missing_stage_manifests = [
        stage for stage, exists in wsl_stage_manifests.items() if not exists
    ]
    missing_sample_manifests = [
        stage for stage, exists in wsl_sample_manifests.items() if not exists
    ]
    if wsl_checked and missing_stage_manifests:
        wsl_failed_checks.append("stage_manifests")
    if wsl_checked and missing_sample_manifests:
        wsl_failed_checks.append("sample_manifests")
    if wsl_checked and not bool(smoke.get("diagnostics_exists")):
        wsl_failed_checks.append("diagnostics")
    passed = native_passed or (native_blocked and wsl_complete)
    if native_passed:
        summary = "Windows-native audit passed and is ready to launch."
    elif native_blocked and wsl_complete:
        summary = (
            "Windows-native audit is blocked, and Windows-hosted WSL smoke-chain "
            "evidence is complete."
        )
    elif not native_checked and not wsl_checked:
        summary = (
            "Windows readiness not checked; pass --windows-audit and "
            "--wsl-smoke-manifest to record evidence."
        )
    else:
        summary = "Windows readiness evidence is incomplete or failing."
    return {
        "checked": native_checked or wsl_checked,
        "passed": passed if native_checked or wsl_checked else None,
        "summary": summary,
        "windows_audit_path": str(audit_path) if audit_path is not None else None,
        "wsl_smoke_manifest_path": str(smoke_path) if smoke_path is not None else None,
        "native_checked": native_checked,
        "native_passed": native_passed,
        "native_blocked": native_blocked,
        "native_failed_required_checks": (
            audit.get("blocker", {}).get("failed_required_checks", [])
            if isinstance(audit, dict)
            else []
        ),
        "native_recommended_fallback": (
            audit.get("blocker", {}).get("recommended_fallback")
            if isinstance(audit, dict)
            else None
        ),
        "wsl_checked": wsl_checked,
        "wsl_smoke_manifest_mode": smoke.get("mode") if isinstance(smoke, dict) else None,
        "wsl_smoke_complete": wsl_complete,
        "wsl_failed_checks": wsl_failed_checks,
        "wsl_stage_manifests": wsl_stage_manifests,
        "wsl_missing_stage_manifests": missing_stage_manifests,
        "wsl_sample_manifests": wsl_sample_manifests,
        "wsl_missing_sample_manifests": missing_sample_manifests,
        "wsl_diagnostics_exists": bool(smoke.get("diagnostics_exists"))
        if isinstance(smoke, dict)
        else False,
        "wsl_smoke_out": smoke.get("smoke_out") if isinstance(smoke, dict) else None,
    }


def _package_source_freshness_contract_status(
    freshness_path: Path | str | None,
    *,
    repo_hygiene_status: dict[str, Any],
) -> dict[str, Any]:
    if freshness_path is None:
        return {
            "checked": False,
            "passed": None,
            "summary": (
                "Package source freshness not checked; pass "
                "--package-source-freshness with a source_freshness JSON report."
            ),
        }
    path = Path(freshness_path).resolve()
    report = _read_json_file(path)
    if not isinstance(report, dict):
        return {
            "checked": True,
            "passed": False,
            "path": str(path),
            "summary": "Package source freshness report is missing or not a JSON object.",
        }
    mode_ok = report.get("mode") == "semantic_mirror_package_source_freshness"
    all_match = bool(report.get("all_compared_files_match"))
    compared_count = report.get("compared_file_count")
    compared_count_ok = isinstance(compared_count, int) and compared_count > 0
    report_commit = report.get("git_commit")
    current_commit = _repo_current_commit(repo_hygiene_status.get("repo_root"))
    commit_matches = (
        None
        if current_commit is None or not isinstance(report_commit, str)
        else report_commit == current_commit
    )
    package_docs = _package_specific_docs_status(
        report.get("package_specific_docs"),
        package_root=report.get("package_root"),
    )
    passed = (
        mode_ok
        and all_match
        and compared_count_ok
        and package_docs["all_present"]
        and commit_matches is not False
    )
    if passed:
        summary = (
            "Package runtime source freshness and package-specific docs are proven."
        )
    elif commit_matches is False:
        summary = "Package source freshness report is for a different repo commit."
    elif not package_docs["all_present"]:
        summary = "Package source freshness evidence is missing package-specific docs."
    else:
        summary = "Package source freshness evidence is incomplete or failing."
    return {
        "checked": True,
        "passed": passed,
        "path": str(path),
        "summary": summary,
        "mode": report.get("mode"),
        "repo_root": report.get("repo_root"),
        "package_root": report.get("package_root"),
        "git_commit": report_commit,
        "current_repo_commit": current_commit,
        "git_commit_matches_repo": commit_matches,
        "compared_scope": report.get("compared_scope"),
        "compared_file_count": compared_count,
        "all_compared_files_match": all_match,
        "mismatched_files": [
            row.get("relative_path")
            for row in report.get("comparisons", [])
            if isinstance(row, dict) and not row.get("match")
        ],
        "all_package_specific_docs_present": package_docs["all_present"],
        "missing_package_specific_docs": package_docs["missing"],
        "package_specific_docs": package_docs["docs"],
    }


def _package_specific_docs_status(
    docs: object,
    *,
    package_root: object,
) -> dict[str, Any]:
    if not isinstance(docs, list):
        return {"all_present": False, "missing": ["package_specific_docs"], "docs": []}
    package = Path(str(package_root)) if package_root else None
    checked_docs = []
    missing = []
    for doc in docs:
        if not isinstance(doc, dict):
            missing.append("malformed_doc_entry")
            continue
        relative_path = doc.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            missing.append("missing_relative_path")
            checked_docs.append(dict(doc))
            continue
        package_file = package / relative_path if package is not None else None
        package_exists = doc.get("package_exists")
        if not isinstance(package_exists, bool):
            package_exists = package_file.exists() if package_file is not None else False
        package_sha256 = doc.get("package_sha256")
        if (
            package_exists
            and not isinstance(package_sha256, str)
            and package_file is not None
        ):
            package_sha256 = _sha256_file(package_file)
        checked_doc = {
            **doc,
            "package_exists": package_exists,
            "package_sha256": package_sha256 if isinstance(package_sha256, str) else None,
        }
        checked_docs.append(checked_doc)
        if not package_exists:
            missing.append(relative_path)
    return {
        "all_present": bool(checked_docs) and not missing,
        "missing": missing,
        "docs": checked_docs,
    }


def _package_command_manifest_contract_status(
    package_source_status: dict[str, Any],
) -> dict[str, Any]:
    package_root = package_source_status.get("package_root")
    if not package_root:
        return {
            "checked": False,
            "passed": None,
            "summary": (
                "Package command manifest not checked; package source freshness "
                "evidence did not include a package_root."
            ),
        }
    path = Path(str(package_root)) / "launch" / "commands_manifest.json"
    report = _read_json_file(path)
    if not isinstance(report, dict):
        return {
            "checked": True,
            "passed": False,
            "path": str(path),
            "summary": "Package command manifest is missing or not a JSON object.",
        }
    commands = report.get("commands")
    if not isinstance(commands, dict):
        return {
            "checked": True,
            "passed": False,
            "path": str(path),
            "summary": "Package command manifest does not contain a commands object.",
            "schema_version": report.get("schema_version"),
            "command_count": 0,
            "failed_checks": ["commands_object"],
        }
    expected_training = {
        "wsl_smoke_chain",
        "sft",
        "dpo",
        "rl",
        "full_training_eval",
        "smoke_chain",
    }
    expected_non_training = {
        "inspect_full_training_eval_resume",
        "inspect_resume",
        "contract_status",
        "source_freshness",
        "report",
        "validate",
        "audit",
        "install",
        "bootstrap_linux_cuda",
        "bootstrap_wsl_ubuntu",
        "generate_candidates",
        "score_candidates",
        "eval_candidates",
        "inspect_samples",
        "compare_sft",
        "compare_sft_raw",
        "compare_dpo",
        "compare_dpo_raw",
        "compare_rl",
        "compare_rl_raw",
    }
    training_commands = {
        name
        for name, command in commands.items()
        if isinstance(command, dict) and command.get("launches_training") is True
    }
    non_training_commands = {
        name
        for name, command in commands.items()
        if isinstance(command, dict) and command.get("launches_training") is False
    }
    missing_training = sorted(expected_training - training_commands)
    unexpected_training = sorted(training_commands - expected_training)
    missing_non_training = sorted(expected_non_training - non_training_commands)
    malformed = sorted(
        name for name, command in commands.items() if not isinstance(command, dict)
    )
    missing_command_text = sorted(
        name
        for name, command in commands.items()
        if isinstance(command, dict) and not isinstance(command.get("command"), str)
    )
    missing_category = sorted(
        name
        for name, command in commands.items()
        if isinstance(command, dict) and not isinstance(command.get("category"), str)
    )
    failed_checks = []
    if report.get("schema_version") != 1:
        failed_checks.append("schema_version")
    if missing_training:
        failed_checks.append("missing_training_launch_flags")
    if unexpected_training:
        failed_checks.append("unexpected_training_launch_flags")
    if missing_non_training:
        failed_checks.append("missing_non_training_launch_flags")
    if malformed:
        failed_checks.append("malformed_command_entries")
    if missing_command_text:
        failed_checks.append("missing_command_text")
    if missing_category:
        failed_checks.append("missing_category")
    passed = not failed_checks
    return {
        "checked": True,
        "passed": passed,
        "path": str(path),
        "summary": (
            "Package command manifest classifies training and non-training commands."
            if passed
            else "Package command manifest is missing required safety metadata."
        ),
        "schema_version": report.get("schema_version"),
        "command_count": len(commands),
        "training_command_count": len(training_commands),
        "non_training_command_count": len(non_training_commands),
        "training_commands": sorted(training_commands),
        "non_training_commands": sorted(non_training_commands),
        "missing_training_commands": missing_training,
        "unexpected_training_commands": unexpected_training,
        "missing_non_training_commands": missing_non_training,
        "malformed_commands": malformed,
        "missing_command_text": missing_command_text,
        "missing_category": missing_category,
        "failed_checks": failed_checks,
    }


def _package_metadata_contract_status(
    package_source_status: dict[str, Any],
) -> dict[str, Any]:
    package_root = package_source_status.get("package_root")
    if not package_root:
        return {
            "checked": False,
            "passed": None,
            "summary": (
                "Package Python metadata not checked; package source freshness "
                "evidence did not include a package_root."
            ),
        }
    path = Path(str(package_root)) / "pyproject.toml"
    if not path.exists():
        return {
            "checked": True,
            "passed": False,
            "path": str(path),
            "summary": "Package pyproject.toml is missing.",
            "expected_requires_python": UNSLOTH_PYTHON_RANGE,
            "requires_python": None,
        }
    requires_python = _pyproject_requires_python(path)
    passed = requires_python == UNSLOTH_PYTHON_RANGE
    return {
        "checked": True,
        "passed": passed,
        "path": str(path),
        "summary": (
            "Package Python metadata matches the Unsloth training runtime range."
            if passed
            else "Package Python metadata does not match the Unsloth training runtime range."
        ),
        "expected_requires_python": UNSLOTH_PYTHON_RANGE,
        "requires_python": requires_python,
        "excludes_python_3_14": requires_python == UNSLOTH_PYTHON_RANGE,
    }


def _pyproject_requires_python(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped.startswith("requires-python"):
            continue
        _, _, value = stripped.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            return value[1:-1]
        return value or None
    return None


def _repo_current_commit(repo_root: Any) -> str | None:
    if not repo_root:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(str(repo_root)),
            check=False,
            capture_output=True,
            encoding="utf-8",
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _repo_git_status_short(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--branch", "--ignored"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            encoding="utf-8",
        )
    except OSError as exc:
        return f"git status failed: {exc}"
    return result.stdout.rstrip()


def _runtime_source_files(source_root: Path) -> list[str]:
    if not source_root.exists():
        return []
    files = [
        path.relative_to(source_root.parent.parent).as_posix()
        for path in source_root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    return sorted(files)


def _source_freshness_comparison(
    *,
    relative_path: str,
    repo_root: Path,
    package_root: Path,
) -> dict[str, Any]:
    repo_file = repo_root / relative_path
    package_file = package_root / relative_path
    repo_exists = repo_file.exists()
    package_exists = package_file.exists()
    repo_hash = _sha256_file(repo_file) if repo_exists else None
    package_hash = _sha256_file(package_file) if package_exists else None
    return {
        "relative_path": relative_path,
        "repo_exists": repo_exists,
        "package_exists": package_exists,
        "repo_sha256": repo_hash,
        "package_sha256": package_hash,
        "match": repo_exists and package_exists and repo_hash == package_hash,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_freshness_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Source Freshness Evidence",
        "",
        f"Generated: {report['generated_at']}",
        f"Repo commit: {report.get('git_commit')}",
        f"Repo root: {report['repo_root']}",
        f"Package root: {report['package_root']}",
        f"Compared scope: {report['compared_scope']}",
        f"Compared files: {report['compared_file_count']}",
        f"All compared files match: {report['all_compared_files_match']}",
        "",
        "## Package-Specific Docs",
        "",
        "| File | Exists | Package SHA256 | Reason |",
        "| --- | --- | --- | --- |",
    ]
    for doc in report["package_specific_docs"]:
        lines.append(
            f"| {doc['relative_path']} | {doc.get('package_exists')} | "
            f"{doc.get('package_sha256')} | {doc['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Compared Files",
            "",
            "| File | Match | Repo SHA256 | Package SHA256 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in report["comparisons"]:
        lines.append(
            f"| {row['relative_path']} | {row['match']} | "
            f"{row['repo_sha256']} | {row['package_sha256']} |"
        )
    mismatches = [row["relative_path"] for row in report["comparisons"] if not row["match"]]
    if mismatches:
        lines.extend(["", "## Mismatches"])
        lines.extend(f"- {path}" for path in mismatches)
    lines.append("")
    return "\n".join(lines)


def _human_usefulness_contract_status(
    suite_path: Path | str | None,
    *,
    coverage_paths: Iterable[Path | str] | None = None,
) -> dict[str, Any]:
    coverage_reports = _human_study_coverage_contract_status(coverage_paths)
    collection_plan_status = _phase6_collection_plan_contract_status(coverage_reports)
    if suite_path is None:
        return {
            "checked": bool(coverage_reports),
            "passed": None,
            "coverage_reports": coverage_reports,
            "collection_plan_status": collection_plan_status,
            "summary": "Human usefulness not checked; pass --human-study-suite with an eval human-study-suite report.",
        }
    path = Path(suite_path).resolve()
    report = _read_json_file(path)
    if not isinstance(report, dict):
        return {
            "checked": True,
            "passed": False,
            "path": str(path),
            "coverage_reports": coverage_reports,
            "collection_plan_status": collection_plan_status,
            "summary": "Human-study suite report is missing or not a JSON object.",
        }
    phase6 = report.get("phase6_gate_summary", {})
    required_phase6_gates = (
        "required_task_sets_present",
        "all_reports_passed",
        "real_timed_reviewer_logs",
        "reviewer_identity_present",
        "answer_text_present",
        "mirror_accuracy_not_lower_than_source",
        "mirror_median_faster_than_source",
        "changed_behavior_accuracy_at_or_above_threshold",
        "visibility_items_acknowledged",
    )
    gate_status = {
        gate: bool(phase6.get(gate)) if isinstance(phase6, dict) else False
        for gate in required_phase6_gates
    }
    passed = (
        report.get("mode") == "human_usefulness_study_suite_summary"
        and bool(report.get("passed"))
        and all(gate_status.values())
    )
    summary = (
        "Phase 6 human-study suite passed with real timed reviewer evidence."
        if passed
        else "Phase 6 human-study suite evidence is incomplete or failing."
    )
    return {
        "checked": True,
        "passed": passed,
        "path": str(path),
        "summary": summary,
        "mode": report.get("mode"),
        "report_passed": bool(report.get("passed")),
        "required_phase6_gates": gate_status,
        "metrics": report.get("metrics", {}),
        "coverage_reports": coverage_reports,
        "collection_plan_status": collection_plan_status,
    }


def _human_study_coverage_contract_status(
    coverage_paths: Iterable[Path | str] | None,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for coverage_path in coverage_paths or []:
        path = Path(coverage_path).resolve()
        report = _read_json_file(path)
        if not isinstance(report, dict):
            reports.append(
                {
                    "path": str(path),
                    "checked": True,
                    "passed": False,
                    "summary": "Coverage report is missing or not a JSON object.",
                }
            )
            continue
        counts = report.get("counts", {})
        gates = report.get("gates", [])
        failed_gates = [
            gate.get("name")
            for gate in gates
            if isinstance(gate, dict) and not gate.get("passed")
        ]
        reports.append(
            {
                "path": str(path),
                "checked": True,
                "passed": (
                    report.get("mode") == "human_usefulness_study_answer_coverage"
                    and bool(report.get("passed"))
                ),
                "mode": report.get("mode"),
                "study": report.get("study"),
                "answers": report.get("answers"),
                "counts": counts if isinstance(counts, dict) else {},
                "failed_gates": failed_gates,
                "pending_task_count": (
                    counts.get("pending_task_records") if isinstance(counts, dict) else None
                ),
                "real_timed_answer_records": (
                    counts.get("real_timed_answer_records")
                    if isinstance(counts, dict)
                    else None
                ),
            }
        )
    return reports


def _phase6_collection_plan_contract_status(
    coverage_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    coverage_parent = None
    for coverage in coverage_reports:
        if coverage.get("path"):
            coverage_parent = Path(coverage["path"]).resolve().parent
            break
    plan = _find_phase6_collection_plan(coverage_parent)
    if plan is None:
        return {
            "checked": bool(coverage_reports),
            "passed": None if coverage_reports else False,
            "summary": (
                "No Phase 6 collection plan found next to coverage reports."
                if coverage_reports
                else "No Phase 6 coverage reports supplied."
            ),
        }
    studies = plan.get("studies")
    if not isinstance(studies, dict):
        return {
            "checked": True,
            "passed": False,
            "path": plan.get("path"),
            "summary": "Phase 6 collection plan has no studies object.",
        }
    study_statuses = {}
    total_required = 0
    total_answer_records = 0
    missing_answer_targets = []
    for label, study in sorted(studies.items()):
        if not isinstance(study, dict):
            study_statuses[label] = {"valid": False}
            continue
        answer_target = study.get("answer_target")
        answer_path = Path(answer_target) if isinstance(answer_target, str) else None
        records = _read_jsonl(answer_path) if answer_path is not None else []
        answer_records = len(records)
        required_records = int(study.get("answer_template_records") or 0)
        exists = bool(answer_path and answer_path.exists())
        if not exists:
            missing_answer_targets.append(label)
        total_required += required_records
        total_answer_records += answer_records
        study_statuses[label] = {
            "answer_target": str(answer_path) if answer_path is not None else None,
            "answer_target_exists": exists,
            "answer_records": answer_records,
            "required_answer_records": required_records,
            "coverage_report": study.get("coverage_report"),
            "eval_report": study.get("eval_report"),
        }
    plan_required = int(plan.get("required_total_answer_records") or total_required)
    passed = total_answer_records >= plan_required and not missing_answer_targets
    return {
        "checked": True,
        "passed": passed,
        "path": plan.get("path"),
        "summary": (
            "Phase 6 real answer targets have the required record count."
            if passed
            else "Phase 6 real answer targets are missing or incomplete."
        ),
        "required_total_answer_records": plan_required,
        "answer_record_count": total_answer_records,
        "missing_answer_targets": missing_answer_targets,
        "studies": study_statuses,
    }


def _contract_scorecard(report: dict[str, Any]) -> list[dict[str, Any]]:
    stage_summary = report["stage_evidence_summary"]
    all_stage_manifests = all(
        stage_summary[stage]["manifest_current"] for stage in ("sft", "dpo", "rl")
    )
    all_samples = all(
        stage_summary[stage]["sample_current"] for stage in ("sft", "dpo", "rl")
    )
    all_eval = all(
        stage_summary[stage]["eval_current"] and stage_summary[stage]["compare_current"]
        for stage in ("sft", "dpo", "rl")
    )
    diagnostics_current = (
        report["diagnostics_status"]["all_required_plots_exist"]
        and report["diagnostics_status"]["sources_current_for_run"]
        and report["diagnostics_status"]["stages_current_for_requested_steps"]
    )
    final_summary_current = (
        report["training_eval_summary_status"]["matches_requested_steps"]
        and _gate_passed(report, "all_final_eval_gates_passed")
    )
    repo_hygiene = report.get("repo_hygiene_status") or {}
    repo_hygiene_passed = (
        repo_hygiene.get("passed") if repo_hygiene.get("checked") else None
    )
    repo_hygiene_evidence = (
        repo_hygiene.get("summary")
        if repo_hygiene.get("checked")
        else "Not proven by a run outputs directory; use pytest, ruff, diff check, and git status."
    )
    windows_readiness = report.get("windows_readiness_status") or {}
    windows_readiness_passed = (
        windows_readiness.get("passed")
        if windows_readiness.get("checked")
        else None
    )
    windows_readiness_evidence = (
        windows_readiness.get("summary")
        if windows_readiness.get("checked")
        else "Not proven by contract-status; use runtime audit and WSL smoke-chain evidence."
    )
    human_usefulness = report.get("human_usefulness_status") or {}
    human_usefulness_passed = (
        human_usefulness.get("passed") if human_usefulness.get("checked") else None
    )
    human_usefulness_evidence = (
        human_usefulness.get("summary")
        if human_usefulness.get("checked")
        else "Not proven by a full-eval outputs directory; requires timed human-study reports."
    )
    scorecard = [
        {
            "area": "repo_hygiene",
            "required": True,
            "max_reward": 25,
            "passed": repo_hygiene_passed,
            "evidence": repo_hygiene_evidence,
        },
        {
            "area": "windows_unsloth_readiness",
            "required": True,
            "max_reward": 65,
            "passed": windows_readiness_passed,
            "evidence": windows_readiness_evidence,
        },
        {
            "area": "sft_dpo_rl_implementation",
            "required": True,
            "max_reward": 85,
            "passed": all_stage_manifests,
            "evidence": "Requested-step stage manifests for SFT, DPO, and RL.",
        },
        {
            "area": "diagnostic_plots",
            "required": True,
            "max_reward": 60,
            "passed": diagnostics_current,
            "evidence": "Required diagnostic files plus current-run source provenance and current stage manifests.",
        },
        {
            "area": "post_training_samples",
            "required": True,
            "max_reward": 85,
            "passed": all_samples,
            "evidence": "Per-stage raw/repaired candidates and sample inspection for current requested stages.",
        },
        {
            "area": "real_training_eval_gates",
            "required": True,
            "max_reward": 100,
            "passed": all_stage_manifests and all_eval and final_summary_current,
            "evidence": "Current SFT/DPO/RL manifests, eval reports, model-compare reports, and final summary gates.",
        },
        {
            "area": "human_usefulness",
            "required": False,
            "max_reward": 70,
            "passed": human_usefulness_passed,
            "evidence": human_usefulness_evidence,
        },
    ]
    for row in scorecard:
        row["earned_reward"] = row["max_reward"] if row["passed"] is True else 0
    return scorecard


def _contract_reward_summary(scorecard: list[dict[str, Any]]) -> dict[str, Any]:
    required_rows = [row for row in scorecard if row["required"]]
    optional_rows = [row for row in scorecard if not row["required"]]
    required_reward_earned = sum(row["earned_reward"] for row in required_rows)
    required_reward_possible = sum(row["max_reward"] for row in required_rows)
    optional_reward_earned = sum(row["earned_reward"] for row in optional_rows)
    optional_reward_possible = sum(row["max_reward"] for row in optional_rows)
    failed_required_areas = [
        row["area"] for row in required_rows if row["passed"] is not True
    ]
    minimum_acceptable_required_reward = 320
    required_reward_threshold_met = (
        required_reward_earned >= minimum_acceptable_required_reward
    )
    zero_failed_required_areas = not failed_required_areas
    return {
        "required_reward_earned": required_reward_earned,
        "required_reward_possible": required_reward_possible,
        "optional_reward_earned": optional_reward_earned,
        "optional_reward_possible": optional_reward_possible,
        "minimum_acceptable_required_reward": minimum_acceptable_required_reward,
        "required_reward_threshold_met": required_reward_threshold_met,
        "zero_failed_required_areas": zero_failed_required_areas,
        "completion_eligible": required_reward_threshold_met
        and zero_failed_required_areas,
        "failed_required_areas": failed_required_areas,
    }


def _gate_passed(report: dict[str, Any], gate_name: str) -> bool:
    return any(gate["name"] == gate_name and gate["passed"] for gate in report["gates"])


def _path_match_tokens(path: Path) -> list[str]:
    resolved = path.resolve()
    tokens = {str(resolved).replace("\\", "/")}
    if resolved.drive:
        drive = resolved.drive.rstrip(":").lower()
        tail = resolved.as_posix().split(":", 1)[-1].lstrip("/")
        tokens.add(f"/mnt/{drive}/{tail}")
    return sorted(tokens)


def _resume_inspection_contract_status(path: Path) -> dict[str, Any]:
    inspection = _read_json_file(path)
    decisions = inspection.get("decisions", {}) if isinstance(inspection, dict) else {}
    return {
        "path": str(path),
        "exists": isinstance(inspection, dict),
        "mode": inspection.get("mode") if isinstance(inspection, dict) else None,
        "requested_max_steps": (
            inspection.get("requested_max_steps") if isinstance(inspection, dict) else None
        ),
        "reuse_stage_outputs_enabled": (
            inspection.get("reuse_stage_outputs_enabled")
            if isinstance(inspection, dict)
            else None
        ),
        "decisions": {
            stage: decisions.get(stage, {}) if isinstance(decisions, dict) else {}
            for stage in ("sft", "dpo", "rl")
        },
    }


def _full_eval_resume_stage_decision(
    *,
    stage: str,
    stage_dir: Path,
    requested_max_steps: int | None,
    reuse_stage_outputs: bool,
    resume_from_checkpoint: Path | None,
) -> dict[str, Any]:
    manifest_path = stage_dir / "training_stage_manifest.json"
    manifest = _read_json_file(manifest_path)
    manifest_error = None
    if manifest_path.exists() and not isinstance(manifest, dict):
        manifest_error = "manifest is missing or not a JSON object"
    manifest_max_steps = manifest.get("max_steps") if isinstance(manifest, dict) else None
    manifest_matches = (
        requested_max_steps is not None and manifest_max_steps == requested_max_steps
    )
    checkpoint = _resume_checkpoint_status(resume_from_checkpoint)
    if reuse_stage_outputs and manifest_matches:
        action = "reuse"
        reason = "manifest max_steps matches requested cap"
    elif checkpoint["path"]:
        action = "resume"
        reason = "resume checkpoint provided"
    elif manifest is None:
        action = "run"
        reason = "no completed stage manifest"
    else:
        action = "rerun"
        reason = "manifest max_steps does not match requested cap"
    return {
        "action": action,
        "reason": reason,
        "stage_dir": str(stage_dir),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "manifest_read_error": manifest_error,
        "manifest_max_steps": manifest_max_steps,
        "requested_max_steps": requested_max_steps,
        "manifest_matches_requested_max_steps": manifest_matches,
        "reuse_stage_outputs_enabled": reuse_stage_outputs,
        "resume_from_checkpoint": checkpoint,
        "resume_supported": stage in {"sft", "dpo"},
    }


def _resume_checkpoint_status(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": None}
    return {"path": str(path), "exists": path.exists(), "is_dir": path.is_dir()}


def _full_eval_resume_inspection_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Semantic Mirror Full-Eval Resume Inspection",
        "",
        f"- Run directory: `{report['run_dir']}`",
        f"- Reuse stage outputs: `{report['reuse_stage_outputs_enabled']}`",
        "",
        "| Stage | Action | Requested Steps | Manifest Steps | Checkpoint | Reason |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    decisions = report["decisions"]
    for stage in ("sft", "dpo", "rl"):
        decision = decisions[stage]
        checkpoint = decision["resume_from_checkpoint"]["path"] or "None"
        exists = decision["resume_from_checkpoint"]["exists"]
        if exists is True:
            checkpoint = f"{checkpoint}:exists"
        elif exists is False:
            checkpoint = f"{checkpoint}:missing"
        lines.append(
            f"| `{stage}` | `{decision['action']}` | "
            f"{decision['requested_max_steps']} | {decision['manifest_max_steps']} | "
            f"`{checkpoint}` | `{decision['reason']}` |"
        )
    return "\n".join(lines) + "\n"


def _training_diagnostics_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Semantic Mirror Training Diagnostics",
        "",
        f"- Run directory: `{summary['run_dir']}`",
        f"- Generated: `{summary['generated_at']}`",
        "",
        "| Plot | Points | Missing | Source files |",
        "| --- | ---: | --- | --- |",
    ]
    for name, plot in summary["plots"].items():
        sources = ", ".join(f"`{Path(source).name}`" for source in plot["source_files"]) or "none"
        lines.append(f"| `{name}` | {plot['points']} | {plot['missing']} | {sources} |")
    if summary["missing_metrics"]:
        lines.extend(["", "Missing metric series:", ""])
        lines.extend(f"- `{name}`" for name in summary["missing_metrics"])
    lines.append("")
    return "\n".join(lines)


def _write_metric_png(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    points: list[dict[str, Any]],
) -> None:
    width, height = 640, 360
    pixels = bytearray([255, 255, 255] * width * height)
    left, right, top, bottom = 56, width - 24, 32, height - 42
    axis_color = (45, 45, 45)
    line_color = (32, 106, 180)
    missing_color = (180, 60, 60)
    _draw_line(pixels, width, height, left, bottom, right, bottom, axis_color)
    _draw_line(pixels, width, height, left, bottom, left, top, axis_color)
    numeric_points = [point for point in points if not point.get("missing")]
    if numeric_points:
        xs = [float(point["x"]) for point in numeric_points]
        ys = [float(point["y"]) for point in numeric_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if min_y == max_y:
            min_y -= 1.0
            max_y += 1.0
        if min_x == max_x:
            min_x -= 1.0
            max_x += 1.0
        plotted: list[tuple[int, int]] = []
        for point in numeric_points:
            x = left + int((float(point["x"]) - min_x) / (max_x - min_x) * (right - left))
            y = bottom - int((float(point["y"]) - min_y) / (max_y - min_y) * (bottom - top))
            plotted.append((x, y))
        for start, end in zip(plotted, plotted[1:], strict=False):
            _draw_line(pixels, width, height, start[0], start[1], end[0], end[1], line_color)
        for x, y in plotted:
            _draw_square(pixels, width, height, x, y, 3, line_color)
    else:
        _draw_line(pixels, width, height, left, top, right, bottom, missing_color)
        _draw_line(pixels, width, height, right, top, left, bottom, missing_color)
    metadata = {"Title": title, "XAxis": x_label, "YAxis": y_label}
    _write_png(path, width, height, pixels, metadata=metadata)


def _draw_line(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _set_pixel(pixels, width, height, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _draw_square(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    for yy in range(y - radius, y + radius + 1):
        for xx in range(x - radius, x + radius + 1):
            _set_pixel(pixels, width, height, xx, yy, color)


def _set_pixel(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    offset = (y * width + x) * 3
    pixels[offset : offset + 3] = bytes(color)


def _write_png(
    path: Path,
    width: int,
    height: int,
    pixels: bytearray,
    *,
    metadata: dict[str, str],
) -> None:
    rows = []
    stride = width * 3
    for y in range(height):
        start = y * stride
        rows.append(b"\x00" + bytes(pixels[start : start + stride]))
    payload = bytearray()
    payload.extend(b"\x89PNG\r\n\x1a\n")
    payload.extend(_png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    for key, value in metadata.items():
        payload.extend(_png_chunk(b"tEXt", key.encode("latin-1") + b"\x00" + value.encode("utf-8")))
    payload.extend(_png_chunk(b"IDAT", zlib.compress(b"".join(rows), level=9)))
    payload.extend(_png_chunk(b"IEND", b""))
    path.write_bytes(bytes(payload))


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _copy_if_different(source: Path, target: Path) -> None:
    if source != target:
        shutil.copyfile(source, target)


def _sample_references(dataset: Path) -> dict[str, dict[str, Any]]:
    manifest_path = dataset / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    references: dict[str, dict[str, Any]] = {}
    for split in ("gold", "silver"):
        rel_path = manifest.get("files", {}).get(split)
        if not rel_path:
            continue
        path = dataset / rel_path
        if not path.exists():
            continue
        for record in _read_jsonl(path):
            references[record["record_id"]] = record
            references[record["unit_id"]] = record
    return references


def _sample_generation_config(rows: list[dict[str, Any]]) -> dict[str, Any]:
    configs = [
        row.get("generation_config")
        for row in rows
        if isinstance(row.get("generation_config"), dict)
    ]
    if not configs:
        return {}
    first = dict(configs[0])
    first["raw_row_generation_config_count"] = len(configs)
    first["mixed_generation_configs"] = any(config != configs[0] for config in configs)
    first["schema_prefix_applied_count"] = sum(
        1 for config in configs if config.get("schema_prefix_applied")
    )
    modes = sorted(
        {
            str(config.get("schema_prefix_mode"))
            for config in configs
            if config.get("schema_prefix_mode") is not None
        }
    )
    if modes:
        first["schema_prefix_modes"] = modes
    field_reports = [
        report
        for row in rows
        for report in row.get("field_generation_reports", [])
        if isinstance(report, dict)
    ]
    if field_reports:
        first["field_generation_report_count"] = len(field_reports)
        first["field_generation_parseable_count"] = sum(
            1 for report in field_reports if report.get("parseable")
        )
        first["field_generation_empty_count"] = sum(
            1 for report in field_reports if report.get("empty")
        )
        first["field_generation_cap_hits"] = sum(
            1 for report in field_reports if report.get("hit_generation_cap")
        )
    return first


def _sample_reference_for_row(
    row: dict[str, Any],
    references: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for key in ("dataset_record_id", "record_id", "unit_id", "positive_unit_id"):
        value = row.get(key)
        if value in references:
            return references[value]
    sir_unit = _candidate_like_sir_unit(row)
    if isinstance(sir_unit, dict) and sir_unit.get("unit_id") in references:
        return references[sir_unit["unit_id"]]
    return None


def _sample_raw_contract_report(
    row: dict[str, Any],
    references: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reference = _sample_reference_for_row(row, references)
    sir_unit = _candidate_like_sir_unit(row)
    parseable = _sample_row_parseable(row)
    report: dict[str, Any] = {
        "record_id": str(row.get("dataset_record_id") or row.get("record_id") or ""),
        "row_unit_id": row.get("unit_id"),
        "raw_unit_id": sir_unit.get("unit_id") if isinstance(sir_unit, dict) else None,
        "expected_unit_id": None if reference is None else reference["unit_id"],
        "parseable": parseable,
        "hit_generation_cap": bool(row.get("hit_generation_cap")),
        "schema_core_valid": False,
        "schema_error": None,
        "top_level_keys_valid": False,
        "missing_top_level_keys": [],
        "extra_top_level_keys": [],
        "identity_exact": False,
        "identity_mismatches": [],
        "compact_shape_valid": False,
        "list_count_overruns": {},
        "data_ml_count_overruns": {},
        "repair_free_contract_valid": False,
    }
    if reference is None:
        report["schema_error"] = "missing_reference"
        return report
    if not isinstance(sir_unit, dict):
        report["schema_error"] = "missing_sir_unit"
        return report
    if sir_unit.get("raw_error"):
        report["schema_error"] = str(sir_unit.get("raw_error"))
        return report

    try:
        validate_unit(sir_unit)
        report["schema_core_valid"] = True
    except SchemaValidationError as exc:
        report["schema_error"] = str(exc)

    target = _schema_output_template(reference)
    observed_keys = set(sir_unit)
    allowed_keys = set(SIR_UNIT_TOP_LEVEL_KEYS)
    report["missing_top_level_keys"] = [
        key for key in SIR_UNIT_TOP_LEVEL_KEYS if key not in observed_keys
    ]
    report["extra_top_level_keys"] = sorted(observed_keys - allowed_keys)
    report["top_level_keys_valid"] = (
        not report["missing_top_level_keys"] and not report["extra_top_level_keys"]
    )

    identity_mismatches = [
        field for field in SIR_IDENTITY_FIELDS if sir_unit.get(field) != target.get(field)
    ]
    report["identity_mismatches"] = identity_mismatches
    report["identity_exact"] = not identity_mismatches

    expected_counts = _compact_expected_counts(target)
    list_overruns: dict[str, dict[str, int]] = {}
    for field in SIR_LIST_FIELDS:
        observed = sir_unit.get(field)
        observed_count = len(observed) if isinstance(observed, list) else 0
        expected_count = int(expected_counts.get(field, 0))
        if observed_count > expected_count:
            list_overruns[field] = {
                "observed": observed_count,
                "expected": expected_count,
            }
    data_ml_overruns: dict[str, dict[str, int]] = {}
    raw_details = sir_unit.get("data_ml_details", {})
    expected_details = expected_counts.get("data_ml_details", {})
    for category in DATA_ML_DETAIL_CATEGORIES:
        observed = raw_details.get(category) if isinstance(raw_details, dict) else None
        observed_count = len(observed) if isinstance(observed, list) else 0
        expected_count = int(expected_details.get(category, 0))
        if observed_count > expected_count:
            data_ml_overruns[category] = {
                "observed": observed_count,
                "expected": expected_count,
            }
    report["list_count_overruns"] = list_overruns
    report["data_ml_count_overruns"] = data_ml_overruns
    report["compact_shape_valid"] = not list_overruns and not data_ml_overruns
    report["repair_free_contract_valid"] = (
        report["parseable"]
        and report["schema_core_valid"]
        and report["top_level_keys_valid"]
        and report["identity_exact"]
        and report["compact_shape_valid"]
        and not report["hit_generation_cap"]
    )
    return report


def _sample_row_parseable(row: dict[str, Any]) -> bool:
    if row.get("repair_applied") is False and isinstance(row.get("raw_text"), str):
        sir_unit = _raw_text_sir_unit(row)
        return isinstance(sir_unit, dict) and not sir_unit.get("raw_error")
    if isinstance(row.get("raw_parseable"), bool):
        return bool(row["raw_parseable"])
    sir_unit = _candidate_like_sir_unit(row)
    return isinstance(sir_unit, dict) and not sir_unit.get("raw_error")


def _candidate_like_sir_unit(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("repair_applied") is False:
        raw_text_unit = _raw_text_sir_unit(row)
        if raw_text_unit is not None:
            return raw_text_unit
    for key in ("sir_unit", "raw_sir_unit", "candidate", "output"):
        value = row.get(key)
        if isinstance(value, dict):
            return value.get("sir_unit") if isinstance(value.get("sir_unit"), dict) else value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _raw_text_sir_unit(row: dict[str, Any]) -> dict[str, Any] | None:
    raw_text = row.get("raw_text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    decoder = json.JSONDecoder()
    last_error = "no JSON object found"
    first_json = raw_text.find("{")
    indices = [first_json] if first_json >= 0 and not raw_text[:first_json].strip() else [
        index for index, char in enumerate(raw_text) if char == "{"
    ]
    for index in indices:
        if index < 0:
            continue
        char = raw_text[index]
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if isinstance(parsed, dict) and "unit_id" in parsed:
            return parsed
    return {"unit_id": "<unparseable>", "raw_error": f"no SIR JSON object found: {last_error}"}


def _sample_inspection_markdown(
    manifest: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    repaired_rows: list[dict[str, Any]],
    raw_eval: dict[str, Any],
    repaired_eval: dict[str, Any],
) -> str:
    raw_by_key = {_sample_row_key(row): row for row in raw_rows}
    repaired_by_key = {_sample_row_key(row): row for row in repaired_rows}
    raw_results = {_sample_result_key(row): row for row in raw_eval["results"]}
    repaired_results = {_sample_result_key(row): row for row in repaired_eval["results"]}
    raw_contracts = {
        str(report.get("row_unit_id") or report.get("expected_unit_id") or report.get("record_id")): report
        for report in manifest.get("raw_contract_reports", [])
    }
    lines = [
        "# Semantic Mirror Sample Inspection",
        "",
        f"- Model: `{manifest['model_name']}`",
        f"- Dataset: `{manifest['dataset']}`",
        f"- Raw parseable: {manifest['raw_parseability_count']} / {manifest['raw_candidate_count']}",
        (
            f"- Raw generation cap hits: {manifest['raw_generation_cap_hits']} / "
            f"{manifest['raw_candidate_count']}"
        ),
        (
            f"- Raw schema valid: {manifest['raw_schema_validity_count']} / "
            f"{manifest['raw_candidate_count']}"
        ),
        (
            f"- Raw repair-free contract valid: {manifest['raw_repair_free_contract_count']} / "
            f"{manifest['raw_candidate_count']}"
        ),
        (
            f"- Raw exact identity: {manifest['raw_exact_identity_count']} / "
            f"{manifest['raw_candidate_count']}"
        ),
        (
            f"- Raw top-level key valid: {manifest['raw_top_level_key_validity_count']} / "
            f"{manifest['raw_candidate_count']}"
        ),
        (
            f"- Raw compact shape valid: {manifest['raw_compact_shape_count']} / "
            f"{manifest['raw_candidate_count']}"
        ),
        (
            f"- Repaired schema valid: {manifest['repaired_schema_validity_count']} / "
            f"{manifest['repaired_candidate_count']}"
        ),
        "",
    ]
    generation_config = manifest.get("generation_config", {})
    if generation_config.get("generation_mode") == "field-wise":
        lines[14:14] = [
            (
                "- Field fragments parseable: "
                f"{generation_config.get('field_generation_parseable_count', 0)} / "
                f"{generation_config.get('field_generation_report_count', 0)}"
            ),
            (
                "- Field fragments empty: "
                f"{generation_config.get('field_generation_empty_count', 0)} / "
                f"{generation_config.get('field_generation_report_count', 0)}"
            ),
            (
                "- Field fragment cap hits: "
                f"{generation_config.get('field_generation_cap_hits', 0)} / "
                f"{generation_config.get('field_generation_report_count', 0)}"
            ),
        ]
    for key in sorted(repaired_by_key):
        raw_row = raw_by_key.get(key, {})
        repaired_row = repaired_by_key[key]
        unit_id = repaired_row.get("unit_id") or key
        raw_result = raw_results.get(unit_id, {})
        repaired_result = repaired_results.get(unit_id, {})
        raw_contract = raw_contracts.get(str(unit_id)) or raw_contracts.get(str(key)) or {}
        raw_parse_error = (
            raw_row.get("raw_parse_error")
            or raw_contract.get("schema_error")
            or "none"
        )
        lines.extend(
            [
                f"## `{unit_id}`",
                "",
                f"- Source: `{repaired_row.get('source_path', '<unknown>')}`",
                f"- Raw score: `{raw_result.get('score', 'n/a')}`",
                f"- Repaired score: `{repaired_result.get('score', 'n/a')}`",
                f"- Raw parse error: `{raw_parse_error}`",
                f"- Raw hit generation cap: `{raw_row.get('hit_generation_cap', False)}`",
                f"- Raw schema valid: `{raw_result.get('schema_valid', False)}`",
                (
                    f"- Raw repair-free contract valid: "
                    f"`{raw_contract.get('repair_free_contract_valid', False)}`"
                ),
                (
                    f"- Raw identity mismatches: "
                    f"`{_format_list(raw_contract.get('identity_mismatches', []))}`"
                ),
                (
                    f"- Raw extra top-level keys: "
                    f"`{_format_list(raw_contract.get('extra_top_level_keys', []))}`"
                ),
                (
                    f"- Raw missing top-level keys: "
                    f"`{_format_list(raw_contract.get('missing_top_level_keys', []))}`"
                ),
                (
                    f"- Raw list overruns: "
                    f"`{_format_mapping(raw_contract.get('list_count_overruns', {}))}`"
                ),
                (
                    f"- Raw data/ML overruns: "
                    f"`{_format_mapping(raw_contract.get('data_ml_count_overruns', {}))}`"
                ),
                f"- Repaired schema valid: `{repaired_result.get('schema_valid', False)}`",
                "",
                "Raw output:",
                "",
                "```text",
                _truncate(str(raw_row.get("raw_text") or raw_row.get("output") or ""), 1600),
                "```",
                "",
                "Repaired candidate:",
                "",
                "```json",
                _truncate(json.dumps(repaired_row.get("sir_unit", {}), indent=2), 1600),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _format_list(values: Any) -> str:
    if not values:
        return "none"
    return ", ".join(str(value) for value in values)


def _format_mapping(value: Any) -> str:
    if not value:
        return "none"
    return json.dumps(value, sort_keys=True)


def _sample_row_key(row: dict[str, Any]) -> str:
    return str(row.get("unit_id") or row.get("dataset_record_id") or row.get("record_id"))


def _sample_result_key(row: dict[str, Any]) -> str:
    return str(row.get("unit_id") or row.get("index"))


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 32].rstrip() + "\n... truncated ..."


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
