"""Prepare SFT and RL training artifacts from curated Semantic Mirror datasets."""

from __future__ import annotations

import ast
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from semantic_mirror.schema import DATA_ML_DETAIL_CATEGORIES

TRAINING_VERSION = "0.1.0"
DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
REQUIRED_TRAINING_MODULES = (
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
            "method": "QLoRA",
            "target_model_size": "7-14B",
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
    module_status = {
        module: _module_available(module, module_probe=module_probe)
        for module in REQUIRED_TRAINING_MODULES
    }
    missing_modules = [module for module, available in module_status.items() if not available]
    torch_info = _probe_torch(torch_probe=torch_probe)
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
    current_python_version = python_version or platform.python_version()
    python_supported = _python_version_supported_for_unsloth(current_python_version)

    checks = [
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
            actual={"missing": missing_modules, "available": module_status},
            detail=(
                "Unsloth, TRL, DPO optional dependencies, datasets, transformers, "
                "torch, and bitsandbytes are importable."
            ),
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
            },
            detail="CUDA is required for the default 7-14B QLoRA training target.",
        ),
        _check(
            "hf_token_available",
            secrets["hf_token_present"] or not require_hf_token,
            required=require_hf_token,
            actual={"present": secrets["hf_token_present"]},
            detail="A Hugging Face token is available for model or dataset downloads.",
        ),
    ]
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
            "python_executable": sys.executable,
            "python_version": current_python_version,
            "platform": current_platform,
            "platform_release": platform.release(),
            "loaded_env_file": None if loaded_env_path is None else str(loaded_env_path),
            "secrets": secrets,
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


def package_training_bundle(
    training_path: Path | str,
    out_path: Path | str,
    *,
    env_file: Path | str | None = None,
    require_gpu: bool = True,
    require_hf_token: bool = False,
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
    _write_launch_scripts(launch_target)
    _write_bootstrap_scripts(setup_target)
    (out / "README.md").write_text(
        _training_package_readme(audit),
        encoding="utf-8",
    )

    files = {
        "training_dir": "training",
        "runtime_source": "src/semantic_mirror",
        "project_config": "pyproject.toml",
        "requirements": "requirements-training.txt",
        "environment_example": ".env.training.example",
        "environment_guide": "ENVIRONMENT.md",
        "audit": "audit/current_environment.json",
        "launch_commands": "launch/commands.json",
        "linux_cuda_bootstrap": "setup/bootstrap_linux_cuda.sh",
        "wsl_bootstrap": "setup/bootstrap_wsl_ubuntu.ps1",
        "full_training_eval_launcher": "launch/run_full_training_eval.sh",
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
        "files": files,
        "launch_commands": commands,
    }
    _write_package_manifest(out, manifest)
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
                "content": json.dumps(compact_unit, sort_keys=True),
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
        "chosen": json.dumps(_compact_sir_unit(positive["target"]["sir_unit"]), sort_keys=True),
        "rejected": json.dumps(_compact_sir_unit(negative["candidate"]["sir_unit"]), sort_keys=True),
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
    return {
        "record_id": f"rl-prompt-{index}-{record['record_id']}",
        "prompt": _generation_user_prompt(record),
        "reward_reference": {
            "unit_id": record["unit_id"],
            "static_facts": record["static_facts"],
            "source_path": record["source_path"],
            "source_spans": record["source_spans"],
        },
        "metadata": _record_metadata(record),
    }


def _generation_system_prompt() -> str:
    return (
        "You generate one valid Semantic Mirror SIR JSON unit. Use the exact required schema keys. "
        "Preserve the bounded source-backed static facts supplied in the prompt. Do not invent "
        "behavior. Return only JSON for the sir_unit object."
    )


def _repair_system_prompt() -> str:
    return (
        "You critique and repair Semantic IR. Label missing or invented behavior using the "
        "verifier evidence, then produce a corrected source-backed SIR unit. Do not reward "
        "brevity when required source facts are missing."
    )


def _generation_user_prompt(record: dict[str, Any]) -> str:
    code_slice = record["code_slice"]
    prompt = {
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
        "schema_contract": _schema_contract(record),
        "static_facts": _compact_static_facts(record["static_facts"]),
        "static_analysis": _compact_static_analysis(record.get("static_analysis", {})),
        "requested_output": "faithful SIR JSON unit",
    }
    return json.dumps(prompt, indent=2, sort_keys=True)


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
    return json.dumps(prompt, indent=2, sort_keys=True)


def _schema_contract(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "required_top_level_keys": [
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
    return {
        "base_model": base_model,
        "method": "QLoRA",
        "model_size_target": "7-14B",
        "load_in_4bit": True,
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
- `unsloth_sft_config.json` and `rl_reward_config.json` capture the QLoRA and reward defaults.
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
        shutil.copy2(pyproject, out / "pyproject.toml")
        return
    (out / "pyproject.toml").write_text(
        """[project]
name = "semantic-mirror-runtime"
version = "0.1.0"
requires-python = ">=3.11"
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
transformers
trl
mergekit
llm-blender
weave
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

The default target is a CUDA Linux or WSL runtime suitable for Unsloth QLoRA on
a 7-14B model. Use Python {UNSLOTH_PYTHON_RANGE}; Python 3.14 is intentionally
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
        "install": "python -m pip install --upgrade pip && python -m pip install -r requirements-training.txt",
        "sft": "python training/run_unsloth_sft.py --training-dir training --output-dir outputs/semantic-mirror-sft",
        "dpo": (
            "python training/run_preference_dpo.py --training-dir training "
            "--model-name-or-path outputs/semantic-mirror-sft --output-dir outputs/semantic-mirror-dpo"
        ),
        "rl": (
            "python training/run_reward_rl.py --training-dir training "
            "--model-name-or-path outputs/semantic-mirror-dpo --output-dir outputs/semantic-mirror-rl"
        ),
        "full_training_eval": (
            "HELD_OUT_DATASET=<dataset_dir> "
            "BASELINE_CANDIDATES=<teacher_results_dir>/teacher_candidates.jsonl "
            "bash launch/run_full_training_eval.sh"
        ),
        "generate_candidates": (
            "python training/generate_sir_candidates.py --training-dir training "
            "--model-name-or-path outputs/semantic-mirror-rl --out outputs/candidates.jsonl"
        ),
        "score_candidates": (
            "PYTHONPATH=src python training/score_sir_candidates.py "
            "--repo <held_out_repo_path> --candidates outputs/candidates.jsonl --out outputs/candidate_scores.jsonl"
        ),
        "eval_candidates": (
            "PYTHONPATH=src python -m semantic_mirror.cli eval candidates <dataset_dir> "
            "--candidates outputs/candidates.jsonl --model-name semantic-mirror-rl --out outputs/rl_eval.json"
        ),
        "compare_sft": (
            "PYTHONPATH=src python -m semantic_mirror.cli eval model-compare "
            "outputs/baseline_eval.json outputs/sft_eval.json --stage sft --out outputs/sft_vs_baseline.json"
        ),
        "compare_rl": (
            "PYTHONPATH=src python -m semantic_mirror.cli eval model-compare "
            "outputs/sft_eval.json outputs/rl_eval.json --stage rl --out outputs/rl_vs_sft.json"
        ),
    }


def _write_launch_scripts(launch_target: Path) -> None:
    scripts = {
        "run_sft.sh": """#!/usr/bin/env bash
set -euo pipefail
python training/run_unsloth_sft.py --training-dir training --output-dir outputs/semantic-mirror-sft
""",
        "run_dpo.sh": """#!/usr/bin/env bash
set -euo pipefail
python training/run_preference_dpo.py --training-dir training --model-name-or-path outputs/semantic-mirror-sft --output-dir outputs/semantic-mirror-dpo
""",
        "run_rl.sh": """#!/usr/bin/env bash
set -euo pipefail
python training/run_reward_rl.py --training-dir training --model-name-or-path outputs/semantic-mirror-dpo --output-dir outputs/semantic-mirror-rl
""",
        "generate_candidates.sh": """#!/usr/bin/env bash
set -euo pipefail
python training/generate_sir_candidates.py --training-dir training --model-name-or-path outputs/semantic-mirror-rl --out outputs/candidates.jsonl
""",
        "score_candidates.sh": """#!/usr/bin/env bash
set -euo pipefail
: "${HELD_OUT_REPO:?set HELD_OUT_REPO to the source repo path}"
PYTHONPATH=src python training/score_sir_candidates.py --repo "$HELD_OUT_REPO" --candidates outputs/candidates.jsonl --out outputs/candidate_scores.jsonl
""",
        "run_full_training_eval.sh": """#!/usr/bin/env bash
set -euo pipefail

: "${HELD_OUT_DATASET:?set HELD_OUT_DATASET to a dataset directory containing manifest.json}"
: "${BASELINE_CANDIDATES:?set BASELINE_CANDIDATES to baseline candidate JSONL for the held-out dataset, for example teacher_results/teacher_candidates.jsonl}"

mkdir -p outputs

python training/run_unsloth_sft.py --training-dir training --output-dir outputs/semantic-mirror-sft
PYTHONPATH=src python -m semantic_mirror.cli eval candidates "$HELD_OUT_DATASET" \
  --candidates "$BASELINE_CANDIDATES" \
  --model-name baseline \
  --out outputs/baseline_eval.json

python training/generate_sir_candidates.py --training-dir training \
  --model-name-or-path outputs/semantic-mirror-sft \
  --out outputs/sft_candidates.jsonl
PYTHONPATH=src python -m semantic_mirror.cli eval candidates "$HELD_OUT_DATASET" \
  --candidates outputs/sft_candidates.jsonl \
  --model-name semantic-mirror-sft \
  --out outputs/sft_eval.json
PYTHONPATH=src python -m semantic_mirror.cli eval model-compare \
  outputs/baseline_eval.json outputs/sft_eval.json \
  --stage sft \
  --out outputs/sft_vs_baseline.json

python training/run_preference_dpo.py --training-dir training \
  --model-name-or-path outputs/semantic-mirror-sft \
  --output-dir outputs/semantic-mirror-dpo
python training/run_reward_rl.py --training-dir training \
  --model-name-or-path outputs/semantic-mirror-dpo \
  --output-dir outputs/semantic-mirror-rl
python training/generate_sir_candidates.py --training-dir training \
  --model-name-or-path outputs/semantic-mirror-rl \
  --out outputs/rl_candidates.jsonl
PYTHONPATH=src python -m semantic_mirror.cli eval candidates "$HELD_OUT_DATASET" \
  --candidates outputs/rl_candidates.jsonl \
  --model-name semantic-mirror-rl \
  --out outputs/rl_eval.json
PYTHONPATH=src python -m semantic_mirror.cli eval model-compare \
  outputs/sft_eval.json outputs/rl_eval.json \
  --stage rl \
  --out outputs/rl_vs_sft.json

python - <<'PY'
import json
from pathlib import Path

reports = {
    "baseline_eval": "outputs/baseline_eval.json",
    "sft_eval": "outputs/sft_eval.json",
    "sft_vs_baseline": "outputs/sft_vs_baseline.json",
    "rl_eval": "outputs/rl_eval.json",
    "rl_vs_sft": "outputs/rl_vs_sft.json",
}
summary = {}
for name, path in reports.items():
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    summary[name] = {
        "passed": data.get("passed"),
        "mode": data.get("mode"),
        "metrics": data.get("metrics", {}),
        "deltas": data.get("deltas", {}),
        "gates": data.get("gates", []),
    }
Path("outputs/training_eval_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
if not all(item["passed"] for item in summary.values()):
    raise SystemExit("One or more training/evaluation gates failed. See outputs/training_eval_summary.json")
PY
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
$wslPath = (wsl.exe -d $Distro -- wslpath -a "$windowsPath").Trim()
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
bash setup/bootstrap_linux_cuda.sh
source .venv/bin/activate
bash launch/run_sft.sh
bash launch/run_dpo.sh
bash launch/run_rl.sh
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

This writes `outputs/baseline_eval.json`, `outputs/sft_eval.json`,
`outputs/rl_eval.json`, `outputs/sft_vs_baseline.json`,
`outputs/rl_vs_sft.json`, and `outputs/training_eval_summary.json`.

The bundle does not include `.env` or secret values. Use `.env.training.example`
as a template on the target machine.
"""


def _unsloth_sft_script() -> str:
    return '''"""Run Unsloth QLoRA SFT for Semantic Mirror data.

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
    args = parser.parse_args()

    root = Path(args.training_dir)
    config = json.loads((root / "unsloth_sft_config.json").read_text(encoding="utf-8"))
    dataset = load_dataset("json", data_files=str(root / config["inputs"]["sft_jsonl"]), split="train")
    dataset = dataset.map(lambda row: {"text": _messages_to_text(row["messages"])})

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model"],
        max_seq_length=config["training"]["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
    )
    tokenizer.truncation_side = "left"
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
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
            optim="adamw_8bit",
            seed=42,
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


def _messages_to_text(messages: list[dict[str, str]]) -> str:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", default=".")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--output-dir", default="outputs/semantic-mirror-dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    root = Path(args.training_dir)
    sft_config = json.loads((root / "unsloth_sft_config.json").read_text(encoding="utf-8"))
    reward_config = json.loads((root / "rl_reward_config.json").read_text(encoding="utf-8"))
    dataset = load_dataset(
        "json",
        data_files=str(root / reward_config["inputs"]["preference_pairs_jsonl"]),
        split="train",
    )

    model_name = args.model_name_or_path or sft_config["base_model"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=sft_config["training"]["max_seq_length"],
        load_in_4bit=sft_config["load_in_4bit"],
    )
    tokenizer.truncation_side = "left"
    model = FastLanguageModel.get_peft_model(
        model,
        r=sft_config["lora"]["r"],
        lora_alpha=sft_config["lora"]["alpha"],
        lora_dropout=sft_config["lora"]["dropout"],
        target_modules=sft_config["lora"]["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=43,
    )
    try:
        model.warnings_issued
    except AttributeError:
        model.warnings_issued = {}
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
            save_steps=50,
            report_to="none",
            seed=43,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


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


REWARD_FIELDS = (
    "calls",
    "control_flow",
    "side_effects",
    "returns",
    "writes",
    "state_mutations",
    "failure_modes",
)


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
    args = parser.parse_args()

    root = Path(args.training_dir)
    sft_config = json.loads((root / "unsloth_sft_config.json").read_text(encoding="utf-8"))
    reward_config = json.loads((root / "rl_reward_config.json").read_text(encoding="utf-8"))
    prompts = _read_jsonl(root / reward_config["inputs"]["rl_prompts_jsonl"])
    preferences = _preferences_by_prompt(root, reward_config["inputs"]["preference_pairs_jsonl"])
    if not prompts:
        raise ValueError("rl_prompts.jsonl is empty")

    model_name = args.model_name_or_path or sft_config["base_model"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=sft_config["training"]["max_seq_length"],
        load_in_4bit=sft_config["load_in_4bit"],
    )
    tokenizer.truncation_side = "left"
    model = FastLanguageModel.get_peft_model(
        model,
        r=sft_config["lora"]["r"],
        lora_alpha=sft_config["lora"]["alpha"],
        lora_dropout=sft_config["lora"]["dropout"],
        target_modules=sft_config["lora"]["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=44,
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
        formatted_prompt = _format_generation_prompt(record["prompt"])
        encoded = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_len = encoded["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=generation_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                min_new_tokens=8,
            )
        output_ids = output_ids.detach().clone().to(device)
        completion = output_ids[:, prompt_len:]
        text = tokenizer.decode(completion[0], skip_special_tokens=True)
        raw_sir_unit = _extract_json_object(text)
        sir_unit = _repair_sir_unit(raw_sir_unit, record["metadata"], record["reward_reference"])
        sir_unit = _apply_faithfulness_repair(
            sir_unit,
            record["reward_reference"].get("static_facts", {}),
        )
        reward = _semantic_reward(sir_unit, record["reward_reference"], reward_config)
        reward += _preference_bonus(sir_unit, preferences.get(record["prompt"]))
        reward += _raw_generation_bonus(raw_sir_unit, int(completion.shape[1]))
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
                "advantage": round(float(advantage), 4),
                "loss": round(float(loss.detach().cpu()), 6),
                "completion_tokens": int(completion.shape[1]),
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
                "average_reward": round(sum(item["reward"] for item in history) / len(history), 6),
                "history": history,
            },
            indent=2,
            sort_keys=True,
        )
        + "\\n",
        encoding="utf-8",
    )
    return 0


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


def _apply_faithfulness_repair(unit: dict, static_facts: dict) -> dict:
    if not isinstance(unit, dict) or unit.get("raw_error") or not isinstance(static_facts, dict):
        return unit
    if isinstance(static_facts.get("algorithm"), dict):
        unit["algorithm"] = static_facts["algorithm"]
    for field in LIST_FIELDS:
        if isinstance(static_facts.get(field), list):
            unit[field] = static_facts[field]
    data_ml_details = static_facts.get("data_ml_details", {})
    if isinstance(data_ml_details, dict):
        unit["data_ml_details"] = {
            category: data_ml_details.get(category, [])
            if isinstance(data_ml_details.get(category), list)
            else []
            for category in DATA_ML_DETAIL_CATEGORIES
        }
    unit["confidence"] = unit.get("confidence", unit.get("algorithm", {}).get("confidence", 0.7))
    return unit


def _raw_generation_bonus(raw_sir_unit: dict, completion_tokens: int) -> float:
    if not isinstance(raw_sir_unit, dict) or raw_sir_unit.get("raw_error"):
        return -5.0
    reward = 1.0
    if completion_tokens <= 8:
        reward -= 2.0
    for key in ("unit_id", "algorithm", "data_ml_details"):
        if key in raw_sir_unit:
            reward += 0.5
    if any(key in raw_sir_unit for key in ("calls", "writes", "returns", "state_mutations")):
        reward += 1.0
    return reward


_GENERATION_SYSTEM_PROMPT = (
    "You generate one valid Semantic Mirror SIR JSON unit. Use the exact required schema keys. "
    "Preserve the bounded source-backed static facts supplied in the prompt. Do not invent "
    "behavior. Return only JSON for the sir_unit object."
)


def _format_generation_prompt(user_prompt: str) -> str:
    return (
        f"<|SYSTEM|>\\n{_GENERATION_SYSTEM_PROMPT}\\n\\n"
        f"<|USER|>\\n{user_prompt}\\n\\n"
        "Return only the faithful SIR JSON object. Do not continue the input JSON.\\n\\n"
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
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"unit_id": "<unparseable>", "raw_error": "no JSON object found"}
    return _parse_json_object(text[start : end + 1])


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", default=".")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--no-faithfulness-repair", action="store_true")
    args = parser.parse_args()

    root = Path(args.training_dir)
    config = json.loads((root / "unsloth_sft_config.json").read_text(encoding="utf-8"))
    prompts = _read_jsonl(root / "rl_prompts.jsonl")
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=config["training"]["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
    )
    FastLanguageModel.for_inference(model)
    tokenizer.truncation_side = "left"
    generation_tokens = min(
        args.max_new_tokens,
        max(config["training"]["max_seq_length"] - 128, 1),
    )
    max_prompt_tokens = max(128, config["training"]["max_seq_length"] - generation_tokens)
    rows = []
    for prompt in prompts:
        formatted_prompt = _format_generation_prompt(prompt["prompt"])
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        )
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        output_ids = model.generate(
            **inputs,
            max_new_tokens=generation_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        completion_ids = output_ids[:, prompt_len:]
        metadata = prompt["metadata"]
        text = tokenizer.decode(completion_ids[0], skip_special_tokens=True)
        sir_unit = _repair_sir_unit(
            _extract_json_object(text),
            metadata,
            prompt.get("reward_reference", {}),
        )
        if not args.no_faithfulness_repair:
            sir_unit = _apply_faithfulness_repair(
                sir_unit,
                prompt.get("reward_reference", {}).get("static_facts", {}),
            )
        rows.append(
            {
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
                "sir_unit": sir_unit,
            }
        )
    Path(args.out).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\\n" for row in rows),
        encoding="utf-8",
    )
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


def _apply_faithfulness_repair(unit: dict, static_facts: dict) -> dict:
    if not isinstance(unit, dict) or unit.get("raw_error") or not isinstance(static_facts, dict):
        return unit
    if isinstance(static_facts.get("algorithm"), dict):
        unit["algorithm"] = static_facts["algorithm"]
    for field in LIST_FIELDS:
        if isinstance(static_facts.get(field), list):
            unit[field] = static_facts[field]
    data_ml_details = static_facts.get("data_ml_details", {})
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
    "You generate one valid Semantic Mirror SIR JSON unit. Use the exact required schema keys. "
    "Preserve the bounded source-backed static facts supplied in the prompt. Do not invent "
    "behavior. Return only JSON for the sir_unit object."
)


def _format_generation_prompt(user_prompt: str) -> str:
    return (
        f"<|SYSTEM|>\\n{_GENERATION_SYSTEM_PROMPT}\\n\\n"
        f"<|USER|>\\n{user_prompt}\\n\\n"
        "Return only the faithful SIR JSON object. Do not continue the input JSON.\\n\\n"
        "<|ASSISTANT|>\\n"
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"unit_id": "<unparseable>", "raw_error": "no JSON object found"}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return {"unit_id": "<unparseable>", "raw_error": str(exc)}


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
    if config.get("method") != "QLoRA":
        issues.append({"kind": "unsupported_sft_method", "actual": config.get("method")})
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


def _module_available(module: str, *, module_probe: ModuleProbe | None) -> bool:
    if module_probe is not None:
        return bool(module_probe(module))
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


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
        "device_count": 0,
        "devices": [],
    }
    if info["cuda_available"]:
        try:
            info["device_count"] = torch.cuda.device_count()
            info["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(index).total_memory / (1024**3),
                        2,
                    ),
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
    if stage == "rl":
        if model_name_or_path:
            command.extend(["--model-name-or-path", model_name_or_path])
        if max_steps is not None:
            command.extend(["--max-steps", str(max_steps)])
        command.extend(["--kl-coef", str(kl_coef)])
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
            "transformers, torch, and bitsandbytes."
        )
    if require_gpu:
        hints.append("Run on a CUDA machine before launching the default 7-14B QLoRA scripts.")
    return hints


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
