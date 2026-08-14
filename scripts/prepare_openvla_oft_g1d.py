#!/usr/bin/env python3
"""Prepare and audit G1-D Family Home data without starting fine-tuning.

This preflight is intentionally CPU-only.  It validates the local OpenVLA
checkpoint, the isolated openvla-oft environment, the upstream OFT checkout,
and every accepted Expert episode.  It then writes the canonical training
manifest and a machine-readable readiness report.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.g1d_openvla_oft_data import build_manifest


REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "configuration_prismatic.py",
    "modeling_prismatic.py",
    "processing_prismatic.py",
)
REQUIRED_PACKAGES = ("torch", "transformers", "peft", "draccus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=ROOT / "outputs/family_home_vln/expert_demos_head",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/family_home_vln/openvla_oft/manifest.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/family_home_vln/openvla_oft/preflight.json",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "checkpoints/openvla-7b",
    )
    parser.add_argument("--min-episodes", type=int, default=100)
    parser.add_argument("--min-action-samples", type=int, default=4000)
    return parser.parse_args()


def _check_model(model_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    shards: list[str] = []
    for name in REQUIRED_MODEL_FILES:
        if not (model_dir / name).is_file():
            errors.append(f"missing model file: {name}")
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shards = sorted(set(index.get("weight_map", {}).values()))
            for shard in shards:
                path = model_dir / shard
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append(f"missing or empty model shard: {shard}")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            errors.append(f"invalid model index: {exc}")
    return errors, shards


def _package_versions() -> tuple[dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    errors: list[str] = []
    for name in REQUIRED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing Python package: {name}")
    return versions, errors


def main() -> int:
    args = parse_args()
    paths = {
        "demo_dir": args.demo_dir.resolve(),
        "manifest": args.manifest.resolve(),
        "report": args.report.resolve(),
        "model": args.model.resolve(),
        "oft_checkout": (ROOT / "third_party/openvla-oft").resolve(),
        "training_launcher": (ROOT / "scripts/train_openvla_oft_g1d.sh").resolve(),
    }
    errors: list[str] = []
    warnings: list[str] = []

    expected_python = (ROOT / ".conda/envs/openvla-oft/bin/python").resolve()
    running_python = Path(sys.executable).resolve()
    if running_python != expected_python:
        errors.append(
            f"wrong Python: {running_python}; expected isolated environment "
            f"{expected_python}"
        )

    if not paths["demo_dir"].is_dir():
        errors.append(f"demo directory does not exist: {paths['demo_dir']}")
    if not (paths["oft_checkout"] / "vla-scripts/finetune.py").is_file():
        errors.append("OpenVLA-OFT checkout is incomplete")
    if not paths["training_launcher"].is_file():
        errors.append("G1-D OFT training launcher is missing")

    model_errors, model_shards = _check_model(paths["model"])
    errors.extend(model_errors)
    packages, package_errors = _package_versions()
    errors.extend(package_errors)

    manifest: dict[str, Any] | None = None
    if not errors:
        try:
            manifest = build_manifest(paths["demo_dir"], paths["manifest"])
        except Exception as exc:
            errors.append(f"dataset audit failed: {exc}")

    dataset_ready = False
    if manifest is not None:
        episode_count = int(manifest["episode_count"])
        sample_count = int(manifest["sample_count"])
        if episode_count < args.min_episodes:
            warnings.append(
                f"only {episode_count}/{args.min_episodes} successful episodes"
            )
        if sample_count < args.min_action_samples:
            warnings.append(
                f"only {sample_count}/{args.min_action_samples} action-chunk samples"
            )
        dataset_ready = (
            episode_count >= args.min_episodes
            and sample_count >= args.min_action_samples
        )

    software_ready = not errors
    training_command = (
        f"CUDA_VISIBLE_DEVICES=6,7 G1D_OFT_NPROC=2 "
        f"{paths['training_launcher']} --demo-dir {paths['demo_dir']} "
        f"--manifest {paths['manifest']} --model {paths['model']} "
        f"--output {ROOT / 'checkpoints/openvla-oft-g1d'}"
    )
    report = {
        "schema_version": 1,
        "fine_tuning_started": False,
        "software_ready": software_ready,
        "dataset_ready": dataset_ready,
        "ready_to_train": software_ready and dataset_ready,
        "paths": {key: str(value) for key, value in paths.items()},
        "python": str(running_python),
        "packages": packages,
        "model_shards": model_shards,
        "dataset": None
        if manifest is None
        else {
            "episode_count": manifest["episode_count"],
            "frame_count": manifest["frame_count"],
            "action_chunk_samples": manifest["sample_count"],
            "rejected": manifest["rejected"],
            "action_frame": manifest["action_frame"],
            "unnorm_key": manifest["unnorm_key"],
            "gripper_convention": manifest["gripper_convention"],
            "observation": manifest["observation"],
        },
        "minimum_dataset_gate": {
            "successful_episodes": args.min_episodes,
            "action_chunk_samples": args.min_action_samples,
        },
        "errors": errors,
        "warnings": warnings,
        "training_command_not_executed": training_command,
    }
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if software_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
