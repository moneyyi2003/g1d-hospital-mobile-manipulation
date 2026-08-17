#!/usr/bin/env python3
"""Audit Family Home Expert trajectories and expose them to OpenVLA-OFT.

The source episodes stay untouched.  This module writes a compact manifest
that references the original RGB/action files and implements the same sample
contract as OpenVLA-OFT's ``RLDSBatchTransform`` without requiring a TFDS
conversion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ACTION_LABELS = (
    "dx_m",
    "dy_m",
    "dz_m",
    "droll_rad",
    "dpitch_rad",
    "dyaw_rad",
    "gripper",
)
ACTION_CHUNK = 8
DATASET_NAME = "g1d_family_home_pick"
UNNORM_KEY = "g1d_family_home_cup_head"
ACTION_FRAME = "world"
GRIPPER_CONVENTION = "1=open,0=closed"
EXPECTED_IMAGE_SIZE = (640, 480)
EXPECTED_CAPTURE_HZ = 10
EXPECTED_NEAR_CLIP_M = 0.1
EXPECTED_FAR_CLIP_M = 1_000_000.0
IGNORE_INDEX = -100


def _black_metrics(image: np.ndarray) -> tuple[float, float, bool]:
    luminance = image[..., :3].astype(np.float32).mean(axis=2)
    black = luminance < 12.0
    bottom = black[int(black.shape[0] * 0.55) :]
    total_fraction = float(black.mean())
    bottom_fraction = float(bottom.mean()) if bottom.size else 1.0
    return total_fraction, bottom_fraction, bool(
        total_fraction > 0.18 or bottom_fraction > 0.45
    )


def _load_action(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("frame") != ACTION_FRAME:
        raise ValueError(f"unsupported action frame: {payload.get('frame')!r}")
    if payload.get("unnorm_key") != UNNORM_KEY:
        raise ValueError(
            f"unexpected unnorm_key: {payload.get('unnorm_key')!r}; "
            f"expected {UNNORM_KEY!r}"
        )
    action = np.asarray([payload[key] for key in ACTION_LABELS], dtype=np.float32)
    if not np.isfinite(action).all():
        raise ValueError("non-finite action")
    if np.max(np.abs(action[:3])) > 0.08:
        raise ValueError("translation delta exceeds 8 cm/sample")
    if np.max(np.abs(action[3:6])) > 0.8:
        raise ValueError("rotation delta exceeds 0.8 rad/sample")
    if not 0.0 <= float(action[6]) <= 1.0:
        raise ValueError("gripper is outside [0, 1]")
    return action


def build_manifest(demo_dir: Path, manifest_path: Path) -> dict[str, Any]:
    demo_dir = demo_dir.resolve()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    all_actions: list[np.ndarray] = []

    # A large collection may be produced by independent GPU workers under
    # shard directories.  Accept both the original flat layout and nested
    # episode directories, while deliberately excluding rejected_ep_*.
    for episode in sorted(demo_dir.rglob("episode_*")):
        if not episode.is_dir():
            continue
        try:
            meta = json.loads((episode / "meta.json").read_text(encoding="utf-8"))
            evidence = meta.get("expert_evidence") or {}
            if not (meta.get("success") and meta.get("ready_for_training")):
                raise ValueError("episode is not marked successful/training-ready")
            if not (evidence.get("success") and evidence.get("physical_execution")):
                raise ValueError("Expert physical-success evidence is missing")
            if float(evidence.get("lift_height_m", 0.0)) < 0.10:
                raise ValueError("verified lift is below 10 cm")
            if int(evidence.get("stable_hold_frames", 0)) < 30:
                raise ValueError("stable hold is shorter than 30 frames")
            if meta.get("camera_mode") != "ego_centric_head":
                raise ValueError("training observation is not the ego-centric head camera")
            if int(meta.get("capture_hz", 0)) != EXPECTED_CAPTURE_HZ:
                raise ValueError(
                    f"capture rate is not {EXPECTED_CAPTURE_HZ} Hz"
                )
            intrinsics = meta.get("camera_intrinsics") or {}
            if not math.isclose(
                float(intrinsics.get("near_clip_m", -1.0)),
                EXPECTED_NEAR_CLIP_M,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"near clipping plane is not {EXPECTED_NEAR_CLIP_M} m"
                )
            if not math.isclose(
                float(intrinsics.get("far_clip_m", -1.0)),
                EXPECTED_FAR_CLIP_M,
                rel_tol=0.0,
                abs_tol=1e-3,
            ):
                raise ValueError(
                    f"far clipping plane is not {EXPECTED_FAR_CLIP_M} m"
                )

            step_dirs = sorted(episode.glob("step_*"))
            if len(step_dirs) < ACTION_CHUNK:
                raise ValueError("trajectory is shorter than one action chunk")
            expected_names = [f"step_{index:04d}" for index in range(len(step_dirs))]
            if [step.name for step in step_dirs] != expected_names:
                raise ValueError("step indices are not contiguous from step_0000")

            episode_actions: list[np.ndarray] = []
            episode_images: list[str] = []
            max_black = 0.0
            max_bottom_black = 0.0
            for step_dir in step_dirs:
                image_path = step_dir / "image.png"
                action_path = step_dir / "action.json"
                if not image_path.is_file() or not action_path.is_file():
                    raise ValueError(f"incomplete step {step_dir.name}")
                with Image.open(image_path) as image:
                    if image.size != EXPECTED_IMAGE_SIZE:
                        raise ValueError(
                            f"unexpected RGB size at {step_dir.name}: "
                            f"{image.size} != {EXPECTED_IMAGE_SIZE}"
                        )
                    rgb = np.asarray(image.convert("RGB"))
                if rgb.size == 0 or float(rgb.std()) < 2.0:
                    raise ValueError(f"invalid RGB at {step_dir.name}")
                black, bottom_black, large_black = _black_metrics(rgb)
                max_black = max(max_black, black)
                max_bottom_black = max(max_bottom_black, bottom_black)
                if large_black:
                    raise ValueError(
                        f"large black region at {step_dir.name} "
                        f"({black:.3f}/{bottom_black:.3f})"
                    )
                episode_images.append(str(image_path.relative_to(demo_dir)))
                episode_actions.append(_load_action(action_path))

            actions = np.stack(episode_actions)
            all_actions.extend(episode_actions)
            accepted.append(
                {
                    "episode": str(episode.relative_to(demo_dir)),
                    "instruction": str(meta.get("instruction") or "pick up the object"),
                    "object_id": str(meta.get("object_id") or ""),
                    "images": episode_images,
                    "actions": actions.tolist(),
                    "frames": len(step_dirs),
                    "samples": len(step_dirs) - ACTION_CHUNK + 1,
                    "max_black_fraction": max_black,
                    "max_bottom_black_fraction": max_bottom_black,
                    "lift_height_m": float(evidence["lift_height_m"]),
                    "stable_hold_frames": int(evidence["stable_hold_frames"]),
                }
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            rejected.append({"episode": episode.name, "reason": str(exc)})

    if not accepted:
        details = "; ".join(f"{x['episode']}: {x['reason']}" for x in rejected)
        raise RuntimeError(f"no valid Family Home Expert episodes: {details}")

    action_array = np.stack(all_actions).astype(np.float32)
    q01 = np.quantile(action_array, 0.01, axis=0).astype(np.float32)
    q99 = np.quantile(action_array, 0.99, axis=0).astype(np.float32)
    stats = {
        "mean": action_array.mean(axis=0).tolist(),
        "std": action_array.std(axis=0).tolist(),
        "min": action_array.min(axis=0).tolist(),
        "max": action_array.max(axis=0).tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "mask": [True] * len(ACTION_LABELS),
    }
    payload = {
        "schema_version": 1,
        "dataset_name": DATASET_NAME,
        "demo_dir": str(demo_dir),
        "action_labels": list(ACTION_LABELS),
        "action_chunk": ACTION_CHUNK,
        "normalization": "bounds_q99",
        "action_frame": ACTION_FRAME,
        "unnorm_key": UNNORM_KEY,
        "gripper_convention": GRIPPER_CONVENTION,
        "observation": {
            "camera_mode": "ego_centric_head",
            "image_size": list(EXPECTED_IMAGE_SIZE),
            "capture_hz": EXPECTED_CAPTURE_HZ,
            "near_clip_m": EXPECTED_NEAR_CLIP_M,
            "far_clip_m": EXPECTED_FAR_CLIP_M,
            "third_person_used_for_training": False,
        },
        "episode_count": len(accepted),
        "frame_count": int(sum(x["frames"] for x in accepted)),
        "sample_count": int(sum(x["samples"] for x in accepted)),
        "episodes": accepted,
        "rejected": rejected,
        "dataset_statistics": {
            DATASET_NAME: {
                "action": stats,
                "num_transitions": int(action_array.shape[0]),
                "num_trajectories": len(accepted),
            }
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


class G1DOFTDataset:
    """Map-style dataset compatible with OFT's action collator."""

    def __init__(self, manifest_path: Path, batch_transform: Any, repeat: int = 1):
        import torch

        self._torch = torch
        self.manifest_path = Path(manifest_path).resolve()
        self.payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.demo_dir = Path(self.payload["demo_dir"])
        self.batch_transform = batch_transform
        self.dataset_statistics = self.payload["dataset_statistics"]
        self.repeat = max(1, int(repeat))
        self.records: list[tuple[dict[str, Any], int]] = []
        for episode in self.payload["episodes"]:
            for start in range(int(episode["samples"])):
                self.records.append((episode, start))
        if not self.records:
            raise RuntimeError("manifest contains no action-chunk samples")

        stats = self.dataset_statistics[DATASET_NAME]["action"]
        self.low = np.asarray(stats["q01"], dtype=np.float32)
        self.high = np.asarray(stats["q99"], dtype=np.float32)
        self.constant_mask = np.asarray(stats["min"], dtype=np.float32) == np.asarray(
            stats["max"], dtype=np.float32
        )

    def __len__(self) -> int:
        return len(self.records) * self.repeat

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode, start = self.records[index % len(self.records)]
        image_path = self.demo_dir / episode["images"][start]
        image = Image.open(image_path).convert("RGB")
        raw = np.asarray(
            episode["actions"][start : start + ACTION_CHUNK], dtype=np.float32
        )
        span = self.high - self.low
        normalized = np.clip(2.0 * (raw - self.low) / (span + 1e-8) - 1.0, -1.0, 1.0)
        normalized[:, self.constant_mask] = 0.0

        action_tokenizer = self.batch_transform.action_tokenizer
        prompt_builder = self.batch_transform.prompt_builder_fn("openvla")
        action_strings = action_tokenizer(normalized)
        if isinstance(action_strings, str):
            action_strings = [action_strings]
        action_chunk_string = "".join(action_strings)
        conversation = [
            {
                "from": "human",
                "value": (
                    "What action should the robot take to "
                    f"{episode['instruction'].strip().lower()}?"
                ),
            },
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        token_ids = self.batch_transform.base_tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids
        input_ids = self._torch.tensor(token_ids, dtype=self._torch.long)
        labels = input_ids.clone()
        labels[: -(len(action_chunk_string) + 1)] = IGNORE_INDEX
        pixels = self.batch_transform.image_transform(image)
        return {
            "pixel_values": pixels,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": DATASET_NAME,
            "actions": normalized.astype(np.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path("outputs/family_home_vln/grasp_demos_integrated"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/family_home_vln/openvla_oft/manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_manifest(args.demo_dir, args.manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "accepted_episodes": payload["episode_count"],
                "rejected_episodes": len(payload["rejected"]),
                "frames": payload["frame_count"],
                "action_chunk_samples": payload["sample_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
