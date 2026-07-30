"""Validate OpenVLA actions before they reach any G1-D controller."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable


ACTION_LABELS = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "gripper",
)
G1D_RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


class ActionContractError(ValueError):
    """Raised when a model result is not a finite seven-dimensional action."""


@dataclass(frozen=True)
class OpenVlaAction:
    """One dataset-decoded OpenVLA end-effector-delta action."""

    values: tuple[float, float, float, float, float, float, float]
    unnorm_key: str

    @classmethod
    def from_values(
        cls,
        values: Iterable[Any],
        *,
        unnorm_key: str,
    ) -> "OpenVlaAction":
        try:
            normalized = tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ActionContractError("OpenVLA action must contain numbers") from exc
        if len(normalized) != len(ACTION_LABELS):
            raise ActionContractError(
                f"OpenVLA action must have 7 values, got {len(normalized)}"
            )
        if not all(math.isfinite(value) for value in normalized):
            raise ActionContractError("OpenVLA action contains a non-finite value")
        if not str(unnorm_key).strip():
            raise ActionContractError("unnorm_key cannot be empty")
        return cls(normalized, str(unnorm_key).strip())  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "labels": list(ACTION_LABELS),
            "unnorm_key": self.unnorm_key,
            "semantics": (
                "dataset-specific decoded end-effector delta "
                "(x,y,z,roll,pitch,yaw,gripper)"
            ),
        }


def build_g1d_right_arm_handoff(action: OpenVlaAction) -> dict[str, Any]:
    """Create a fail-closed handoff instead of treating action values as joints."""

    return {
        "schema_version": 1,
        "source": "openvla",
        "target": {
            "robot": "g1_d",
            "side": "right",
            "arm_joint_count": 7,
            "joint_order": list(G1D_RIGHT_ARM_JOINTS),
        },
        "model_action": action.to_dict(),
        "joint_command": None,
        "execution_permitted": False,
        "blocked_reasons": [
            "checkpoint_not_finetuned_or_calibrated_for_g1_d",
            "openvla_action_frame_not_mapped_to_g1_d_base_or_tool_frame",
            "target_visibility_not_revalidated_in_final_openvla_frame",
            "collision_checked_right_arm_ik_not_connected",
            "g1_d_multifinger_hand_mapping_not_connected",
        ],
        "required_next_adapter": {
            "input": (
                "calibrated Cartesian delta in a declared camera/base/tool frame"
            ),
            "output": (
                "joint-limit and velocity-limited targets for "
                "right_shoulder_pitch_joint through right_wrist_yaw_joint"
            ),
            "safety_gates": [
                "base_stopped",
                "fresh_rgb_and_tf",
                "target_visible_and_not_edge_clipped_in_current_rgb",
                "right_arm_ik_feasible",
                "self_and_scene_collision_free",
                "operator_enable_for_real_hardware",
            ],
        },
    }


__all__ = [
    "ACTION_LABELS",
    "ActionContractError",
    "G1D_RIGHT_ARM_JOINTS",
    "OpenVlaAction",
    "build_g1d_right_arm_handoff",
]
