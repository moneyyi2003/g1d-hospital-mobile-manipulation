"""Bridge module: wraps MaChuanhao's G1-D Expert DLS IK controller
for family-home VLN pick operations.

Importable without Isaac Sim — expert imports are deferred until
:func:`run_expert_pick` is called inside the running simulation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent
EXPERT_PACKAGE = ROOT / "g1d-expert-MaChuanhao" / "g1d-expert"
DEFAULT_EXPERT_CONFIG = EXPERT_PACKAGE / "contracts" / "cup_expert_config.json"
DEFAULT_CONTROL_CONFIG = EXPERT_PACKAGE / "contracts" / "g1d_control_parameters.json"


def _read_json(path: Path) -> dict[str, Any]:
    """Read and validate a JSON file, raising with a clear path on failure."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read expert config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expert config root must be an object: {path}")
    return value


def _ensure_expert_imports():
    """Add the expert package to sys.path so its modules are importable."""
    import sys

    expert_root = str(EXPERT_PACKAGE.resolve())
    if expert_root not in sys.path:
        sys.path.insert(0, expert_root)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_expert_pick(
    *,
    robot: Any,
    stage: Any,
    target_prim_path: str,
    table_top_prim_path: str,
    robot_prim_path: str,
    palm_prim_path: str,
    arm: str = "right",
    advance_fn: Callable[[int], None],
    output_dir: Path,
    config_overrides: dict[str, Any] | None = None,
    control_step_observer: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the expert DLS-IK pick-lift-drop pipeline in-process.

    Parameters
    ----------
    robot:
        The G1-D ``WheeledRobot`` articulation (already loaded and simulated).
    stage:
        The live USD stage (``omni.usd.get_context().get_stage()``).
    target_prim_path:
        USD prim path of the dynamic rigid-body object to grasp, e.g.
        ``/World/OperationTable/CoffeeCup``.
    table_top_prim_path:
        USD prim whose world bounding-box *max.z* defines the table surface
        the fingers must clear, e.g. ``/World/FamilyHome/dining_table``.
    robot_prim_path:
        Root prim of the robot articulation, e.g. ``/World/G1_D``.
    palm_prim_path:
        Full USD path to the right-hand palm link prim.
    arm:
        ``"right"`` or ``"left"`` — the arm to control.
    advance_fn:
        Per-control-step callback ``advance_fn(steps: int)``.  The bridge
        calls this once per expert state-machine iteration.  It must step
        physics, pin the mobile base, and apply joint holds.
    output_dir:
        Directory for expert episode artifacts (action.jsonl, metadata.json,
        collection_summary.json).
    config_overrides:
        Optional merged overrides applied on top of the validated cup-expert
        config (e.g. ``{"expert": {"grasp_point_z_offset_m": 0.04}}``).

    Returns
    -------
    dict
        Evidence dict shaped like ``_execute_sim_pick``'s return:

        * ``success`` — whether the expert reached DONE
        * ``reason`` — short label (``"expert_pick_lift_drop_completed"`` or
          ``"expert_failed: <reason>"``)
        * ``physical_execution`` — always ``True``
        * ``state`` — final expert state name
        * ``steps`` — total control steps
        * ``lift_height_m`` — peak observed object lift
        * ``stable_hold_frames`` — consecutive verified physical hold frames
        * ``object_prim_path``, ``table_top_prim_path`` — validated prim paths
        * ``episode_dir`` — path to the expert output directory
        * ``summary`` — the ``collection_summary.json`` payload
    """
    _ensure_expert_imports()

    # Deferred imports — only valid after SimulationApp has started.
    from expert.g1d import (
        G1DArmController,
        JointPositionHold,
        configure_g1d_drives,
        get_dof_positions,
        initial_positions_from_config,
        pin_root,
        read_root_pose,
        set_root_pose,
    )
    from expert.pick_lift_drop_expert import PickLiftDropExpert
    from expert.task import PickLiftDropTask

    # ---- load config -------------------------------------------------------
    expert_config = _read_json(DEFAULT_EXPERT_CONFIG)
    control_config = _read_json(DEFAULT_CONTROL_CONFIG)

    # Apply runtime overrides
    expert_config["robot_prim_path"] = robot_prim_path
    expert_config["palm_prim_paths"] = {arm: palm_prim_path}
    expert_config["operation_table_top_prim_path"] = table_top_prim_path
    # Drop the unused episode camera section so the expert doesn't try to
    # create a recorder we don't need.
    expert_config.pop("episode_camera", None)
    expert_config.pop("webrtc", None)

    if config_overrides:
        _deep_merge(expert_config, config_overrides)
    drive_overrides = expert_config.get("drive_overrides", {})
    if drive_overrides:
        if not isinstance(drive_overrides, dict):
            raise ValueError("drive_overrides must be an object")
        _deep_merge(control_config, drive_overrides)

    # ---- validate prims ---------------------------------------------------
    missing = []
    for label, prim_path in [
        ("robot", robot_prim_path),
        ("palm", palm_prim_path),
        ("table_top", table_top_prim_path),
        ("target", target_prim_path),
    ]:
        if not stage.GetPrimAtPath(prim_path).IsValid():
            missing.append(f"{label} ({prim_path})")
    if missing:
        return {
            "success": False,
            "reason": f"expert_init_failed: missing prims: {'; '.join(missing)}",
            "physical_execution": False,
        }

    # ---- ensure xformOp:orient on target and table prims -------------------
    # Family-home dynamic objects may only define ``xformOp:translate`` (or no
    # explicit xformOp at all).  The Isaac Sim ``XformPrim`` wrapper (used by the
    # expert at pick_lift_drop_expert.py:753/774) calls ``set_local_poses()``
    # which *asserts* that ``xformOp:orient`` exists as a quaternion attribute.
    # ``XformCommonAPI`` stubs ``xformOp:rotateXYZ`` (Euler) instead, so we must
    # directly author a ``xformOp:orient`` quaternion attribute via the lower-level
    # ``UsdGeom.Xformable`` API.
    from pxr import Gf, UsdGeom, UsdPhysics

    for _label, _path in [
        ("target", target_prim_path),
        ("table_top", table_top_prim_path),
    ]:
        _prim = stage.GetPrimAtPath(_path)
        _orient_attr = _prim.GetAttribute("xformOp:orient")
        if not _orient_attr.IsValid():
            _xform = UsdGeom.Xformable(_prim)
            _orient_op = _xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            _orient_op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
            print(
                f"[ExpertBridge] added xformOp:orient on {_label} prim {_path}",
                flush=True,
            )

    # ---- reposition target onto table surface ------------------------------
    # During the VLN navigation phase the dynamic rigid-body cup may have fallen
    # to the floor.  Teleport it back onto the table so the expert's end-effector
    # can reach it.
    #
    # The family-home dining-table asset is a complex reference whose
    # ``world_bbox`` may not reflect the actual tabletop surface.  Fall back to
    # a reasonable dining-table height (~0.75 m above the floor) when the
    # computed bounding box is implausibly low.
    from pxr import Usd

    _bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    _target_prim = stage.GetPrimAtPath(target_prim_path)
    _table_prim = stage.GetPrimAtPath(table_top_prim_path)

    # A previous failed attempt may already have authored the simulation
    # grasp joint.  Remove it before repositioning the cup so every retry
    # starts from a clean physics state.
    _stale_constraint_path = "/World/G1DExpertGraspConstraint"
    if stage.GetPrimAtPath(_stale_constraint_path).IsValid():
        stage.RemovePrim(_stale_constraint_path)
        print("[ExpertBridge] removed stale grasp constraint", flush=True)

    _obj_box = _bbox_cache.ComputeWorldBound(_target_prim).ComputeAlignedBox()
    _obj_half_h = float((_obj_box.GetMax()[2] - _obj_box.GetMin()[2]) / 2.0)
    _obj_bottom_z = float(_obj_box.GetMin()[2])

    _tbl_box = _bbox_cache.ComputeWorldBound(_table_prim).ComputeAlignedBox()
    _tbl_surface_z = float(_tbl_box.GetMax()[2])
    # The family-home floor is below world zero, so a valid tabletop can have
    # a slightly negative world Z.  Clamp only against an explicitly supplied
    # world-frame floor-relative minimum; never assume world Z == floor Z.
    _minimum_table_surface_world_z = float(
        expert_config.get("minimum_table_surface_world_z", float("-inf"))
    )
    _tbl_surface_z = max(
        _tbl_surface_z, _minimum_table_surface_world_z,
    )

    _target_body = UsdPhysics.RigidBodyAPI(_target_prim)
    _target_body.GetKinematicEnabledAttr().Set(True)
    _target_body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    _target_body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    _obj_translate = _target_prim.GetAttribute("xformOp:translate")
    _current_pos = _obj_translate.Get()
    # The rigid body's authored origin is not necessarily its bbox centre.
    # Item05's visual/collision bbox is offset upward from the root by about
    # 3.1 cm, so assigning ``table + half_height`` made the cup float 4.1 cm
    # above the tabletop.  Move the root by the measured bottom-plane error
    # instead; this works for arbitrary asset origins.
    _desired_bottom_z = _tbl_surface_z + 0.002
    _new_z = float(_current_pos[2]) + (_desired_bottom_z - _obj_bottom_z)
    _obj_translate.Set(Gf.Vec3d(_current_pos[0], _current_pos[1], _new_z))
    print(
        f"[ExpertBridge] repositioned cup to z={_new_z:.4f} "
        f"(table_surface={_tbl_surface_z:.4f}, "
        f"previous_bottom={_obj_bottom_z:.4f}, cup_half_h={_obj_half_h:.4f})",
        flush=True,
    )
    # Keep the cup pinned while the expert approaches. The expert switches it
    # back to dynamic atomically when the grasp constraint is authored; doing
    # so here lets an unsupported asset fall before the fingers arrive.
    advance_fn(10)

    # ---- build task -------------------------------------------------------
    task = PickLiftDropTask(
        arm=arm,
        target_prim_path=target_prim_path,
        lift_height_m=0.20,
        lower_distance_m=0.08,
        collection_rounds=1,
        schema_version=1,
    )

    # ---- configure drives -------------------------------------------------
    configure_g1d_drives(robot, control_config)

    # ---- baseline positions & base pose -----------------------------------
    # Merge initial joints from config; left arm defaults to down pose.
    raw_initial = dict(expert_config.get("initial_joint_positions_rad", {}))
    left_down = dict(expert_config.get("left_arm_down_joint_positions_rad", {}))
    allowed_left = {
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    }
    raw_initial.update(
        {k: float(v) for k, v in left_down.items() if k in allowed_left}
    )
    baseline_q = initial_positions_from_config(robot, raw_initial)
    base_pose = read_root_pose(robot)
    set_root_pose(robot, base_pose)

    joint_hold = JointPositionHold(robot, baseline_q)
    joint_hold.teleport()

    # ---- build the wrapped advance function --------------------------------
    # The expert calls advance_fn(steps) once per state-machine iteration.
    # We wrap the caller-provided advance_fn to also pin the base pose and
    # apply the expert's joint hold on every physics step.
    _user_advance = advance_fn

    _observe_control_steps = False

    def _wrapped_advance(steps: int) -> None:
        for _ in range(max(int(steps), 0)):
            pin_root(robot, base_pose)
            joint_hold.apply()
            if _observe_control_steps and control_step_observer is not None:
                control_step_observer("before")
            _user_advance(1)
            if _observe_control_steps and control_step_observer is not None:
                control_step_observer("after")

    # ---- controller + calibration ----------------------------------------
    print("[ExpertBridge] creating G1DArmController", flush=True)
    controller = G1DArmController(
        robot,
        stage,
        arm,
        robot_prim_path,
        palm_prim_path,
        control=expert_config.get("ik", {}),
        advance_fn=_wrapped_advance,
        joint_hold=joint_hold,
    )

    print("[ExpertBridge] calibrating grasp geometry", flush=True)
    calibration = controller.calibrate_grasp_geometry(
        base_pin_fn=lambda: pin_root(robot, base_pose)
    )
    print(
        f"[ExpertBridge] grasp geometry calibrated: "
        f"middle/thumb distance={calibration['closed_middle_thumb_distance_m']:.4f} m",
        flush=True,
    )

    # Reset to baseline after calibration
    joint_hold.set_targets(baseline_q)
    joint_hold.teleport()
    _wrapped_advance(10)

    # ---- create expert + run ----------------------------------------------
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[ExpertBridge] creating PickLiftDropExpert", flush=True)
    expert = PickLiftDropExpert(
        task=task,
        robot=robot,
        stage=stage,
        controller=controller,
        baseline_joint_positions=baseline_q,
        base_pose=base_pose,
        advance_fn=_wrapped_advance,
        config=expert_config,
        output_dir=output_dir,
        camera_recorder=None,
    )

    print("[ExpertBridge] running expert collection", flush=True)
    retain_grasp = bool(
        expert_config.get("expert", {}).get("retain_grasp_after_lift", False)
    )
    if retain_grasp:
        # run_collection() always resets after its final episode, which would
        # detach the cup.  The VLN carry mission deliberately preserves the
        # successful final state until return navigation finishes.
        expert.reset_episode()
        _observe_control_steps = True
        try:
            results = [expert.run_episode(0)]
        finally:
            _observe_control_steps = False
    else:
        _observe_control_steps = True
        try:
            results = expert.run_collection()
        finally:
            _observe_control_steps = False

    # ---- build evidence ---------------------------------------------------
    if not results:
        return {
            "success": False,
            "reason": "expert_returned_no_results",
            "physical_execution": True,
        }

    result = results[0]  # collection_rounds=1 → single episode

    def _prim_origin_world(prim_path: str) -> np.ndarray:
        matrix = UsdGeom.Xformable(
            stage.GetPrimAtPath(prim_path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = matrix.ExtractTranslation()
        return np.asarray(
            [translation[0], translation[1], translation[2]],
            dtype=np.float64,
        )

    def _bbox_center_world(prim_path: str) -> np.ndarray:
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        aligned = cache.ComputeWorldBound(
            stage.GetPrimAtPath(prim_path)
        ).ComputeAlignedBox()
        minimum = aligned.GetMin()
        maximum = aligned.GetMax()
        return np.asarray(
            [
                0.5 * (minimum[0] + maximum[0]),
                0.5 * (minimum[1] + maximum[1]),
                0.5 * (minimum[2] + maximum[2]),
            ],
            dtype=np.float64,
        )

    stable_hold_frames = 0
    hold_contact_frames = 0
    hold_max_palm_object_drift_m = 0.0
    hold_min_lift_height_m = float("inf")
    physical_hold_verified = False
    if result.success and retain_grasp:
        # Observe more than the required window so a brief solver transient
        # cannot hide a later slip.  A valid window requires two-finger
        # contact, retained lift, bounded palm/object drift, and no authored
        # grasp constraint.
        hold_origin_object = _bbox_center_world(target_prim_path)
        hold_origin_palm = _prim_origin_world(palm_prim_path)
        consecutive_hold = 0
        minimum_hold_frames = 30
        for _ in range(45):
            _wrapped_advance(1)
            object_now = _bbox_center_world(target_prim_path)
            palm_now = _prim_origin_world(palm_prim_path)
            initial_center = np.asarray(
                result.object_initial_center_world_m, dtype=np.float64
            )
            lift_now = float(object_now[2] - initial_center[2])
            hold_min_lift_height_m = min(hold_min_lift_height_m, lift_now)
            relative_now = object_now - palm_now
            relative_origin = hold_origin_object - hold_origin_palm
            relative_drift = float(np.linalg.norm(relative_now - relative_origin))
            hold_max_palm_object_drift_m = max(
                hold_max_palm_object_drift_m, relative_drift
            )
            contact_ok = False
            if expert.contact_sensor is not None:
                contact_ok, _ = expert.contact_sensor.grasp_contact()
            if contact_ok:
                hold_contact_frames += 1
            constraint_absent = not stage.GetPrimAtPath(
                expert.grasp_constraint_path
            ).IsValid()
            frame_ok = (
                contact_ok
                and constraint_absent
                and lift_now >= 0.10
                and relative_drift <= 0.025
            )
            consecutive_hold = consecutive_hold + 1 if frame_ok else 0
            stable_hold_frames = max(stable_hold_frames, consecutive_hold)
        physical_hold_verified = stable_hold_frames >= minimum_hold_frames
    if not np.isfinite(hold_min_lift_height_m):
        hold_min_lift_height_m = 0.0

    final_palm_object_distance_m = float(
        np.linalg.norm(
            _prim_origin_world(palm_prim_path)
            - _prim_origin_world(target_prim_path)
        )
    )

    fixed_joint_present = stage.GetPrimAtPath(
        expert.grasp_constraint_path
    ).IsValid()
    physical_success = bool(
        result.success
        and (not retain_grasp or physical_hold_verified)
        and not fixed_joint_present
    )
    # In carry mode a failed physical hold must reset the episode so the next
    # attempt starts from a deterministic state.
    if retain_grasp and not physical_success:
        expert.reset_episode()
    summary = {
        "schema_version": 1,
        "task": task.to_dict(),
        "grasp_geometry_calibration": calibration,
        "episodes": [r.__dict__ for r in results],
        "success": all(r.success for r in results),
    }
    summary_path = output_dir / "collection_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Convert expert result to the evidence dict shape expected by the VLN
    # session's VERIFY skill and SkillResult mapping.
    fixed_joint_used = bool(
        expert_config.get("expert", {}).get("use_fixed_joint_attach", False)
    )
    evidence: dict[str, Any] = {
        "success": physical_success,
        "reason": (
            "expert_physical_pick_lift_hold_completed"
            if physical_success
            else (
                f"expert_failed: {result.failure_reason or 'unknown'}"
                if not result.success
                else "expert_failed: physical_hold_verification_failed"
            )
        ),
        "physical_execution": True,
        "execution_environment": "isaac_sim_only",
        "hardware_output": False,
        "grasp_mechanism": (
            "g1d_expert_dls_ik_with_fixed_joint_attach"
            if fixed_joint_used
            else "g1d_expert_dls_ik_finger_collision_and_friction_only"
        ),
        "fixed_joint_created": fixed_joint_present,
        "fixed_joint_configured": fixed_joint_used,
        "physical_hold_verified": physical_hold_verified,
        "retained_for_return_navigation": retain_grasp and physical_success,
        "state": result.state,
        "steps": result.steps,
        "failure_reason": result.failure_reason,
        "object_initial_center_world_m": result.object_initial_center_world_m,
        "object_final_center_world_m": result.object_final_center_world_m,
        "object_prim_path": target_prim_path,
        "table_top_prim_path": table_top_prim_path,
        "robot_prim_path": robot_prim_path,
        "episode_dir": str(output_dir / result.episode_id),
        "summary_path": str(summary_path),
        "grasp_geometry_calibration": calibration,
        "stable_hold_frames": stable_hold_frames,
        "hold_contact_frames": hold_contact_frames,
        "hold_max_palm_object_drift_m": hold_max_palm_object_drift_m,
        "hold_min_lift_height_m": hold_min_lift_height_m,
        "final_palm_object_distance_m": final_palm_object_distance_m,
        "lift_height_m": (
            float(result.object_final_center_world_m[2] - result.object_initial_center_world_m[2])
            if result.success
            else 0.0
        ),
        "body_selection": {
            "selection": "expert_target_prim_path",
            "object_prim_path": target_prim_path,
            "palm_prim_path": palm_prim_path,
        },
    }
    return evidence


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
