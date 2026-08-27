#!/usr/bin/env python3
"""Summarize a collector run: per-episode outcome + grasp-phase telemetry.

Usage: python3 scripts/_analyze_collection.py <output_dir>
Reads expert_run_NNNN/episode_0000/action.jsonl records.
"""
import json
import sys
from collections import Counter
from pathlib import Path


def load_records(episode_dir: Path) -> list[dict]:
    path = episode_dir / "action.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    runs = sorted(
        [p for p in root.glob("expert_run_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[2]),
    )
    if not runs:
        print("no expert_run_* dirs found")
        return 1
    print(f"{len(runs)} runs in {root}\n")

    n_done = 0
    for run in runs:
        recs = load_records(run / "episode_0000")
        if not recs:
            print(f"{run.name}: no telemetry")
            continue
        final_state = recs[-1].get("state_after", recs[-1].get("state", "?"))
        states = Counter(r.get("state") for r in recs)
        close = [r for r in recs if r.get("state") == "grasp_object"]
        verify = [r for r in recs if r.get("state") == "verify_grasp"]

        # ---- close phase summary ----
        gap = [round(c["controller"].get("finger_surface_gap_m", float("nan")) * 1000, 1) for c in close]
        cfrac = [c["controller"].get("scheduled_grasp_close_fraction", 0.0) for c in close]
        released = any(c["controller"].get("object_dynamics_released_during_close") for c in close)
        # middle_0 joint rad
        m0 = []
        for c in close:
            names = c.get("hand_joint_names", [])
            vals = c.get("hand_joint_position_rad", [])
            if "right_hand_middle_0_joint" in names:
                m0.append(round(vals[names.index("right_hand_middle_0_joint")], 3))
        # end-of-close contact
        last_close = close[-1]["controller"] if close else {}
        force = last_close.get("grasp_contact_force_n")
        sens = last_close.get("grasp_contact_sensed")
        csteps = last_close.get("grasp_contact_steps", 0)
        drift = last_close.get("grasp_object_xy_drift_m")
        cerr = last_close.get("grasp_control_error_m")

        # ---- verify phase summary ----
        if verify:
            v = verify[-1]["controller"]
            lift = round(v.get("verify_grasp_observed_lift_m", 0.0) * 1000, 1)
            vdrift = round(v.get("verify_grasp_relative_drift_m", 0.0) * 1000, 1)
            vc = v.get("verify_grasp_contact_sensed")
            vstab = v.get("verify_grasp_stable_steps")
        else:
            lift = vdrift = vc = vstab = None

        gap_s = "".join(
            f"{g:.0f}," for g in (gap or [])
        )[:60] or "-"
        m0_s = "".join(f"{a:.2f}," for a in m0)[:60] or "-"
        print(
            f"{run.name}: end={final_state:7s} "
            f"states={states.get('grasp_object',0)}g/{states.get('verify_grasp',0)}v "
            f"released={released} "
            f"gap_mm=[{gap_s}] m0_rad=[{m0_s}] "
            f"force={force} sens={sens} csteps={csteps} drift={drift} "
            f"cerr={cerr} | verify lift={lift}mm drift={vdrift}mm "
            f"contact={vc} stable={vstab}"
        )
        if final_state == "done":
            n_done += 1
    print(f"\n{n_done}/{len(runs)} ended in 'done'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
