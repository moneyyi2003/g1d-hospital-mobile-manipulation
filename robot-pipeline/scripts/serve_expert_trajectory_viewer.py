#!/usr/bin/env python3
"""Serve an inspectable RGB/action replay for G1-D Expert demonstrations.

This viewer deliberately separates OpenVLA-ready samples produced by
``collect_grasp_demos.py`` from older Expert validation trajectories.  A
successful simulation episode is useful evidence, but it is not training data
until it has per-step RGB and a 7-D action in the dataset layout.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _read_actions(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _images(directory: Path) -> list[Path]:
    return sorted(
        [*directory.glob("*.png"), *directory.glob("*.jpg"), *directory.glob("*.jpeg")]
    )


def _phase_summary(actions: list[dict]) -> list[str]:
    phases: list[str] = []
    for action in actions:
        state = str(action.get("state_after") or action.get("state") or "")
        if state and (not phases or phases[-1] != state):
            phases.append(state)
    return phases


def _quality(metadata: dict, actions: list[dict], image_count: int, *, train_ready: bool) -> dict:
    result = metadata.get("result", metadata)
    success = bool(result.get("success", metadata.get("success", False)))
    zs = []
    for action in actions:
        point = action.get("object_center_after_world_m")
        if isinstance(point, list) and len(point) >= 3:
            zs.append(float(point[2]))
    initial = result.get("object_initial_center_world_m")
    initial_z = float(initial[2]) if isinstance(initial, list) and len(initial) >= 3 else (zs[0] if zs else None)
    peak_lift = max(zs) - initial_z if zs and initial_z is not None else None
    phases = _phase_summary(actions)
    phase_ok = all(phase in phases for phase in ("move_end_effector", "grasp_object", "lift_end_effector"))
    action_7d_keys = (
        "dx_m", "dy_m", "dz_m", "droll_rad", "dpitch_rad", "dyaw_rad", "gripper",
    )
    valid_7d_actions = bool(actions) and all(
        all(key in action for key in action_7d_keys) for action in actions
    )
    checks = {
        "expert_reported_success": success,
        "rgb_present": image_count > 0,
        "trajectory_present": len(actions) > 0,
        "expected_pick_phases_present": phase_ok,
        "complete_7d_actions_present": valid_7d_actions,
        "observed_peak_lift_m": peak_lift,
        "openvla_training_layout": train_ready,
    }
    checks["ready_for_training"] = bool(
        success
        and image_count > 0
        and len(actions) > 0
        and train_ready
        and (valid_7d_actions if train_ready else phase_ok)
    )
    return checks


def _catalog(root: Path) -> tuple[dict[str, dict], list[dict]]:
    """Return file-safe episode records and compact API payloads."""

    records: dict[str, dict] = {}
    payload: list[dict] = []

    # OpenVLA-ready collection layout.
    # Do not recursively walk the whole outputs tree on every browser poll:
    # it can contain many large unrelated Isaac artifacts.  These are the two
    # supported locations of this project's training collection.
    for dataset_dir in (root / "grasp_demos", root / "family_home_vln" / "grasp_demos"):
        if not dataset_dir.is_dir():
            continue
        manifest = _read_json(dataset_dir / "dataset.json")
        for ep_dir in sorted(dataset_dir.glob("episode_*")):
            metadata = _read_json(ep_dir / "meta.json")
            step_dirs = sorted(ep_dir.glob("step_*"))
            images = [step / "image.png" for step in step_dirs if (step / "image.png").is_file()]
            actions = [_read_json(step / "action.json") for step in step_dirs]
            episode_id = f"dataset:{dataset_dir.name}:{ep_dir.name}"
            quality = _quality(metadata, actions, len(images), train_ready=True)
            records[episode_id] = {"images": images, "actions": actions}
            payload.append({
                "id": episode_id,
                "source": "OpenVLA training collection",
                "path": str(ep_dir),
                "metadata": metadata,
                "manifest_summary": manifest.get("summary", {}),
                "image_count": len(images),
                "action_count": len(actions),
                "phases": _phase_summary(actions),
                "quality": quality,
            })

    # Expert validation layout: useful for reviewing control, but do not mark
    # it train-ready because it lacks the per-step OpenVLA action/image pairs.
    validation_sources = {
        "g1d_cup_expert": "旧 OperationTable 测试场景（非家庭场景，不可训练）",
        "g1d_expert_validation": "旧 Expert 验证场景（非家庭场景，不可训练）",
        "family_home_vln/g1d_expert": "家庭场景执行日志（无 RGB，不可训练）",
    }
    for source_name, source_label in validation_sources.items():
        source_dir = root / source_name
        for ep_dir in sorted(source_dir.glob("episode_*")):
            metadata = _read_json(ep_dir / "metadata.json")
            actions = _read_actions(ep_dir / "action.jsonl")
            images = _images(ep_dir / "episode_rgb")
            episode_id = f"validation:{source_name.replace('/', '_')}:{ep_dir.name}"
            quality = _quality(metadata, actions, len(images), train_ready=False)
            records[episode_id] = {"images": images, "actions": actions}
            payload.append({
                "id": episode_id,
                "source": source_label,
                "path": str(ep_dir),
                "metadata": metadata,
                "image_count": len(images),
                "action_count": len(actions),
                "phases": _phase_summary(actions),
                "quality": quality,
            })
    return records, payload


PAGE = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G1-D Expert 数据回放</title>
<style>body{margin:24px auto;max-width:1100px;padding:0 16px;font:15px system-ui;background:#10151b;color:#e9eef4}select,input,button{font:inherit;padding:8px}main{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);gap:18px}img{width:100%;min-height:350px;object-fit:contain;background:#05080b}pre{white-space:pre-wrap;background:#19232e;padding:12px;border-radius:7px;overflow:auto}.good{color:#70d6a6}.bad{color:#ff8f8f}.warn{color:#f6c85f}.empty{padding:64px;text-align:center;color:#aab9c9}@media(max-width:760px){main{grid-template-columns:1fr}img{min-height:220px}}</style>
<h1>G1-D Expert 数据回放</h1><p id="note">加载中…</p><label>轨迹 <select id="episode"></select></label><p><button id="prev">上一帧</button> <input id="frame" type="range" min="0" value="0"> <button id="next">下一帧</button> <span id="label"></span></p><main><section id="visual"></section><aside><h2>质量检查</h2><pre id="quality"></pre><h2>当前动作 / 状态</h2><pre id="action"></pre></aside></main>
<script>let episodes=[],current=null,index=0;const el=id=>document.getElementById(id);const clamp=()=>Math.max(0,Math.min(index,Math.max(0,(current?.image_count||1)-1)));function render(){if(!current)return;index=clamp();el('frame').max=Math.max(0,current.image_count-1);el('frame').value=index;el('label').textContent=`${index+1} / ${current.image_count||0}`;el('quality').textContent=JSON.stringify(current.quality,null,2);let v=el('visual');if(current.image_count){v.innerHTML=`<img alt="Expert RGB frame" src="/frame/${encodeURIComponent(current.id)}/${index}">`}else{v.innerHTML='<div class="empty">这条轨迹没有 RGB。它只能用于控制日志复核，不能作为视觉训练样本。</div>'}let a=current.actions?.[Math.min(index,(current.actions?.length||1)-1)]||{};el('action').textContent=JSON.stringify(a,null,2)}async function loadEpisode(){current=episodes.find(x=>x.id===el('episode').value);if(!current)return;let r=await fetch('/api/episode/'+encodeURIComponent(current.id));let d=await r.json();current.actions=d.actions||[];index=0;render()}fetch('/api/episodes').then(r=>r.json()).then(d=>{episodes=d.episodes;el('note').textContent=d.note;el('episode').innerHTML=episodes.map(x=>`<option value="${x.id}">${x.source} — ${x.id} (${x.image_count} RGB / ${x.action_count} actions)</option>`).join('');el('episode').onchange=loadEpisode;if(episodes.length)loadEpisode();else el('visual').innerHTML='<div class="empty">没有找到轨迹。</div>'});el('prev').onclick=()=>{index--;render()};el('next').onclick=()=>{index++;render()};el('frame').oninput=e=>{index=+e.target.value;render()};</script></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6014)
    args = parser.parse_args()
    root = args.root.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            return

        def send_json(self, data: dict) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            records, episodes = _catalog(root)
            path = urlparse(self.path).path
            if path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            if path == "/api/episodes":
                self.send_json({"episodes": episodes, "note": "仅 ready_for_training=true 的家庭场景 RGB+7D 数据可训练；旧 OperationTable 和验证轨迹不可训练。"}); return
            if path.startswith("/api/episode/"):
                episode_id = unquote(path.removeprefix("/api/episode/")); record = records.get(episode_id)
                if record is None: self.send_error(HTTPStatus.NOT_FOUND); return
                self.send_json({"actions": record["actions"]}); return
            if path.startswith("/frame/"):
                parts = path.split("/")
                if len(parts) != 4: self.send_error(HTTPStatus.NOT_FOUND); return
                episode_id, raw_index = unquote(parts[2]), parts[3]
                record = records.get(episode_id)
                try: image = record["images"][int(raw_index)] if record else None
                except (IndexError, ValueError): image = None
                if image is None or not image.is_file(): self.send_error(HTTPStatus.NOT_FOUND); return
                body = image.read_bytes(); self.send_response(HTTPStatus.OK); self.send_header("Content-Type", mimetypes.guess_type(image.name)[0] or "image/png"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            self.send_error(HTTPStatus.NOT_FOUND)

    print(f"Expert trajectory viewer: http://{args.host}:{args.port}/", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
