#!/usr/bin/env python3
"""Serve an interactive object-relative docking dashboard on a separate port."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "object_docking_dashboard"
DEFAULT_SCENES = ROOT / "hospital_vln/object_docking_scenes.json"
sys.path.insert(0, str(ROOT))

from hospital_vln.artifacts import HOSPITAL_START  # noqa: E402
from hospital_vln.object_docking import (  # noqa: E402
    ObjectDockingPlan,
    ObjectTarget,
    build_object_docking_plan,
    load_object_targets,
    parse_standoff,
    resolve_object,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True)
class SceneProfile:
    scene_id: str
    name: str
    runner: str
    map_yaml: Path
    places: Path
    mapping_summary: Path
    objects_path: Path
    output: Path
    objects: tuple[ObjectTarget, ...]
    mapping: dict
    map_assets: dict[str, Path]


def load_scene_profiles(path: Path) -> tuple[str, dict[str, SceneProfile]]:
    payload = _read_json(path)
    if payload.get("activation") != "explicit_object_docking_dashboard":
        raise ValueError("scene config must use explicit_object_docking_dashboard activation")
    profiles: dict[str, SceneProfile] = {}
    for value in payload.get("scenes", []):
        if value.get("status") != "enabled":
            continue
        scene_id = str(value["id"])
        if scene_id in profiles:
            raise ValueError(f"duplicate object docking scene id: {scene_id}")
        runner = str(value["runner"])
        if runner != "hospital_object_docking":
            raise ValueError(f"unsupported object docking runner: {runner}")
        map_yaml = _resolve_project_path(str(value["map"]))
        places = _resolve_project_path(str(value["places"]))
        mapping_summary = _resolve_project_path(str(value["mapping_summary"]))
        objects_path = _resolve_project_path(str(value["objects"]))
        output = _resolve_project_path(str(value["output"]))
        required = (map_yaml, places, mapping_summary, objects_path)
        missing = [str(item) for item in required if not item.is_file()]
        if missing:
            raise FileNotFoundError(
                f"object docking scene {scene_id} is incomplete: " + ", ".join(missing)
            )
        mapping = _read_json(mapping_summary)
        artifacts = mapping_summary.parent
        map_assets: dict[str, Path] = {}
        for key, raw_path in mapping.get("assets", {}).items():
            source = Path(raw_path)
            candidates = (
                source if source.is_absolute() else artifacts / source,
                artifacts / "map_preview" / f"{key}.png",
            )
            asset = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if asset is not None:
                map_assets[str(key)] = asset
        missing_layers = sorted({"rgb_pointcloud", "occupancy"} - map_assets.keys())
        if missing_layers:
            raise FileNotFoundError(
                f"object docking scene {scene_id} lacks map previews: "
                + ", ".join(missing_layers)
            )
        objects = tuple(load_object_targets(objects_path))
        profiles[scene_id] = SceneProfile(
            scene_id=scene_id,
            name=str(value["name"]),
            runner=runner,
            map_yaml=map_yaml,
            places=places,
            mapping_summary=mapping_summary,
            objects_path=objects_path,
            output=output,
            objects=objects,
            mapping=mapping,
            map_assets=map_assets,
        )
    if not profiles:
        raise ValueError("scene config has no enabled object docking scenes")
    default_scene_id = str(payload.get("default_scene_id", ""))
    if default_scene_id not in profiles:
        raise ValueError(f"default object docking scene is unavailable: {default_scene_id}")
    return default_scene_id, profiles


class ObjectDockingSession:
    def __init__(self, args: argparse.Namespace) -> None:
        self.default_scene_id, self.profiles = load_scene_profiles(
            args.scenes.resolve()
        )
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._log_stream = None
        self._started_at: float | None = None
        self._active_scene_id = self.default_scene_id
        self._last_command = ""
        self._last_plan: dict | None = None
        self._last_target: dict | None = None
        self.live_fps = int(args.live_fps)
        self.live_resolution = str(args.live_resolution)
        if not 1 <= self.live_fps <= 30:
            raise ValueError("--live-fps must be between 1 and 30")
        default_profile = self.profiles[self.default_scene_id]
        try:
            previous_plan = _read_json(default_profile.output / "docking_plan.json")
        except (OSError, ValueError, json.JSONDecodeError):
            previous_plan = None
        if isinstance(previous_plan, dict):
            self._last_plan = previous_plan
            target = previous_plan.get("target", {})
            if isinstance(target, dict) and target.get("object_id"):
                self._last_target = {
                    "id": target["object_id"],
                    "name": target.get("name", target["object_id"]),
                    "position": target.get("position", {}),
                    "interaction_face_yaw": target.get("interaction_face_yaw", 0.0),
                    "size_m": target.get("size_m", 0.0),
                }
        try:
            previous_state = _read_json(default_profile.output / "live/state.json")
        except (OSError, ValueError, json.JSONDecodeError):
            previous_state = None
        if isinstance(previous_state, dict):
            self._last_command = str(previous_state.get("command", ""))

    def _profile(self, scene_id: str | None) -> SceneProfile:
        key = (scene_id or self.default_scene_id).strip()
        try:
            return self.profiles[key]
        except KeyError as exc:
            raise ValueError(f"未配置或未启用的场景：{key}") from exc

    @staticmethod
    def _target_payload(target: ObjectTarget) -> dict:
        return {
            "id": target.object_id,
            "name": target.name,
            "aliases": list(target.aliases),
            "position": {"x": target.x, "y": target.y, "z": target.z},
            "interaction_face_yaw": target.interaction_face_yaw,
            "size_m": target.size_m,
            "examples": [
                f"请停到{target.name}前0.6米",
                f"请停到{target.name}前0.8米",
                f"请停到{target.name}前1.0米",
            ],
        }

    def config(self) -> dict:
        scenes = []
        for profile in self.profiles.values():
            map_config = dict(profile.mapping["map"])
            map_config["layers"] = [
                {
                    **layer,
                    "asset": f"/asset/map/{profile.scene_id}/{layer['id']}.png",
                }
                for layer in map_config.get("layers", [])
            ]
            scenes.append(
                {
                    "id": profile.scene_id,
                    "name": profile.name,
                    "runner": profile.runner,
                    "map": map_config,
                    "objects": [self._target_payload(item) for item in profile.objects],
                }
            )
        return {
            "schema_version": 1,
            "mode": "interactive_object_relative_docking",
            "default_scene_id": self.default_scene_id,
            "scenes": scenes,
            "camera_stream": "/stream/camera.mjpg",
            "limits": {
                "maximum_standoff_m": 2.0,
                "position_tolerance_m": 0.03,
                "yaw_tolerance_rad": 0.05,
            },
        }

    def plan(self, command: str, scene_id: str | None = None) -> tuple[SceneProfile, ObjectDockingPlan]:
        command = command.strip()
        if not command:
            raise ValueError("请输入物体级停靠指令")
        profile = self._profile(scene_id)
        target = resolve_object(command, list(profile.objects))
        standoff_m = parse_standoff(command)
        plan = build_object_docking_plan(profile.map_yaml, target, standoff_m)
        return profile, plan

    def _idle_state(self) -> dict:
        return {
            "schema_version": 1,
            "state": "idle",
            "message": "选择场景并输入物体级停靠指令。",
            "command": self._last_command,
            "scene_id": self._active_scene_id,
            "task": None,
            "frame": 0,
            "action": "idle",
            "pose": {
                "x": HOSPITAL_START.x,
                "y": HOSPITAL_START.y,
                "yaw": HOSPITAL_START.yaw,
            },
            "linear_velocity_mps": 0.0,
            "angular_velocity_rps": 0.0,
            "waypoint": 0,
            "waypoint_count": 0,
            "planned_trajectory": [],
            "trajectory": [],
            "result": None,
            "docking_plan": self._last_plan,
            "object_target": self._last_target,
            "elapsed_sec": 0.0,
            "process_running": False,
        }

    def snapshot(self) -> dict:
        with self._lock:
            process = self._process
            running = process is not None and process.poll() is None
            started_at = self._started_at
            profile = self._profile(self._active_scene_id)
            if process is not None and not running and self._log_stream is not None:
                self._log_stream.close()
                self._log_stream = None
        try:
            state = _read_json(profile.output / "live/state.json")
        except (OSError, ValueError, json.JSONDecodeError):
            state = self._idle_state()
        state["scene_id"] = profile.scene_id
        state["process_running"] = running
        state["docking_plan"] = self._last_plan
        state["object_target"] = self._last_target
        state["elapsed_sec"] = max(0.0, time.time() - started_at) if started_at else 0.0
        if process is not None and not running and state.get("state") in {"starting", "loading", "running"}:
            state["state"] = "failed"
            state["message"] = f"Isaac Sim 进程已退出，返回码 {process.returncode}。"
        return state

    @staticmethod
    def _other_kit_processes() -> list[int]:
        result = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                    "utf-8", "replace"
                )
            except OSError:
                continue
            if "/isaacsim/kit/kit " in command:
                result.append(int(entry.name))
        return result

    def submit(self, command: str, scene_id: str | None = None) -> dict:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise ValueError("已有物体停靠任务正在运行，请先等待或停止任务")
        profile, plan = self.plan(command, scene_id)
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise ValueError("已有物体停靠任务正在运行，请先等待或停止任务")
            kit_pids = self._other_kit_processes()
            if kit_pids:
                raise ValueError(
                    "检测到其他 Isaac Kit 进程："
                    + ", ".join(str(pid) for pid in kit_pids)
                    + "；请先停止其他仿真任务，避免多个 Kit 实例争用 GPU"
                )
            output = profile.output
            live_dir = output / "live"
            output.mkdir(parents=True, exist_ok=True)
            live_dir.mkdir(parents=True, exist_ok=True)
            for name in ("state.json", "camera.jpg"):
                stale = live_dir / name
                if stale.exists():
                    stale.unlink()
            plan_payload = plan.to_dict()
            target_payload = self._target_payload(plan.target)
            _atomic_json(output / "docking_plan.json", plan_payload)
            initial = self._idle_state()
            initial.update(
                {
                    "state": "starting",
                    "message": (
                        f"已解析 {plan.target.name}，目标距离 {plan.requested_standoff_m:.2f} m；"
                        "停靠位姿通过 footprint、occupancy 与可达性检查，正在启动 Isaac Sim…"
                    ),
                    "command": command.strip(),
                    "scene_id": profile.scene_id,
                    "task": plan.target.object_id,
                    "action": "starting",
                    "planned_trajectory": [{"x": x, "y": y} for x, y in plan.path],
                    "waypoint_count": max(0, len(plan.path) - 1),
                    "path_length_m": plan.path_length_m,
                    "docking_plan": plan_payload,
                    "object_target": target_payload,
                    "process_running": True,
                }
            )
            _atomic_json(live_dir / "state.json", initial)
            log_path = output / "isaac.log"
            self._log_stream = log_path.open("ab", buffering=0)
            environment = os.environ.copy()
            environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            environment.setdefault("OMNI_KIT_ALLOW_ROOT", "1")
            pose = plan.docking_pose
            target = plan.target
            argv = [
                str(ROOT / "isaacsim/python.sh"),
                str(ROOT / "run_g1d_hospital_vln.py"),
                "--headless",
                "--test",
                "--command",
                command.strip(),
                "--target-id",
                "waiting_area",
                f"--dynamic-docking-pose={pose.x},{pose.y},{pose.yaw}",
                f"--demo-object-pose={target.x},{target.y},{target.z}",
                "--demo-object-size",
                str(target.size_m),
                "--map",
                str(profile.map_yaml),
                "--places",
                str(profile.places),
                "--output-dir",
                str(output),
                "--live-dir",
                str(live_dir),
                "--live-fps",
                str(self.live_fps),
                "--live-resolution",
                self.live_resolution,
                "--position-tolerance",
                "0.03",
                "--yaw-tolerance",
                "0.05",
            ]
            self._process = subprocess.Popen(
                argv,
                cwd=ROOT,
                env=environment,
                stdout=self._log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._started_at = time.time()
            self._active_scene_id = profile.scene_id
            self._last_command = command.strip()
            self._last_plan = plan_payload
            self._last_target = target_payload
        return self.snapshot()

    def cancel(self) -> dict:
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                profile = self._profile(self._active_scene_id)
                state = self.snapshot()
                state.update(
                    {
                        "state": "canceled",
                        "message": "已请求停止物体级精确停靠任务。",
                        "action": "canceled",
                    }
                )
                _atomic_json(profile.output / "live/state.json", state)
        return self.snapshot()

    def map_asset(self, scene_id: str, layer: str) -> Path | None:
        profile = self.profiles.get(scene_id)
        if profile is None:
            return None
        path = profile.map_assets.get(layer)
        return path if path is not None and path.is_file() else None

    def wait_for_camera(self, previous_mtime: int) -> tuple[int, bytes | None]:
        profile = self._profile(self._active_scene_id)
        target = profile.output / "live/camera.jpg"
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            try:
                stat = target.stat()
                if stat.st_mtime_ns != previous_mtime:
                    return stat.st_mtime_ns, target.read_bytes()
            except OSError:
                pass
            time.sleep(0.04)
        return previous_mtime, None


class ObjectDockingHandler(BaseHTTPRequestHandler):
    server_version = "ObjectDockingDashboard/1.0"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        static = {
            "/": (WEB_ROOT / "index.html", "text/html; charset=utf-8"),
            "/app.js": (WEB_ROOT / "app.js", "text/javascript; charset=utf-8"),
            "/styles.css": (WEB_ROOT / "styles.css", "text/css; charset=utf-8"),
        }
        if path in static:
            source, content_type = static[path]
            return self._send_bytes(source.read_bytes(), content_type)
        if path == "/api/config":
            return self._send_json(self.server.session.config())
        if path == "/api/state":
            return self._send_json(self.server.session.snapshot())
        if path.startswith("/asset/map/") and path.endswith(".png"):
            parts = path.removeprefix("/asset/map/").removesuffix(".png").split("/", 1)
            if len(parts) == 2:
                asset = self.server.session.map_asset(parts[0], parts[1])
                if asset is not None:
                    return self._send_bytes(asset.read_bytes(), "image/png")
        if path == "/stream/camera.mjpg":
            return self._send_camera_stream()
        self.send_error(HTTPStatus.NOT_FOUND, "Resource not found")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/cancel":
                return self._send_json(self.server.session.cancel())
            if path != "/api/command":
                self.send_error(HTTPStatus.NOT_FOUND, "Resource not found")
                return
            size = int(self.headers.get("Content-Length", "0"))
            if not 1 <= size <= 4096:
                raise ValueError("请求内容长度无效")
            payload = json.loads(self.rfile.read(size))
            self._send_json(
                self.server.session.submit(
                    str(payload.get("command", "")),
                    str(payload.get("scene_id", "")),
                )
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _send_camera_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        mtime = -1
        try:
            while True:
                mtime, jpeg = self.server.session.wait_for_camera(mtime)
                if jpeg is None:
                    continue
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        if self.path.split("?", 1)[0] not in {"/api/state", "/stream/camera.mjpg"}:
            super().log_message(fmt, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--live-fps", type=int, default=10)
    parser.add_argument("--live-resolution", default="960x540")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = ObjectDockingSession(args)
    server = ThreadingHTTPServer((args.host, args.port), ObjectDockingHandler)
    server.daemon_threads = True
    server.session = session
    print(f"Object docking dashboard: http://{args.host}:{args.port}")
    print("Commands may change object and standoff distance; scenes come from the profile registry.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.cancel()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
