#!/usr/bin/env python3
"""Serve a TCP-only Hospital dashboard and launch live Isaac Sim navigation."""

from __future__ import annotations

import argparse
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
WEB_ROOT = ROOT / "hospital_dashboard"
DEFAULT_ARTIFACTS = ROOT / "outputs/hospital_vln"
DEFAULT_OUTPUT = ROOT / "outputs/hospital_web"
DEFAULT_DOCKING_CANDIDATES = (
    ROOT / "outputs/hospital_docking/waiting_area_candidates.json"
)
sys.path.insert(0, str(ROOT))

from hospital_vln.artifacts import HOSPITAL_START  # noqa: E402
from hospital_vln.intent import HospitalIntentResolver  # noqa: E402
from simple_room_vln.artifacts import load_lingbot_artifacts  # noqa: E402
from simple_room_vln.core import Place, Pose2D, path_length  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class HospitalDashboardSession:
    def __init__(self, args: argparse.Namespace) -> None:
        self.artifacts = args.artifacts.resolve()
        self.output = args.output.resolve()
        self.live_dir = self.output / "live"
        self.intent_resolution_path = self.output / "intent_resolution.json"
        self.output.mkdir(parents=True, exist_ok=True)
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.map_yaml = self.artifacts / "lingbot_map/map.yaml"
        self.places_path = self.artifacts / "places_formal.json"
        self.mapping_summary_path = self.artifacts / "mapping_summary.json"
        required = (self.map_yaml, self.places_path, self.mapping_summary_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Hospital dashboard artifacts missing: " + ", ".join(missing))

        self.grid, self.places = load_lingbot_artifacts(self.map_yaml, self.places_path)
        self.intent_resolver = getattr(args, "intent_resolver", None) or HospitalIntentResolver(
            self.places_path,
            provider=getattr(args, "intent_provider", "deepseek"),
            allow_rule_fallback=not getattr(args, "no_rule_fallback", False),
        )
        self._places_by_id = {place.place_id: place for place in self.places}
        self.mapping = _read_json(self.mapping_summary_path)
        self.place_payload = _read_json(self.places_path)
        self.dynamic_docking = bool(getattr(args, "dynamic_docking", False))
        self.dynamic_candidate = None
        if self.dynamic_docking:
            docking_path = getattr(
                args,
                "docking_candidates",
                DEFAULT_DOCKING_CANDIDATES,
            ).resolve()
            if not docking_path.is_file():
                raise FileNotFoundError(
                    f"Dynamic docking artifact missing: {docking_path}; "
                    "run ./mobilemanibench.sh hospital-docking first"
                )
            docking = _read_json(docking_path)
            if docking.get("activation") != "explicit_opt_in_only":
                raise ValueError("Dynamic docking artifact is not explicit-opt-in")
            if docking.get("map", {}).get("sha256") != self.place_payload["map"]["sha256"]:
                raise ValueError("Dynamic docking artifact does not match the formal map")
            runtime_blocked = set(getattr(args, "blocked_candidate", []))
            self.dynamic_candidate = next(
                (
                    item
                    for item in docking.get("candidates", [])
                    if item.get("eligible")
                    and item.get("candidate_id") not in runtime_blocked
                ),
                None,
            )
            if self.dynamic_candidate is None:
                raise ValueError("No eligible dynamic docking candidate remains")
        self.map_assets = {}
        for key, value in self.mapping.get("assets", {}).items():
            source = Path(value)
            candidates = (
                source if source.is_absolute() else self.artifacts / source,
                self.artifacts / "map_preview" / f"{key}.png",
            )
            self.map_assets[key] = next(
                (candidate.resolve() for candidate in candidates if candidate.is_file()),
                candidates[0].resolve(),
            )
        expected_layers = {"rgb_pointcloud", "occupancy"}
        missing_layers = sorted(expected_layers - self.map_assets.keys())
        missing_assets = sorted(
            key for key in expected_layers if not self.map_assets.get(key, Path()).is_file()
        )
        if missing_layers or missing_assets:
            details = []
            if missing_layers:
                details.append("missing layers: " + ", ".join(missing_layers))
            if missing_assets:
                details.append("missing files: " + ", ".join(missing_assets))
            raise FileNotFoundError("Hospital dashboard map assets invalid: " + "; ".join(details))
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._log_stream = None
        self._started_at: float | None = None
        self._last_command = ""
        self._last_resolution: dict | None = None
        try:
            saved_resolution = _read_json(self.intent_resolution_path)
        except (OSError, ValueError, json.JSONDecodeError):
            saved_resolution = {}
        if isinstance(saved_resolution, dict):
            self._last_command = str(saved_resolution.get("command", ""))
            value = saved_resolution.get("intent_resolution")
            if isinstance(value, dict):
                self._last_resolution = value
        try:
            previous_state = _read_json(self.live_dir / "state.json")
        except (OSError, ValueError, json.JSONDecodeError):
            previous_state = {}
        if isinstance(previous_state, dict):
            self._last_command = str(previous_state.get("command", self._last_command))
            previous_resolution = previous_state.get("intent_resolution")
            if isinstance(previous_resolution, dict):
                self._last_resolution = previous_resolution

    def config(self) -> dict:
        places = []
        for item in self.place_payload.get("places", []):
            if item.get("status") != "approved":
                continue
            pose = item.get("entrance_pose", {})
            places.append(
                {
                    "id": item["id"],
                    "name": item["name"],
                    "aliases": item.get("aliases", []),
                    "examples": item.get("metadata", {}).get("typical_requests", []),
                    "pose": {
                        "x": float(pose["x"]),
                        "y": float(pose["y"]),
                        "yaw": float(pose.get("yaw", 0.0)),
                    },
                }
            )
        map_config = dict(self.mapping["map"])
        map_config["layers"] = [
            {
                **layer,
                "asset": f"/asset/map/{layer['id']}.png",
            }
            for layer in map_config.get("layers", [])
        ]
        return {
            "schema_version": 1,
            "scene": "Isaac Sim Hospital · G1-D",
            "mode": "isaac_hospital",
            "map": map_config,
            "places": places,
            "camera_stream": "/stream/camera.mjpg",
            "intent_parser": self.intent_resolver.name,
            "docking_mode": (
                "experimental_dynamic_candidate"
                if self.dynamic_docking
                else "formal_fixed_pose"
            ),
        }

    def _idle_state(self) -> dict:
        return {
            "schema_version": 1,
            "state": "idle",
            "message": "输入指令后启动 Isaac Sim Hospital 导航。",
            "command": self._last_command,
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
            "intent_resolution": self._last_resolution,
            "elapsed_sec": 0.0,
            "process_running": False,
        }

    def snapshot(self) -> dict:
        with self._lock:
            process = self._process
            running = process is not None and process.poll() is None
            started_at = self._started_at
            if process is not None and not running and self._log_stream is not None:
                self._log_stream.close()
                self._log_stream = None
        state_path = self.live_dir / "state.json"
        try:
            state = _read_json(state_path)
        except (OSError, ValueError, json.JSONDecodeError):
            state = self._idle_state()
        state["process_running"] = running
        state["intent_resolution"] = self._last_resolution
        state["elapsed_sec"] = max(0.0, time.time() - started_at) if started_at else 0.0
        if process is not None and not running and state.get("state") in {"loading", "running"}:
            state["state"] = "failed"
            state["message"] = f"Isaac Sim 进程已退出，返回码 {process.returncode}。"
        return state

    def plan(self, command: str) -> tuple[object, list[tuple[float, float]], dict]:
        resolution = self.intent_resolver.resolve(command)
        target = self._places_by_id.get(resolution.place_id)
        if target is None:
            raise ValueError(f"解析结果不在 Hospital 正式地点库中：{resolution.place_id}")
        docking_resolution = {
            "mode": "formal_fixed_pose",
            "candidate_id": f"{target.place_id}_reviewed_v1",
            "pose": {
                "x": target.pose.x,
                "y": target.pose.y,
                "yaw": target.pose.yaw,
            },
        }
        if (
            self.dynamic_docking
            and self.dynamic_candidate is not None
            and resolution.place_id == "waiting_area"
        ):
            value = self.dynamic_candidate
            pose = value["pose"]
            target = Place(
                target.place_id,
                target.name,
                target.aliases,
                Pose2D(float(pose["x"]), float(pose["y"]), float(pose["yaw"])),
                target.status,
            )
            docking_resolution = {
                "mode": "experimental_dynamic_candidate",
                "candidate_id": value["candidate_id"],
                "chair_instance_id": value["chair_instance_id"],
                "score": value["score"],
                "clearance_m": value["clearance_m"],
                "path_length_m": value["path_length_m"],
                "pose": pose,
            }
        path = self.grid.plan(
            (HOSPITAL_START.x, HOSPITAL_START.y),
            (target.pose.x, target.pose.y),
        )
        return target, path, {
            "place_id": resolution.place_id,
            "place_name": resolution.place_name,
            "parser": resolution.parser,
            "confidence": resolution.confidence,
            "intent": resolution.intent,
            "docking": docking_resolution,
        }

    @staticmethod
    def _other_kit_processes() -> list[int]:
        result = []
        proc = Path("/proc")
        for entry in proc.iterdir():
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

    def submit(self, command: str) -> dict:
        command = command.strip()
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise ValueError("已有 Hospital 任务正在运行，请先等待或停止任务")
        target, path, resolution = self.plan(command)
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise ValueError("已有 Hospital 任务正在运行，请先等待或停止任务")
            kit_pids = self._other_kit_processes()
            if kit_pids:
                raise ValueError(
                    "检测到其他 Isaac Kit 进程："
                    + ", ".join(str(pid) for pid in kit_pids)
                    + "；请先停止 Streaming Kit，避免多个 Kit 实例争用 GPU"
                )
            for name in ("state.json", "camera.jpg"):
                stale = self.live_dir / name
                if stale.exists():
                    stale.unlink()
            initial = self._idle_state()
            initial.update(
                {
                    "state": "starting",
                    "message": (
                        f"{resolution['parser']} 将指令理解为 {target.name}"
                        f"（置信度 {resolution['confidence']:.2f}），停靠点 "
                        f"{resolution['docking']['candidate_id']}，正在启动 Isaac Sim…"
                    ),
                    "command": command,
                    "task": target.place_id,
                    "intent_resolution": resolution,
                    "planned_trajectory": [{"x": x, "y": y} for x, y in path],
                    "waypoint_count": max(0, len(path) - 1),
                    "path_length_m": path_length(path),
                    "process_running": True,
                }
            )
            (self.live_dir / "state.json").write_text(
                json.dumps(initial, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            temporary_resolution = self.intent_resolution_path.with_name(
                f".{self.intent_resolution_path.name}.tmp"
            )
            temporary_resolution.write_text(
                json.dumps(
                    {
                        "command": command,
                        "intent_resolution": resolution,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            temporary_resolution.replace(self.intent_resolution_path)
            log_path = self.output / "isaac.log"
            self._log_stream = log_path.open("ab", buffering=0)
            environment = os.environ.copy()
            environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            environment.setdefault("OMNI_KIT_ALLOW_ROOT", "1")
            argv = [
                str(ROOT / "isaacsim/python.sh"),
                str(ROOT / "run_g1d_hospital_vln.py"),
                "--headless",
                "--no-camera",
                "--command",
                command,
                "--target-id",
                target.place_id,
                "--map",
                str(self.map_yaml),
                "--places",
                str(self.places_path),
                "--output-dir",
                str(self.artifacts),
                "--live-dir",
                str(self.live_dir),
                "--live-fps",
                "10",
                "--live-resolution",
                "960x540",
            ]
            docking = resolution["docking"]
            if docking["mode"] == "experimental_dynamic_candidate":
                pose = docking["pose"]
                argv.extend(
                    [
                        "--dynamic-docking-pose",
                        f"{pose['x']},{pose['y']},{pose['yaw']}",
                    ]
                )
            self._process = subprocess.Popen(
                argv,
                cwd=ROOT,
                env=environment,
                stdout=self._log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._started_at = time.time()
            self._last_command = command
            self._last_resolution = resolution
        return self.snapshot()

    def cancel(self) -> dict:
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                state = self.snapshot()
                state.update(
                    {
                        "state": "canceled",
                        "message": "已请求停止 Hospital 导航任务。",
                        "action": "canceled",
                    }
                )
                (self.live_dir / "state.json").write_text(
                    json.dumps(state, ensure_ascii=False, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
        return self.snapshot()

    def map_asset(self, layer: str) -> Path | None:
        path = self.map_assets.get(layer)
        return path if path is not None and path.is_file() else None

    def wait_for_camera(self, previous_mtime: int) -> tuple[int, bytes | None]:
        target = self.live_dir / "camera.jpg"
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            try:
                stat = target.stat()
                mtime = stat.st_mtime_ns
                if mtime != previous_mtime:
                    return mtime, target.read_bytes()
            except OSError:
                pass
            time.sleep(0.04)
        return previous_mtime, None


class HospitalHandler(BaseHTTPRequestHandler):
    server_version = "HospitalDashboard/1.0"
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
            layer = path.removeprefix("/asset/map/").removesuffix(".png")
            asset = self.server.session.map_asset(layer)
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
            self._send_json(self.server.session.submit(str(payload.get("command", ""))))
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
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--intent-provider",
        choices=("deepseek", "rule"),
        default=os.getenv("LLM_PROVIDER", "deepseek"),
    )
    parser.add_argument(
        "--no-rule-fallback",
        action="store_true",
        help="Fail startup instead of using exact aliases when the LLM is not configured",
    )
    parser.add_argument(
        "--dynamic-docking",
        action="store_true",
        help="Explicitly use the isolated experimental multi-chair docking selector",
    )
    parser.add_argument(
        "--docking-candidates",
        type=Path,
        default=DEFAULT_DOCKING_CANDIDATES,
    )
    parser.add_argument(
        "--blocked-candidate",
        action="append",
        default=[],
        help="Runtime dynamic-occupancy rejection; may be repeated",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = HospitalDashboardSession(args)
    server = ThreadingHTTPServer((args.host, args.port), HospitalHandler)
    server.daemon_threads = True
    server.session = session
    print(f"Hospital dashboard: http://{args.host}:{args.port}")
    print(f"Intent parser: {session.intent_resolver.name}")
    print(
        "Docking mode: "
        + ("experimental_dynamic_candidate" if session.dynamic_docking else "formal_fixed_pose")
    )
    print("Fuzzy examples: 带我去找个能坐着等医生的地方 / 我想找工作人员问点事情")
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
