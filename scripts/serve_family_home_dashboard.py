#!/usr/bin/env python3
"""Serve the TCP-only G1-D family-home navigation dashboard."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "family_home_dashboard"
DEFAULT_ARTIFACTS = ROOT / "outputs/family_home_vln"
DEFAULT_OUTPUT = ROOT / "outputs/family_home_web"
sys.path.insert(0, str(ROOT))

from family_home_vln.layout import (  # noqa: E402
    BASE_OBSTACLES,
    HOME_FIXTURES,
    HOME_REGIONS,
    MAP_BOUNDS,
    PLACES,
    ROBOT_RADIUS_M,
    SCENE_NAME,
    START_POSE,
    build_grid,
)
from simple_room_vln.core import path_length, resolve_place  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rectangle_points(
    bounds: tuple[float, float, float, float],
    *,
    category: str,
    step: float = 0.08,
) -> list[dict]:
    xmin, ymin, xmax, ymax = bounds
    points = []
    horizontal_count = max(2, math.ceil((xmax - xmin) / step))
    vertical_count = max(2, math.ceil((ymax - ymin) / step))
    for index in range(horizontal_count + 1):
        x = xmin + (xmax - xmin) * index / horizontal_count
        points.extend(
            (
                {"x": x, "y": ymin, "category": category},
                {"x": x, "y": ymax, "category": category},
            )
        )
    for index in range(1, vertical_count):
        y = ymin + (ymax - ymin) * index / vertical_count
        points.extend(
            (
                {"x": xmin, "y": y, "category": category},
                {"x": xmax, "y": y, "category": category},
            )
        )
    return points


class FamilyHomeDashboardSession:
    def __init__(self, args: argparse.Namespace) -> None:
        self.artifacts = args.artifacts.resolve()
        self.output = args.output.resolve()
        self.live_dir = self.output / "live"
        self.output.mkdir(parents=True, exist_ok=True)
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.grid = build_grid()
        self.places = list(PLACES)
        self._places_by_id = {place.place_id: place for place in self.places}
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._log_stream = None
        self._started_at: float | None = None
        self._last_command = ""
        self.formal_bundle_detected = all(
            (
                (self.artifacts / "lingbot_map/map.yaml").is_file(),
                (self.artifacts / "places_formal.json").is_file(),
                (self.artifacts / "mapping_summary.json").is_file(),
            )
        )
        # A formal navigation bundle alone is not enough for this four-layer
        # UI. Keep every layer bootstrap-labelled until pointcloud, semantic,
        # occupancy and region assets are each reviewed and wired.
        self.formal_map_ready = False

    def config(self) -> dict:
        xmin, ymin, xmax, ymax = MAP_BOUNDS
        source_status = "formal" if self.formal_map_ready else "bootstrap"
        source_label = (
            "LingBot / SAM3 正式制品"
            if self.formal_map_ready
            else "BOOTSTRAP · 等待 LingBot / SAM3 正式替换"
        )
        return {
            "schema_version": 1,
            "scene": SCENE_NAME,
            "mode": "isaac_family_home",
            "map": {
                "width": self.grid.width,
                "height": self.grid.height,
                "resolution": self.grid.resolution,
                "flip_y": True,
                "bounds": {
                    "min_x": xmin,
                    "max_x": xmax,
                    "min_y": ymin,
                    "max_y": ymax,
                },
                "source_status": source_status,
                "source_label": source_label,
                "formal_bundle_detected": self.formal_bundle_detected,
                "robot_radius_m": ROBOT_RADIUS_M,
            },
            "layers": [
                {
                    "id": "pointcloud",
                    "label": "Point Cloud",
                    "status": source_status,
                    "description": (
                        "LingBot RGB 点云"
                        if self.formal_map_ready
                        else "碰撞几何点云代理，不是 LingBot RGB-only 输出"
                    ),
                },
                {
                    "id": "semantic",
                    "label": "Semantic",
                    "status": source_status,
                    "description": "家具语义和审核地点叠加",
                },
                {
                    "id": "occupancy",
                    "label": "Occupancy",
                    "status": source_status,
                    "description": "按 G1-D footprint 膨胀的可通行栅格",
                },
                {
                    "id": "region",
                    "label": "Region",
                    "status": source_status,
                    "description": "卧室、客厅、餐区、厨房和通行区",
                },
            ],
            "places": [
                {
                    "id": place.place_id,
                    "name": place.name,
                    "aliases": list(place.aliases),
                    "pose": asdict(place.pose),
                    "example": f"请带我到{place.name}",
                }
                for place in self.places
            ],
            "camera_stream": "/stream/camera.mjpg",
        }

    def map_data(self) -> dict:
        xmin, ymin, xmax, ymax = MAP_BOUNDS
        pointcloud = _rectangle_points(
            (xmin, ymin, xmax, ymax),
            category="wall",
        )
        for bounds in BASE_OBSTACLES:
            pointcloud.extend(_rectangle_points(bounds, category="existing_furniture"))
        for fixture in HOME_FIXTURES:
            pointcloud.extend(
                _rectangle_points(fixture.bounds_xy, category=fixture.category)
            )
        return {
            "schema_version": 1,
            "source": (
                "formal_lingbot_sam3"
                if self.formal_map_ready
                else "reviewed_procedural_family_home_bootstrap"
            ),
            "truth_boundary": (
                None
                if self.formal_map_ready
                else "Point cloud is a geometry proxy; replace all layers after LingBot/SAM3 review."
            ),
            "occupancy_rows": [
                "".join("." if cell else "#" for cell in row)
                for row in self.grid.free
            ],
            "pointcloud": pointcloud,
            "fixtures": [
                {
                    **asdict(fixture),
                    "bounds_xy": fixture.bounds_xy,
                }
                for fixture in HOME_FIXTURES
            ],
            "regions": [asdict(region) for region in HOME_REGIONS],
        }

    def _idle_state(self) -> dict:
        return {
            "schema_version": 1,
            "state": "idle",
            "message": "输入家庭导航指令后启动 Isaac Sim。",
            "command": self._last_command,
            "task": None,
            "frame": 0,
            "action": "idle",
            "pose": asdict(START_POSE),
            "linear_velocity_mps": 0.0,
            "angular_velocity_rps": 0.0,
            "waypoint": 0,
            "waypoint_count": 0,
            "planned_trajectory": [],
            "trajectory": [],
            "result": None,
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
        try:
            state = _read_json(self.live_dir / "state.json")
        except (OSError, ValueError, json.JSONDecodeError):
            state = self._idle_state()
        state["process_running"] = running
        state["elapsed_sec"] = max(0.0, time.time() - started_at) if started_at else 0.0
        if process is not None and not running and state.get("state") in {"loading", "running"}:
            state["state"] = "failed"
            state["message"] = f"Isaac Sim 进程已退出，返回码 {process.returncode}。"
        return state

    def plan(self, command: str):
        target = resolve_place(command, self.places)
        path = self.grid.plan(
            (START_POSE.x, START_POSE.y),
            (target.pose.x, target.pose.y),
        )
        return target, path

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

    def submit(self, command: str) -> dict:
        command = command.strip()
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise ValueError("已有家庭导航任务正在运行，请先等待或停止任务")
        target, path = self.plan(command)
        with self._lock:
            kit_pids = self._other_kit_processes()
            if kit_pids:
                raise ValueError(
                    "检测到其他 Isaac Kit 进程："
                    + ", ".join(str(pid) for pid in kit_pids)
                    + "；请先停止其他仿真实例"
                )
            for name in ("state.json", "camera.jpg"):
                stale = self.live_dir / name
                if stale.exists():
                    stale.unlink()
            initial = self._idle_state()
            initial.update(
                {
                    "state": "starting",
                    "message": f"指令匹配审核地点“{target.name}”，正在启动 Isaac Sim…",
                    "command": command,
                    "task": target.place_id,
                    "target_name": target.name,
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
            log_path = self.output / "isaac.log"
            self._log_stream = log_path.open("ab", buffering=0)
            environment = os.environ.copy()
            environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            environment.setdefault("OMNI_KIT_ALLOW_ROOT", "1")
            argv = [
                str(ROOT / "isaacsim/python.sh"),
                str(ROOT / "run_g1d_simple_room_vln.py"),
                "--scene-profile",
                "family-home",
                "--allow-bootstrap",
                "--headless",
                "--test",
                "--no-camera",
                "--command",
                command,
                "--output-dir",
                str(self.artifacts),
                "--live-dir",
                str(self.live_dir),
                "--live-fps",
                "10",
                "--live-resolution",
                "960x540",
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
            self._last_command = command
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
                        "message": "已请求停止家庭导航任务。",
                        "action": "canceled",
                    }
                )
                (self.live_dir / "state.json").write_text(
                    json.dumps(state, ensure_ascii=False, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
        return self.snapshot()

    def wait_for_camera(self, previous_mtime: int) -> tuple[int, bytes | None]:
        target = self.live_dir / "camera.jpg"
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


class FamilyHomeHandler(BaseHTTPRequestHandler):
    server_version = "FamilyHomeDashboard/1.0"
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
        if path == "/api/map-data":
            return self._send_json(self.server.session.map_data())
        if path == "/api/state":
            return self._send_json(self.server.session.snapshot())
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
    parser.add_argument("--port", type=int, default=6012)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = FamilyHomeDashboardSession(args)
    server = ThreadingHTTPServer((args.host, args.port), FamilyHomeHandler)
    server.daemon_threads = True
    server.session = session
    print(f"Family-home dashboard: http://{args.host}:{args.port}")
    print(f"Map source: {session.config()['map']['source_label']}")
    print("Examples: 请带我到卧室床边 / 请带我到餐桌旁 / 请带我到厨房操作台")
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
