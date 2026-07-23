#!/usr/bin/env python3
"""Serve RGB-only LingBot mapping artifacts with occupancy route previews."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lingbot_nav.errors import ConfigurationError  # noqa: E402
from lingbot_nav.intent import create_intent_parser  # noqa: E402
from lingbot_nav.mission import MissionResolver  # noqa: E402
from lingbot_nav.models import Place  # noqa: E402
from lingbot_nav.place_db import PlaceDatabase, normalize_label  # noqa: E402
from lingbot_nav.sim.map_views import render_mapping_views  # noqa: E402
from lingbot_nav.sim.occupancy_planner import (  # noqa: E402
    OccupancyPathPlanner,
    OccupancyPlannerConfig,
)
from lingbot_nav.sim.habitat_rgb_camera import (  # noqa: E402
    HabitatRenderAlignment,
    HabitatRgbCamera,
    HabitatRgbCameraConfig,
)
from lingbot_nav.sim.wheel_nav2 import (  # noqa: E402
    WheelNav2Config,
    WheelNav2Runtime,
    run_wheel_nav2_route,
)


WEB_ROOT = ROOT / "habitat_dashboard"


def _mapping_app_javascript() -> bytes:
    """Use the 8081 UI without changing its source or behavior."""
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    replacements = {
        'document.title = "LingBot · RGB 重建地图导航";': (
            'document.title = "LingBot · RGB 重建地图 Nav2 轮式仿真";'
        ),
        '$("appSubtitle").textContent = "RGB-ONLY RECONSTRUCTION NAVIGATION";': (
            '$("appSubtitle").textContent = "RGB-ONLY · NAV2 WHEEL ROBOT";'
        ),
        '$("navigationSource").textContent = "RGB ONLY · LINGBOT-MAP · OCCUPANCY A*";': (
            '$("navigationSource").textContent = "RGB ONLY · LINGBOT-MAP · NAV2";'
        ),
        '$("cameraLabel").textContent = "RGB RECONSTRUCTION INPUT";': (
            '$("cameraLabel").textContent = "HABITAT RGB · LIVE RENDER";'
        ),
        'if (app.config.mode === "habitat") connectCameraStream();': (
            'if (app.config.mode === "habitat" || app.config.camera_stream) '
            'connectCameraStream();'
        ),
        'if (app.config.mode !== "habitat") updateCamera(state.camera_url);': (
            'if (app.config.mode !== "habitat" && !app.config.camera_stream) '
            'updateCamera(state.camera_url);'
        ),
        '`${state.linear_velocity_mps.toFixed(2)} m/s`': (
            '`${state.linear_velocity_mps.toFixed(2)} map-unit/s`'
        ),
        '`x ${state.pose.x.toFixed(2)} · z ${state.pose.y.toFixed(2)} · yaw ${state.pose.yaw.toFixed(2)}`': (
            '`x ${state.pose.x.toFixed(2)} · y ${state.pose.y.toFixed(2)} · yaw ${state.pose.yaw.toFixed(2)}`'
        ),
        'const regionPlaces = app.config.places.filter((place) => place.metadata?.target_type === "semantic_region");': (
            'const regionPlaces = app.config.places.filter((place) => '
            'place.metadata?.target_type === "semantic_region" || '
            'String(place.metadata?.target_type || "").endsWith("object_instance"));'
        ),
        '$("placeCount").textContent = `${regionPlaces.length} REGIONS`;': (
            '$("placeCount").textContent = `${regionPlaces.length} TARGETS`;'
        ),
    }
    for old, new in replacements.items():
        if old in source:
            source = source.replace(old, new)
    return source.encode("utf-8")


class MappingHandler(BaseHTTPRequestHandler):
    """HTTP adapter for the LingBot/Nav2 session and render-only RGB stream."""

    server_version = "LingBotMappingDashboard/1.0"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        static = {
            "/": (WEB_ROOT / "index.html", "text/html; charset=utf-8"),
            "/styles.css": (WEB_ROOT / "styles.css", "text/css; charset=utf-8"),
            "/map_layers.css": (
                WEB_ROOT / "map_layers.css",
                "text/css; charset=utf-8",
            ),
            "/object.css": (WEB_ROOT / "object.css", "text/css; charset=utf-8"),
        }
        if path == "/app.js":
            return self._send_bytes(
                _mapping_app_javascript(), "text/javascript; charset=utf-8"
            )
        if path in static:
            file_path, content_type = static[path]
            return self._send_bytes(file_path.read_bytes(), content_type)
        if path == "/api/config":
            return self._send_json(self.server.session.config())
        if path == "/api/state":
            return self._send_json(self.server.session.snapshot())
        if path.startswith("/asset/map/") and path.endswith(".png"):
            layer_id = path.removeprefix("/asset/map/").removesuffix(".png")
            asset = self.server.session.map_asset(layer_id)
            if asset is not None:
                return self._send_file(asset, "image/png")
        if path == "/asset/camera.png":
            return self._send_file(self.server.session.latest_rgb(), "image/png")
        if path == "/stream/camera.mjpg":
            return self._send_camera_stream()
        if path.startswith("/asset/object-review/") and path.endswith(".png"):
            place_id = path.removeprefix("/asset/object-review/").removesuffix(".png")
            review = self.server.session.review_image(place_id, ROOT)
            if review is not None:
                return self._send_file(review, "image/png")
        self.send_error(HTTPStatus.NOT_FOUND, "Resource not found")

    def _send_camera_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        sequence = -1
        try:
            while True:
                sequence, jpeg = self.server.session.wait_for_camera(sequence)
                if jpeg is None:
                    continue
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                )
                self.wfile.write(header)
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

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

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            self._send_bytes(path.read_bytes(), content_type)
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "Asset not found")

    def _send_json(
        self, value: object, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self._send_bytes(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(),
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
        path = self.path.split("?", 1)[0]
        if path not in {"/api/state", "/asset/camera.png", "/stream/camera.mjpg"}:
            super().log_message(fmt, *args)


def _artifact_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _require_rgb_only_provenance(
    args: argparse.Namespace,
    candidates: dict[str, object],
    catalog: dict[str, object],
    manifest: dict[str, object],
) -> None:
    prohibited = manifest.get("prohibited_ground_truth_inputs", {})
    required_manifest_flags = (
        "habitat_poses",
        "habitat_depth",
        "habitat_navmesh",
        "habitat_semantics",
    )
    if any(prohibited.get(key) is not False for key in required_manifest_flags):
        raise ConfigurationError(
            "RGB-only manifest must explicitly exclude every Habitat ground-truth input"
        )
    trajectory = manifest.get("predicted_camera_trajectory", {})
    if trajectory.get("source") != "lingbot_predicted_extrinsics_only":
        raise ConfigurationError("Navigation start must come from LingBot-predicted extrinsics")
    manifest_map = manifest.get("artifacts", {}).get("map_yaml")
    if not manifest_map or _artifact_path(manifest_map) != args.map_yaml.resolve():
        raise ConfigurationError("Map YAML does not match the RGB-only LingBot manifest")
    manifest_pointcloud = manifest.get("artifacts", {}).get("pointcloud")
    if (
        not manifest_pointcloud
        or _artifact_path(manifest_pointcloud) != args.pointcloud.resolve()
    ):
        raise ConfigurationError(
            "Point cloud does not match the RGB-only LingBot manifest"
        )
    rgb_directory = manifest.get("inputs", {}).get("rgb_directory")
    if not rgb_directory:
        raise ConfigurationError("RGB-only manifest does not identify its RGB input")
    try:
        args.initial_rgb.resolve().relative_to(_artifact_path(rgb_directory))
    except ValueError as exc:
        raise ConfigurationError(
            "Initial RGB frame is outside the manifest RGB sequence"
        ) from exc

    if candidates.get("artifact_type") != "unverified_place_candidates":
        raise ConfigurationError("Unsupported object candidate artifact")
    geometry = candidates.get("geometry", {})
    if geometry.get("backend") != "lingbot_map":
        raise ConfigurationError("Object geometry must come from LingBot-Map")
    if _artifact_path(geometry.get("map_source", "")) != args.map_yaml.resolve():
        raise ConfigurationError("Object candidates and occupancy use different map frames")
    manifest_predictions = manifest.get("inputs", {}).get("predictions")
    if (
        not manifest_predictions
        or _artifact_path(geometry.get("predictions", ""))
        != _artifact_path(manifest_predictions)
    ):
        raise ConfigurationError(
            "Object candidates do not come from the selected LingBot RGB predictions"
        )
    candidate_maps = candidates.get("maps", {})
    expected_candidate_maps = {
        "semantic_map": args.semantic_map,
        "instance_map": args.instance_map,
    }
    for key, expected in expected_candidate_maps.items():
        if _artifact_path(candidate_maps.get(key, "")) != expected.resolve():
            raise ConfigurationError(f"Object {key} does not match the selected artifact")

    if catalog.get("artifact_type") != "rgb_only_semantic_regions":
        raise ConfigurationError("Unsupported semantic region catalog")
    if catalog.get("frame_id") != candidates.get("frame_id"):
        raise ConfigurationError("Region and object artifacts use different map frames")
    provenance = catalog.get("provenance", {})
    required_region_flags = (
        "habitat_depth",
        "habitat_camera_poses",
        "habitat_navmesh",
        "habitat_semantics",
    )
    if any(provenance.get(key) is not False for key in required_region_flags):
        raise ConfigurationError(
            "Region catalog must explicitly exclude every Habitat ground-truth input"
        )
    if provenance.get("geometry") != "lingbot_map_predicted_depth_intrinsics_extrinsics":
        raise ConfigurationError("Region geometry is not a LingBot-Map prediction")
    if _artifact_path(provenance.get("object_evidence", "")) != args.candidates.resolve():
        raise ConfigurationError("Region catalog and object candidates have different evidence")


class MappingViewSession:
    """Interactive Nav2 wheel simulation using only LingBot map artifacts."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.scene = args.scene_label
        self.output_root = args.output.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._animation_delay = args.animation_delay
        self._navigation_backend = args.navigation_backend
        self._realtime_factor = args.realtime_factor
        self._robot_radius = args.robot_radius
        self._map_yaml = args.map_yaml.resolve()

        candidates = json.loads(args.candidates.resolve().read_text(encoding="utf-8"))
        catalog = json.loads(args.region_catalog.resolve().read_text(encoding="utf-8"))
        manifest_path = args.map_yaml.resolve().parent / "rgb_only_manifest.json"
        if not manifest_path.is_file():
            raise ConfigurationError(
                f"RGB-only manifest is required beside map.yaml: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require_rgb_only_provenance(args, candidates, catalog, manifest)

        self.map_meta, self.map_assets = render_mapping_views(
            args.map_yaml.resolve(),
            self.output_root / "map_layers",
            pointcloud_path=args.pointcloud.resolve(),
            semantic_map_path=args.semantic_map.resolve(),
            instance_map_path=args.instance_map.resolve(),
            region_map_path=args.region_map.resolve(),
        )
        self.map_meta["unit_label"] = "LingBot unit"
        self.topdown_path = self.map_assets["occupancy"]
        self._initial_rgb = args.initial_rgb.resolve()
        self._latest_rgb = self._initial_rgb
        self._planner = OccupancyPathPlanner(
            args.map_yaml.resolve(),
            OccupancyPlannerConfig(
                robot_radius=args.robot_radius,
                max_snap_distance=args.max_snap_distance,
                unknown_is_occupied=True,
            ),
        )

        start_values = manifest["predicted_camera_trajectory"]["start"]
        render_alignment_payload = json.loads(
            args.renderer_alignment.resolve().read_text(encoding="utf-8")
        )
        render_alignment = HabitatRenderAlignment.from_mapping(
            render_alignment_payload
        )
        self._start = {
            "x": float(start_values[0]),
            "y": float(start_values[1]),
            "yaw": render_alignment.map_start_yaw_rad,
        }
        self._render_alignment = render_alignment
        self._render_alignment_path = args.renderer_alignment.resolve()
        self._camera_config = HabitatRgbCameraConfig(
            scene=args.habitat_scene,
            scene_dataset_config=args.habitat_scene_dataset_config,
            alignment=render_alignment,
            width=args.camera_width,
            height=args.camera_height,
            sensor_height=args.camera_sensor_height,
            hfov_degrees=args.camera_hfov,
        )
        self._camera_condition = threading.Condition()
        self._camera_pending: tuple[int, dict[str, float]] | None = None
        self._camera_request_sequence = 0
        self._camera_sequence = 0
        self._camera_stopping = False
        self._camera_ready = False
        self._camera_error: Exception | None = None
        self._camera_path = self.output_root / "camera" / "latest.png"
        preview = BytesIO()
        from PIL import Image

        with Image.open(self._initial_rgb) as image:
            image.convert("RGB").save(preview, format="JPEG", quality=85)
        self._latest_jpeg: bytes | None = preview.getvalue()
        start_xy = (self._start["x"], self._start["y"])
        frame_id = str(catalog.get("frame_id", "map"))
        start_place = Place.from_mapping(
            {
                "id": "lingbot_start",
                "name": "LingBot RGB轨迹起点",
                "aliases": ["lingbot_start"],
                "entrance_pose": {**self._start, "frame_id": frame_id},
                "region": "lingbot_predicted_trajectory",
                "metadata": {
                    "target_type": "internal_start",
                    "internal": True,
                    "source": "lingbot_predicted_extrinsics_only",
                    "habitat_ground_truth_used": False,
                },
            },
            frame_id,
        )

        self._regions = [dict(item) for item in catalog.get("regions", [])]
        region_by_place = {
            str(item.get("place_id", "")): item for item in self._regions
        }
        target_places: list[Place] = []
        for raw in catalog.get("places", []):
            place = Place.from_mapping(raw, frame_id)
            reachable = self._is_reachable(
                start_xy, (place.entrance_pose.x, place.entrance_pose.y)
            )
            value = place.to_dict()
            metadata = dict(value["metadata"])
            metadata["navigation"] = {
                "planner": "lingbot_occupancy_astar",
                "reachable": reachable,
                "habitat_ground_truth_used": False,
            }
            value["metadata"] = metadata
            place = Place.from_mapping(value, frame_id)
            region = region_by_place.get(place.place_id)
            if region is not None:
                region["navigable"] = reachable
            if reachable:
                target_places.append(place)

        raw_navigable: list[tuple[dict[str, object], dict[str, object]]] = []
        for item in candidates.get("instances", []):
            selected = self._reachable_candidate(start_xy, item.get("candidate_poses", []))
            if selected is not None:
                raw_navigable.append((item, selected))
        primary_by_label: dict[str, str] = {}
        for item, selected in raw_navigable:
            label = str(item.get("semantic_label", ""))
            score = (
                int(item.get("observation_count", 0)),
                float(selected.get("geometry_score", 0.0)),
            )
            current = next(
                (
                    pair
                    for pair in raw_navigable
                    if pair[0].get("instance_id") == primary_by_label.get(label)
                ),
                None,
            )
            current_score = (
                (
                    int(current[0].get("observation_count", 0)),
                    float(current[1].get("geometry_score", 0.0)),
                )
                if current is not None
                else (-1, -math.inf)
            )
            if score > current_score:
                primary_by_label[label] = str(item.get("instance_id", ""))

        reserved_aliases = {
            normalize_label(alias)
            for place in target_places
            for alias in (place.name, *place.aliases)
        }
        selected_by_id = {
            str(item.get("instance_id", "")): selected
            for item, selected in raw_navigable
        }
        self._objects: list[dict[str, object]] = []
        self._review_images: dict[str, Path] = {}
        for item in candidates.get("instances", []):
            instance_id = str(item.get("instance_id", ""))
            selected = selected_by_id.get(instance_id)
            center = item.get("center_map", {})
            navigable = selected is not None
            entry = {
                "id": instance_id,
                "name": item.get("name", instance_id),
                "semantic_label": item.get("semantic_label", ""),
                "x": center.get("x"),
                "y": center.get("y"),
                "observation_count": item.get("observation_count", 0),
                "candidate_pose_count": len(item.get("candidate_poses", [])),
                "navigable": navigable,
                "status": "navigable" if navigable else "unreachable_on_lingbot_occupancy",
            }
            self._objects.append(entry)
            if not navigable:
                continue
            label = str(item.get("semantic_label", ""))
            aliases = []
            for alias in item.get("aliases", []):
                normalized = normalize_label(str(alias))
                numbered = any(character.isdigit() for character in str(alias))
                if (
                    normalized
                    and normalized not in reserved_aliases
                    and (
                        instance_id == primary_by_label.get(label)
                        or numbered
                    )
                ):
                    aliases.append(str(alias))
            aliases.extend((str(item.get("name", instance_id)), instance_id))
            value = {
                "id": instance_id,
                "name": item.get("name", instance_id),
                "aliases": list(dict.fromkeys(aliases)),
                "entrance_pose": {
                    "x": selected["x"],
                    "y": selected["y"],
                    "yaw": selected.get("yaw", 0.0),
                    "frame_id": frame_id,
                },
                "region": f"object_instance_{instance_id}",
                "metadata": {
                    "target_type": "navigable_object_instance",
                    "semantic_label": label,
                    "instance_id": instance_id,
                    "instance_center": center,
                    "standoff_m": selected.get("standoff_m", 0.0),
                    "clearance_m": selected.get("clearance_m", 0.0),
                    "review_image": item.get("representative_review_image", ""),
                    "verification": {
                        "status": "demo_enabled",
                        "reviewer": "geometry-auto-selection",
                        "evidence": [
                            "candidate pose generated from LingBot point cloud",
                            "route exists on LingBot occupancy",
                        ],
                    },
                    "provenance": {
                        "geometry_backend": "lingbot_map",
                        "habitat_ground_truth_used": False,
                    },
                },
            }
            target_places.append(Place.from_mapping(value, frame_id))
            review = str(item.get("representative_review_image", ""))
            if review:
                path = _artifact_path(review)
                if path.is_file():
                    self._review_images[instance_id] = path

        self._database = PlaceDatabase((start_place, *target_places), frame_id)
        intent_parser = create_intent_parser(
            args.llm_provider,
            self._database,
            allow_rule_fallback=not args.no_rule_fallback,
        )
        self._resolver = MissionResolver(
            intent_parser, self._database
        )
        self._intent_parser_name = intent_parser.name
        self._places = []
        for place in self._database.places:
            if place.metadata.get("internal"):
                continue
            value = place.to_dict()
            if place.place_id in self._review_images:
                value["review_asset"] = f"/asset/object-review/{place.place_id}.png"
            self._places.append(value)
        self._navigable_object_count = sum(
            bool(item["navigable"]) for item in self._objects
        )
        self._started = time.monotonic()
        self._state = self._idle_state()
        self._wheel_runtime: WheelNav2Runtime | None = None
        self._manifest = manifest
        self._examples = [
            "请带我到用餐区域",
            *(
                f"请带我到{place.name}"
                for place in target_places
                if place.metadata.get("target_type") == "semantic_region"
                and place.name != "用餐区域"
            ),
            *(
                f"请带我到{place.name}旁边"
                for place in target_places
                if str(place.metadata.get("target_type", "")).endswith("object_instance")
            ),
        ][:8]
        self._camera_thread = threading.Thread(
            target=self._camera_loop,
            name="habitat-rgb-render-camera",
            daemon=True,
        )
        self._camera_thread.start()
        with self._camera_condition:
            initialized = self._camera_condition.wait_for(
                lambda: self._camera_ready or self._camera_error is not None,
                timeout=30.0,
            )
            if not initialized:
                raise ConfigurationError("Habitat RGB renderer startup timed out")
            if self._camera_error is not None:
                raise ConfigurationError(
                    f"Habitat RGB renderer failed to start: {self._camera_error}"
                ) from self._camera_error
        self._request_camera(dict(self._start))

    def _idle_state(self) -> dict[str, object]:
        nav2 = self._navigation_backend == "nav2"
        return {
            "state": "idle",
            "message": (
                "输入区域或物品指令，Nav2 将在 LingBot occupancy 上驱动轮式机器人"
                if nav2
                else "输入区域或物品指令，使用 LingBot occupancy 规划路线"
            ),
            "command": "",
            "destination": "",
            "destination_name": "",
            "route": [],
            "planned_trajectory": [],
            "exploration_trajectory": [],
            "trajectory": [dict(self._start)],
            "pose": dict(self._start),
            "action": "nav2_wheel_ready" if nav2 else "lingbot_map_ready",
            "collisions": 0,
            "frame": 0,
            "terminal": False,
            "controller": (
                "nav2_regulated_pure_pursuit+differential_wheel"
                if nav2
                else "lingbot_occupancy_preview"
            ),
            "linear_velocity_mps": 0.0,
            "angular_velocity_rps": 0.0,
            "pipeline": [
                {"id": "rgb", "name": "RGB 图像输入", "state": "done"},
                {"id": "lingbot", "name": "LingBot-Map 几何重建", "state": "done"},
                {"id": "occupancy", "name": "LingBot occupancy", "state": "done"},
                {"id": "semantic", "name": "Semantic / Instance", "state": "done"},
                {"id": "region", "name": "Region", "state": "done"},
            ],
        }

    def _is_reachable(
        self, start: tuple[float, float], goal: tuple[float, float]
    ) -> bool:
        try:
            self._planner.plan(start, goal)
        except ConfigurationError:
            return False
        return True

    def _reachable_candidate(
        self, start: tuple[float, float], candidates: object
    ) -> dict[str, object] | None:
        if not isinstance(candidates, list):
            return None
        for pose in candidates:
            try:
                goal = (float(pose["x"]), float(pose["y"]))
                self._planner.plan(start, goal)
            except (KeyError, TypeError, ValueError, ConfigurationError):
                continue
            return dict(pose)
        return None

    def config(self) -> dict[str, object]:
        nav2 = self._navigation_backend == "nav2"
        return {
            "mode": "rgb_only_mapping_navigation",
            "camera_stream": True,
            "control_mode": "differential_wheel" if nav2 else "map_preview",
            "navigation_backend": "nav2" if nav2 else "lingbot_occupancy_astar",
            "intent_parser": self._intent_parser_name,
            "scene": self.scene,
            "map": self.map_meta,
            "places": self._places,
            "object_candidates": self._objects,
            "detected_objects": self._objects,
            "detected_regions": self._regions,
            "target_summary": {
                "navigable_regions": sum(
                    bool(item.get("navigable")) for item in self._regions
                ),
                "detected_objects": len(self._objects),
                "navigable_objects": self._navigable_object_count,
            },
            "exploration": {
                "source": "lingbot_predicted_extrinsics+occupancy",
                "start": dict(self._start),
                "habitat_ground_truth_used": False,
            },
            "examples": self._examples,
            "ground_truth_inputs": {
                "habitat_depth": False,
                "habitat_camera_poses": False,
                "habitat_semantics": False,
                "habitat_navmesh": False,
                "simulator_map_truth": False,
                "habitat_rgb_scene_rendering": True,
                "render_only_rgb_correspondence_alignment": True,
            },
            "provenance": {
                "sensor_input": "rgb_sequence_only",
                "geometry": "lingbot_map_predicted_depth_intrinsics_extrinsics",
                "occupancy": "lingbot_map_rgb_reconstruction",
                "object_targets": "owlv2_sam2_on_rgb_plus_lingbot_geometry",
                "region_targets": "clipseg_on_rgb_plus_lingbot_geometry",
                "motion": (
                    "nav2_cmd_vel+differential_wheel_kinematics"
                    if nav2
                    else "occupancy_astar_preview"
                ),
                "camera": (
                    "habitat_textured_scene_rgb_from_integrated_cmd_vel_pose+"
                    "render_only_rgb_correspondence_sim2"
                ),
            },
            "planning_inputs": {
                "planner": "nav2" if nav2 else "astar_preview",
                "occupancy": "lingbot_map_rgb_only",
                "object_targets": "lingbot_map+owlv2+sam2",
                "region_targets": "lingbot_map+clipseg+object_context",
                "habitat_navmesh": False,
                "habitat_depth": False,
                "habitat_semantics": False,
                "habitat_camera_poses": False,
                "simulator_ground_truth_map": False,
                "simulator_role": "rgb_renderer_only",
            },
            "render_alignment": {
                "artifact": str(self._render_alignment_path),
                "consumer": "habitat_rgb_camera_only",
                "scale_m_per_lingbot_unit": self._render_alignment.scale,
                "rotation_rad": self._render_alignment.rotation_rad,
                "translation_xz_m": [
                    self._render_alignment.translation_x,
                    self._render_alignment.translation_z,
                ],
                "yaw_offset_rad": self._render_alignment.yaw_offset_rad,
                "correspondence_count": self._render_alignment.correspondence_count,
                "position_rmse_m": self._render_alignment.position_rmse_m,
                "navigation_consumer": False,
            },
        }

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            value = dict(self._state)
            value["route"] = list(self._state.get("route", []))
            value["planned_trajectory"] = list(
                self._state.get("planned_trajectory", [])
            )
            value["exploration_trajectory"] = list(
                self._state.get("exploration_trajectory", [])
            )
            value["trajectory"] = list(self._state.get("trajectory", []))
            value["pipeline"] = [
                dict(item) for item in self._state.get("pipeline", [])
            ]
            value["elapsed_sec"] = time.monotonic() - self._started
            value["camera_url"] = f"/asset/camera.png?frame={self._camera_sequence}"
            planned = value["planned_trajectory"]
            if planned:
                goal = planned[-1]
                pose = value["pose"]
                value["distance_remaining"] = math.hypot(
                    float(goal["x"]) - float(pose["x"]),
                    float(goal["y"]) - float(pose["y"]),
                )
            else:
                value["distance_remaining"] = None
            return value

    def submit(self, command: str) -> dict[str, object]:
        command = command.strip()
        if not command:
            raise ValueError("指令不能为空")
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise ValueError("轮式机器人正在导航，请等待完成或先取消")
            try:
                mission = self._resolver.resolve(command)
                current = self._state.get("pose", self._start)
                route_start = (float(current["x"]), float(current["y"]))
                route_points: list[tuple[float, float]] = []
                for step in mission.steps:
                    target_xy = (
                        step.place.entrance_pose.x,
                        step.place.entrance_pose.y,
                    )
                    segment = self._planner.plan(route_start, target_xy)
                    route_points.extend(segment if not route_points else segment[1:])
                    route_start = target_xy
            except Exception as exc:
                raise ValueError(str(exc)) from exc
            if self._navigation_backend == "nav2":
                return self._submit_nav2(command, mission, route_points)
            target = mission.place
            points = route_points
            planned = [{"x": float(x), "y": float(y)} for x, y in points]
            is_object = str(target.metadata.get("target_type", "")).endswith(
                "object_instance"
            )
            self._cancel.clear()
            self._started = time.monotonic()
            self._state = {
                "state": "navigating",
                "message": "正在执行 LingBot occupancy A* 路线预览",
                "command": command,
                "destination": target.place_id,
                "destination_name": target.name,
                "route": [
                    {"action": "arrive", "id": target.place_id, "name": target.name}
                ],
                "planned_trajectory": planned,
                "exploration_trajectory": planned,
                "trajectory": [dict(self._start)],
                "pose": dict(self._start),
                "action": "occupancy_astar",
                "collisions": 0,
                "frame": 0,
                "terminal": False,
                "controller": "lingbot_occupancy_preview",
                "linear_velocity_mps": 0.0,
                "angular_velocity_rps": 0.0,
                "target": target.to_dict(),
                "pipeline": [
                    {"id": "language", "name": "解析区域/物品指令", "state": "done"},
                    {
                        "id": "goal",
                        "name": "读取安全停靠点" if is_object else "读取区域入口点",
                        "state": "done",
                    },
                    {
                        "id": "occupancy",
                        "name": "LingBot occupancy A*",
                        "state": "done",
                    },
                    {
                        "id": "motion",
                        "name": "地图路线预览",
                        "state": "active",
                    },
                ],
            }
            self._worker = threading.Thread(
                target=self._animate, args=(points,), daemon=True
            )
            self._worker.start()
            return self.snapshot()

    def _submit_nav2(self, command, mission, points) -> dict[str, object]:
        target = mission.place
        current = dict(self._state.get("pose", self._start))
        planned = [{"x": float(x), "y": float(y)} for x, y in points]
        is_object = str(target.metadata.get("target_type", "")).endswith(
            "object_instance"
        )
        self._cancel.clear()
        self._started = time.monotonic()
        self._state = {
            "state": "understanding",
            "message": "目标已解析，正在等待 Nav2 生成 LingBot occupancy 路径",
            "command": command,
            "destination": target.place_id,
            "destination_name": target.name,
            "route": [
                {
                    "action": step.action.value,
                    "id": step.place.place_id,
                    "name": step.place.name,
                }
                for step in mission.steps
            ],
            "planned_trajectory": planned,
            "exploration_trajectory": [],
            "trajectory": [current],
            "pose": current,
            "action": "waiting_for_nav2",
            "collisions": 0,
            "frame": 0,
            "terminal": False,
            "controller": "nav2_regulated_pure_pursuit+differential_wheel",
            "linear_velocity_mps": 0.0,
            "angular_velocity_rps": 0.0,
            "target": target.to_dict(),
            "pipeline": [
                {"id": "language", "name": "解析区域/物品指令", "state": "done"},
                {
                    "id": "goal",
                    "name": "读取物品安全停靠点" if is_object else "读取区域入口点",
                    "state": "done",
                },
                {
                    "id": "occupancy",
                    "name": "Nav2 全局规划（LingBot occupancy）",
                    "state": "active",
                },
                {
                    "id": "motion",
                    "name": "差速轮式机器人执行 /cmd_vel",
                    "state": "pending",
                },
                {
                    "id": "arrive",
                    "name": "到达物品旁" if is_object else "进入目标区域",
                    "state": "pending",
                },
            ],
        }
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output = self.output_root / "runs" / run_id
        self._worker = threading.Thread(
            target=self._run_nav2,
            args=(command, mission, output),
            daemon=True,
        )
        self._worker.start()
        return self.snapshot()

    def _run_nav2(self, command, mission, output: Path) -> None:
        try:
            with self._lock:
                start = dict(self._state["pose"])
            result = run_wheel_nav2_route(
                WheelNav2Config(
                    map_yaml=self._map_yaml,
                    output_dir=output,
                    instruction=command,
                    start_x=float(start["x"]),
                    start_y=float(start["y"]),
                    start_yaw=float(start["yaw"]),
                    robot_radius=self._robot_radius,
                    realtime_factor=self._realtime_factor,
                ),
                self._resolver,
                mission=mission,
                progress_callback=self._on_nav2_progress,
                cancel_check=self._cancel.is_set,
                runtime=self._wheel_runtime,
            )
            with self._lock:
                self._state["state"] = "arrived"
                self._state["message"] = f"已到达{self._state['destination_name']}"
                self._state["terminal"] = True
                self._state["action"] = "nav2_arrived"
                self._state["collisions"] = result["collisions"]
                for stage in self._state["pipeline"]:
                    stage["state"] = "done"
        except Exception as exc:
            with self._lock:
                cancelled = self._cancel.is_set()
                self._state["state"] = "cancelled" if cancelled else "failed"
                self._state["message"] = "导航已取消" if cancelled else str(exc)
                self._state["terminal"] = True
                self._state["action"] = "cancelled" if cancelled else "nav2_failed"
                for stage in self._state.get("pipeline", []):
                    if stage["state"] == "active":
                        stage["state"] = "failed"

    def _on_nav2_progress(self, update: dict[str, object]) -> None:
        with self._lock:
            kind = update["kind"]
            if kind == "goal":
                self._state["destination"] = update["destination"]
                self._state["destination_name"] = update["destination_name"]
                self._state["route"] = list(update["route"])
                self._state["target"] = update["target"]
                return
            if kind == "planned":
                self._state["planned_trajectory"] = [
                    {"x": float(point[0]), "y": float(point[2])}
                    for point in update["points"]
                ]
                self._state["state"] = "navigating"
                self._state["message"] = "Nav2 正在 LingBot 重建地图上驱动轮式机器人"
                self._state["controller"] = str(update["controller"])
                for stage in self._state["pipeline"]:
                    if stage["id"] == "occupancy":
                        stage["state"] = "done"
                    elif stage["id"] == "motion":
                        stage["state"] = "active"
                return
            sample = update["sample"]
            position = sample["map_position"]
            pose = {
                "x": float(position[0]),
                "y": float(position[1]),
                "yaw": float(sample["map_yaw"]),
            }
            self._state["pose"] = pose
            self._state["trajectory"].append(pose)
            self._state["frame"] = int(sample["frame"])
            self._state["action"] = str(sample["action"])
            self._state["linear_velocity_mps"] = float(
                sample["linear_velocity_mps"]
            )
            self._state["angular_velocity_rps"] = float(
                sample["angular_velocity_rps"]
            )
            self._state["collisions"] = int(self._state["collisions"]) + int(
                bool(sample["collided"])
            )
            self._request_camera(pose)

    def _animate(self, points: list[tuple[float, float]]) -> None:
        samples: list[tuple[float, float]] = [points[0]]
        for before, after in zip(points, points[1:]):
            distance = math.hypot(after[0] - before[0], after[1] - before[1])
            steps = max(1, int(math.ceil(distance / 0.04)))
            samples.extend(
                (
                    before[0] + (after[0] - before[0]) * index / steps,
                    before[1] + (after[1] - before[1]) * index / steps,
                )
                for index in range(1, steps + 1)
            )
        for frame, point in enumerate(samples):
            if self._cancel.is_set():
                with self._lock:
                    self._state["state"] = "cancelled"
                    self._state["message"] = "路线预览已取消"
                    self._state["terminal"] = True
                    self._state["action"] = "cancelled"
                return
            next_point = samples[min(frame + 1, len(samples) - 1)]
            yaw = math.atan2(next_point[1] - point[1], next_point[0] - point[0])
            pose = {"x": point[0], "y": point[1], "yaw": yaw}
            with self._lock:
                self._state["pose"] = pose
                self._state["frame"] = frame
                self._state["trajectory"].append(pose)
                self._state["action"] = "follow_lingbot_occupancy"
            self._request_camera(pose)
            if self._animation_delay:
                time.sleep(self._animation_delay)
        with self._lock:
            self._state["state"] = "arrived"
            self._state["message"] = f"已到达{self._state['destination_name']}"
            self._state["terminal"] = True
            self._state["action"] = "arrived"
            for stage in self._state["pipeline"]:
                stage["state"] = "done"

    def cancel(self) -> dict[str, object]:
        self._cancel.set()
        return self.snapshot()

    def _request_camera(self, pose: dict[str, float]) -> None:
        with self._camera_condition:
            self._camera_request_sequence += 1
            self._camera_pending = (
                self._camera_request_sequence,
                {key: float(pose[key]) for key in ("x", "y", "yaw")},
            )
            self._camera_condition.notify()

    def _camera_loop(self) -> None:
        try:
            renderer = HabitatRgbCamera(self._camera_config)
        except Exception as exc:
            with self._camera_condition:
                self._camera_error = exc
                self._camera_condition.notify_all()
            return
        with self._camera_condition:
            self._camera_ready = True
            self._camera_condition.notify_all()
        try:
            while True:
                with self._camera_condition:
                    while self._camera_pending is None and not self._camera_stopping:
                        self._camera_condition.wait()
                    if self._camera_stopping:
                        return
                    sequence, pose = self._camera_pending
                    self._camera_pending = None
                try:
                    path, jpeg = renderer.render_to(
                        self._camera_path,
                        x=pose["x"],
                        y=pose["y"],
                        yaw=pose["yaw"],
                    )
                except Exception:
                    # Navigation must remain available if only the visual
                    # stream fails; retain the newest valid RGB frame.
                    continue
                with self._camera_condition:
                    self._latest_rgb = path
                    self._latest_jpeg = jpeg
                    self._camera_sequence = sequence
                    self._camera_condition.notify_all()
        finally:
            renderer.close()

    def latest_rgb(self) -> Path:
        with self._camera_condition:
            return self._latest_rgb

    def wait_for_camera(
        self, after_sequence: int, timeout: float = 10.0
    ) -> tuple[int, bytes | None]:
        with self._camera_condition:
            self._camera_condition.wait_for(
                lambda: (
                    self._camera_sequence > after_sequence
                    and self._latest_jpeg is not None
                ),
                timeout=timeout,
            )
            if (
                self._camera_sequence <= after_sequence
                or self._latest_jpeg is None
            ):
                return after_sequence, None
            return self._camera_sequence, self._latest_jpeg

    def close(self) -> None:
        self.cancel()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=5.0)
        if self._wheel_runtime is not None:
            self._wheel_runtime.close()
            self._wheel_runtime = None
        with self._camera_condition:
            self._camera_stopping = True
            self._camera_condition.notify_all()
        self._camera_thread.join(timeout=2.0)

    def start_pose(self) -> dict[str, float]:
        return dict(self._start)

    def start_wheel_runtime(self) -> None:
        if self._navigation_backend != "nav2" or self._wheel_runtime is not None:
            return
        self._wheel_runtime = WheelNav2Runtime(
            self._start["x"], self._start["y"], self._start["yaw"]
        )

    def map_asset(self, layer_id: str) -> Path | None:
        return self.map_assets.get(layer_id)

    def review_image(self, place_id: str, _root: Path) -> Path | None:
        return self._review_images.get(place_id)


def _ensure_ros_environment() -> None:
    """Re-exec once with ROS libraries while retaining the Habitat interpreter."""
    if os.environ.get("LINGBOT_8083_ROS_READY") == "1":
        return
    setup = "source /opt/ros/humble/setup.bash"
    install_setup = ROOT / "ros2_ws/install/setup.bash"
    if install_setup.is_file():
        setup += f" && source {shlex.quote(str(install_setup))}"
    command = shlex.join([sys.executable, *sys.argv])
    shell = (
        f"{setup} && export LINGBOT_8083_ROS_READY=1 && "
        "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
        "${LD_PRELOAD:+:$LD_PRELOAD} && exec " + command
    )
    os.execv("/bin/bash", ["/bin/bash", "-lc", shell])


def _start_nav2(
    map_yaml: Path,
    robot_radius: float,
    start_pose: dict[str, float],
) -> subprocess.Popen:
    command = [
        "ros2",
        "launch",
        "lingbot_semantic_nav_ros",
        "lingbot_map_wheel_nav2.launch.py",
        f"map:={map_yaml.resolve()}",
        f"robot_radius:={robot_radius}",
        f"initial_x:={start_pose['x']}",
        f"initial_y:={start_pose['y']}",
        f"initial_yaw:={start_pose['yaw']}",
    ]
    parent_pid = os.getpid()

    def terminate_with_dashboard() -> None:
        libc = ctypes.CDLL(None)
        if libc.prctl(1, signal.SIGTERM) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGTERM)

    return subprocess.Popen(
        command,
        start_new_session=True,
        preexec_fn=terminate_with_dashboard,
    )


def _stop_nav2(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _acquire_dashboard_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    stream = (output / ".dashboard.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RuntimeError(f"Another 8083 dashboard is using {output}") from exc
    return stream

def main() -> int:
    replica = ROOT / "data/habitat_assets/versioned_data/replica_cad_dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--pointcloud", type=Path, required=True)
    parser.add_argument("--semantic-map", type=Path, required=True)
    parser.add_argument("--instance-map", type=Path, required=True)
    parser.add_argument("--region-map", type=Path, required=True)
    parser.add_argument("--region-catalog", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--initial-rgb", type=Path, required=True)
    parser.add_argument("--scene-label", default="ReplicaCAD apt_1 · RGB-only")
    parser.add_argument(
        "--habitat-scene",
        default="apt_1",
        help="textured Habitat scene used only by the live RGB camera",
    )
    parser.add_argument(
        "--habitat-scene-dataset-config",
        type=Path,
        default=replica / "replicaCAD.scene_dataset_config.json",
    )
    parser.add_argument(
        "--renderer-alignment",
        type=Path,
        help=(
            "render-only LingBot-to-Habitat Sim(2) JSON; defaults beside map.yaml"
        ),
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-sensor-height", type=float, default=1.0)
    parser.add_argument("--camera-hfov", type=float, default=90.0)
    parser.add_argument("--robot-radius", type=float, default=0.06)
    parser.add_argument("--max-snap-distance", type=float, default=0.75)
    parser.add_argument("--animation-delay", type=float, default=0.025)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument(
        "--navigation-backend",
        choices=("nav2", "preview"),
        default="nav2",
        help="Nav2 differential-wheel simulation or legacy A* preview",
    )
    parser.add_argument(
        "--llm-provider",
        choices=("rule", "deepseek", "openai"),
        default="rule",
    )
    parser.add_argument("--no-rule-fallback", action="store_true")
    parser.add_argument(
        "--ros-domain-id",
        type=int,
        default=32,
        help="isolate 8083 Nav2 topics from the 8081 process",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/replica_cad_mapping_dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8083)
    args = parser.parse_args()
    if args.renderer_alignment is None:
        args.renderer_alignment = (
            args.map_yaml.expanduser().resolve().parent
            / "habitat_render_alignment.json"
        )
    if (
        not 1 <= args.port <= 65535
        or args.robot_radius < 0
        or args.max_snap_distance <= 0
        or args.animation_delay < 0
        or args.realtime_factor <= 0
        or args.camera_width <= 0
        or args.camera_height <= 0
        or args.camera_sensor_height < 0
        or not 10 <= args.camera_hfov < 180
        or not 0 <= args.ros_domain_id <= 232
    ):
        parser.error("invalid navigation or Habitat RGB renderer configuration")
    os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    if args.navigation_backend == "nav2":
        _ensure_ros_environment()
    nav2_process = None
    dashboard_lock = None
    try:
        dashboard_lock = _acquire_dashboard_lock(args.output.resolve())
        session = MappingViewSession(args)
        if args.navigation_backend == "nav2":
            session.start_wheel_runtime()
            nav2_process = _start_nav2(
                args.map_yaml,
                args.robot_radius,
                session.start_pose(),
            )
            time.sleep(0.5)
            if nav2_process.poll() is not None:
                raise RuntimeError("Nav2 launch exited during 8083 startup")
        server = ThreadingHTTPServer((args.host, args.port), MappingHandler)
        server.session = session
    except Exception as exc:
        _stop_nav2(nav2_process)
        if dashboard_lock is not None:
            dashboard_lock.close()
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"RGB-only Nav2 wheel dashboard ready: http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to stop the server.")
    def request_shutdown(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGHUP, request_shutdown)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        server.server_close()
        _stop_nav2(nav2_process)
        if dashboard_lock is not None:
            dashboard_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
