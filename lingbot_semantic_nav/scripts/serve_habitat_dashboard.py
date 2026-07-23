#!/usr/bin/env python3
"""Serve a command-driven live Habitat navigation dashboard."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lingbot_nav.place_db import PlaceDatabase, normalize_label  # noqa: E402
from lingbot_nav.models import Place  # noqa: E402
from lingbot_nav.intent import create_intent_parser  # noqa: E402
from lingbot_nav.mapping.exploration import (  # noqa: E402
    build_exploration_topology,
    retarget_outward_regions,
)
from lingbot_nav.sim.habitat_dashboard import HabitatNavigationSession  # noqa: E402
from lingbot_nav.topology import TopologyEdge, TopologyGraph  # noqa: E402


WEB_ROOT = ROOT / "habitat_dashboard"


class Handler(BaseHTTPRequestHandler):
    server_version = "LingBotHabitatDashboard/1.0"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        static = {
            "/": (WEB_ROOT / "index.html", "text/html; charset=utf-8"),
            "/app.js": (WEB_ROOT / "app.js", "text/javascript; charset=utf-8"),
            "/styles.css": (WEB_ROOT / "styles.css", "text/css; charset=utf-8"),
            "/map_layers.css": (WEB_ROOT / "map_layers.css", "text/css; charset=utf-8"),
            "/object.css": (WEB_ROOT / "object.css", "text/css; charset=utf-8"),
        }
        if path in static:
            file_path, content_type = static[path]
            return self._send_bytes(file_path.read_bytes(), content_type)
        if path == "/api/config":
            return self._send_json(self.server.session.config())
        if path == "/api/state":
            return self._send_json(self.server.session.snapshot())
        if path == "/asset/topdown.png":
            return self._send_file(self.server.session.topdown_path, "image/png")
        if path.startswith("/asset/map/") and path.endswith(".png"):
            layer_id = path.removeprefix("/asset/map/").removesuffix(".png")
            asset = self.server.session.map_asset(layer_id)
            if asset is not None:
                return self._send_file(asset, "image/png")
        if path == "/asset/camera.png":
            frame = self.server.session.latest_rgb()
            if frame is not None:
                return self._send_file(frame, "image/png")
        if path == "/stream/camera.mjpg":
            return self._send_camera_stream()
        if path.startswith("/asset/object-review/") and path.endswith(".png"):
            place_id = path.removeprefix("/asset/object-review/").removesuffix(".png")
            review = self.server.session.review_image(place_id, ROOT)
            if review is not None:
                return self._send_file(review, "image/png")
        self.send_error(HTTPStatus.NOT_FOUND, "Resource not found")

    def _send_camera_stream(self) -> None:
        """Push the newest JPEG over one persistent multipart response."""
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
        if path == "/api/cancel":
            return self._send_json(self.server.session.cancel())
        if path != "/api/command":
            self.send_error(HTTPStatus.NOT_FOUND, "Resource not found")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 1 <= size <= 4096:
                raise ValueError("请求内容长度无效")
            payload = json.loads(self.rfile.read(size))
            state = self.server.session.submit(str(payload.get("command", "")))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        self._send_json(state)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            self._send_bytes(path.read_bytes(), content_type)
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "Asset not found")

    def _send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(),
            "application/json; charset=utf-8",
            status,
        )

    def _send_bytes(
        self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
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


def parse_args() -> argparse.Namespace:
    assets = ROOT / "data/habitat_assets/versioned_data/habitat_test_scenes"
    parser = argparse.ArgumentParser(description="Interactive Habitat navigation dashboard")
    parser.add_argument("--scene", type=Path, default=assets / "apartment_1.glb")
    parser.add_argument("--places", type=Path, default=ROOT / "data/habitat/apartment_1_route_places.json")
    parser.add_argument("--topology", type=Path, default=ROOT / "data/habitat/apartment_1_route_topology.json")
    parser.add_argument("--topology-start", default="habitat_start")
    parser.add_argument(
        "--object-demo",
        action="store_true",
        help="use the full apartment region and object-instance catalog",
    )
    parser.add_argument(
        "--replica-cad-apt1-demo",
        action="store_true",
        help="use ReplicaCAD apt_1 with LingBot RGB-only maps and Nav2",
    )
    parser.add_argument("--scene-dataset-config", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/habitat_dashboard")
    parser.add_argument("--step-delay", type=float, default=0.08)
    parser.add_argument("--control-mode", choices=("continuous", "discrete"), default="continuous")
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--map-yaml", type=Path, help="aligned ROS occupancy map")
    parser.add_argument("--pointcloud", type=Path, help="aligned RGB PLY point cloud")
    parser.add_argument("--semantic-map", type=Path, help="aligned semantic_map.npy")
    parser.add_argument("--instance-map", type=Path, help="aligned instance_map.npy")
    parser.add_argument("--region-map", type=Path, help="aligned region_map.npy")
    parser.add_argument("--region-catalog", type=Path, help="recognized semantic regions and Nav2 goal poses")
    parser.add_argument("--initial-rgb", type=Path, help="RGB frame shown before navigation starts")
    parser.add_argument("--show-habitat-gt", action="store_true", help="render navmesh only as a comparison layer")
    parser.add_argument("--map-unit-to-sim-meter", type=float)
    parser.add_argument(
        "--navigation-backend",
        choices=("nav2", "legacy"),
        default=None,
        help="motion backend; defaults to Nav2",
    )
    parser.add_argument(
        "--llm-provider",
        choices=("rule", "deepseek", "openai"),
        default="rule",
        help="online language parser; use deepseek/openai for open-ended paraphrases",
    )
    parser.add_argument(
        "--no-rule-fallback",
        action="store_true",
        help="fail instead of using deterministic aliases when the remote parser is unavailable",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--ros-domain-id",
        type=int,
        help="isolate this dashboard's Nav2 graph from other running demos",
    )
    return parser.parse_args()


def _ensure_ros_environment() -> None:
    """Re-exec once with ROS libraries while retaining the Habitat interpreter."""
    if os.environ.get("LINGBOT_HABITAT_ROS_READY") == "1":
        return
    install_setup = ROOT / "ros2_ws/install/setup.bash"
    setup = "source /opt/ros/humble/setup.bash"
    if install_setup.is_file():
        setup += f" && source {shlex.quote(str(install_setup))}"
    command = shlex.join([sys.executable, *sys.argv])
    shell = (
        f"{setup} && export LINGBOT_HABITAT_ROS_READY=1 && "
        "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
        "${LD_PRELOAD:+:$LD_PRELOAD} && exec " + command
    )
    os.execv("/bin/bash", ["/bin/bash", "-lc", shell])


def _start_nav2(map_yaml: Path, start_pose) -> subprocess.Popen:
    command = [
        "ros2", "launch", "lingbot_semantic_nav_ros", "habitat_nav2.launch.py",
        f"map:={map_yaml.resolve()}",
        f"initial_x:={start_pose.x}",
        f"initial_y:={start_pose.y}",
        f"initial_yaw:={start_pose.yaw}",
    ]
    parent_pid = os.getpid()

    def terminate_with_dashboard() -> None:
        # Keep the Nav2 launch session from surviving a forced dashboard exit.
        # PR_SET_PDEATHSIG is Linux-specific, as are the ROS/Habitat runtime
        # assumptions made by this demo.
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
    """Terminate the whole launch group, including orphaned ROS children."""
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
    """Prevent two dashboard-owned Nav2 graphs from sharing global topics."""
    output.mkdir(parents=True, exist_ok=True)
    stream = (output / ".dashboard.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RuntimeError(
            f"Another Habitat dashboard is already using {output}"
        ) from exc
    return stream


def _expand_object_demo_catalog(
    places: PlaceDatabase,
    topology: TopologyGraph,
    candidates_path: Path,
    topology_start: str,
    *,
    include_all_candidates: bool = False,
) -> tuple[PlaceDatabase, TopologyGraph]:
    """Enable geometrically valid instance candidates as explicit demo targets."""
    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "unverified_place_candidates":
        raise ValueError(f"Unsupported object candidate catalog: {candidates_path}")
    if payload.get("geometry", {}).get("backend") != "lingbot_map":
        raise ValueError("Object candidates must use LingBot-Map geometry")
    existing = {place.place_id for place in places.places}
    start = places.resolve(topology_start).place.entrance_pose
    eligible = []
    for item in payload.get("instances", []):
        recognized = (
            bool(item.get("candidate_poses"))
            and (
                include_all_candidates
                or (
                    int(item.get("observation_count", 0)) >= 3
                    and float(item.get("mean_detection_score", 0.0)) >= 0.28
                    and float(item.get("mean_mask_score", 0.0)) >= 0.80
                )
            )
        )
        if item.get("instance_id") not in existing and recognized:
            eligible.append(item)
    reserved_labels = {
        str(place.metadata.get("semantic_label", ""))
        for place in places.places
        if place.metadata.get("semantic_label")
    }
    primary_by_label: dict[str, str] = {}
    for item in eligible:
        pose = item["candidate_poses"][0]
        label = str(item.get("semantic_label", ""))
        if label in reserved_labels:
            continue
        distance = math.hypot(float(pose["x"]) - start.x, float(pose["y"]) - start.y)
        previous = next(
            (candidate for candidate in eligible if candidate.get("instance_id") == primary_by_label.get(label)),
            None,
        )
        if previous is None:
            primary_by_label[label] = str(item["instance_id"])
        else:
            previous_pose = previous["candidate_poses"][0]
            previous_distance = math.hypot(
                float(previous_pose["x"]) - start.x,
                float(previous_pose["y"]) - start.y,
            )
            if distance < previous_distance:
                primary_by_label[label] = str(item["instance_id"])

    expanded = list(places.places)
    reserved_aliases = {
        normalize_label(alias)
        for place in places.places
        for alias in (place.name, *place.aliases)
    }
    added_ids: list[str] = []
    for item in eligible:
        instance_id = str(item["instance_id"])
        label = str(item.get("semantic_label", ""))
        pose = item["candidate_poses"][0]
        # Only the nearest instance keeps generic aliases such as “椅子”.
        # Other instances retain numbered aliases, avoiding ambiguous commands.
        aliases = [
            str(alias) for alias in item.get("aliases", [])
            if normalize_label(str(alias)) not in reserved_aliases
            and (
                instance_id == primary_by_label.get(label)
                or any(character.isdigit() for character in str(alias))
            )
        ]
        if instance_id not in aliases:
            aliases.append(instance_id)
        expanded.append(
            Place.from_mapping(
                {
                    "id": instance_id,
                    "name": item.get("name", instance_id),
                    "aliases": aliases,
                    "entrance_pose": {
                        "x": pose["x"],
                        "y": pose["y"],
                        "yaw": pose["yaw"],
                    },
                    "region": f"object_instance_{instance_id}",
                    "metadata": {
                        "habitat_y": -1.60025,
                        "target_type": "navigable_object_instance",
                        "semantic_label": label,
                        "instance_id": instance_id,
                        "instance_center": item.get("center_map", {}),
                        "standoff_m": pose.get("standoff_m", 0.0),
                        "clearance_m": pose.get("clearance_m", 0.0),
                        "review_image": item.get("representative_review_image", ""),
                        "verification": {
                            "status": "demo_enabled",
                            "reviewer": "geometry-auto-selection",
                            "evidence": [
                                "user enabled candidate-object navigation",
                                "selected stop pose is free and satisfies recorded clearance/standoff",
                            ],
                        },
                        "provenance": {
                            "candidate_artifact": str(candidates_path),
                            "candidate_index": 0,
                            "geometry_backend": payload.get("geometry", {}).get("backend", ""),
                        },
                    },
                },
                places.frame_id,
            )
        )
        added_ids.append(instance_id)

    expanded_places = PlaceDatabase(expanded, places.frame_id)
    expanded_topology = TopologyGraph(
        (*topology.nodes, *added_ids),
        (*topology.edges, *(TopologyEdge(topology_start, item) for item in added_ids)),
        expanded_places,
    )
    return expanded_places, expanded_topology


def _expand_region_catalog(
    places: PlaceDatabase,
    topology: TopologyGraph,
    region_catalog_path: Path,
    topology_start: str,
) -> tuple[PlaceDatabase, TopologyGraph]:
    """Add RGB-recognized regions as goals in the same LingBot map frame."""
    payload = json.loads(region_catalog_path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "rgb_only_semantic_regions":
        raise ValueError(f"Unsupported region catalog: {region_catalog_path}")
    provenance = payload.get("provenance", {})
    prohibited = (
        "habitat_navmesh", "habitat_semantics", "habitat_depth", "habitat_camera_poses"
    )
    if any(provenance.get(key) is not False for key in prohibited):
        raise ValueError("Region catalog must explicitly exclude every Habitat ground-truth input")
    existing = {place.place_id for place in places.places}
    additions = [
        Place.from_mapping(item, places.frame_id)
        for item in payload.get("places", [])
        if str(item.get("id", "")) not in existing
    ]
    if not additions:
        return places, topology
    expanded_places = PlaceDatabase((*places.places, *additions), places.frame_id)
    added_ids = tuple(place.place_id for place in additions)
    expanded_topology = TopologyGraph(
        (*topology.nodes, *added_ids),
        (*topology.edges, *(TopologyEdge(topology_start, place_id) for place_id in added_ids)),
        expanded_places,
    )
    return expanded_places, expanded_topology


def _build_direct_nav2_topology(
    places: PlaceDatabase,
    topology_start: str,
) -> tuple[PlaceDatabase, TopologyGraph, dict[str, object]]:
    """Let Nav2, not incomplete semantic-mask adjacency, decide reachability."""
    annotated = []
    target_ids = []
    for place in places.places:
        value = place.to_dict()
        if place.place_id != topology_start and not place.metadata.get("internal"):
            metadata = dict(value["metadata"])
            metadata["exploration"] = {
                "source": "lingbot_occupancy_nav2_direct",
                "outside_start_region": True,
                "reachable_via_region_map": None,
                "habitat_ground_truth_used": False,
            }
            value["metadata"] = metadata
            target_ids.append(place.place_id)
        annotated.append(Place.from_mapping(value, places.frame_id))
    expanded = PlaceDatabase(annotated, places.frame_id)
    topology = TopologyGraph(
        tuple(place.place_id for place in annotated),
        tuple(TopologyEdge(topology_start, place_id) for place_id in target_ids),
        expanded,
    )
    return expanded, topology, {
        "source": "lingbot_occupancy_nav2_direct",
        "target_count": len(target_ids),
        "semantic_region_adjacency_used_for_planning": False,
        "habitat_ground_truth_used": False,
    }


def main() -> int:
    args = parse_args()
    simulation_start = None
    if args.object_demo:
        # Use the expanded catalog generated directly in the RGB-only map
        # frame. It preserves the executable occupancy route while increasing
        # recognition from one queried class to eight classes.
        args.topology_start = "lingbot_start"
        args.places = ROOT / "data/habitat/apartment_1_rgb_only_places.json"
        args.topology = ROOT / "data/habitat/apartment_1_rgb_only_topology.json"
        args.candidates = ROOT / "outputs/maps/apartment_1_rgb_only_instances_full/place_candidates.json"
        args.map_yaml = ROOT / "outputs/maps/apartment_1_rgb_only_blind/map.yaml"
        args.pointcloud = ROOT / "outputs/maps/apartment_1_rgb_only_blind/lingbot_local.ply"
        args.semantic_map = ROOT / "outputs/maps/apartment_1_rgb_only_instances_full/semantic_map.npy"
        args.region_map = ROOT / "outputs/maps/apartment_1_rgb_only_regions/region_map.npy"
        args.region_catalog = ROOT / "outputs/maps/apartment_1_rgb_only_regions/region_catalog.json"
        args.initial_rgb = ROOT / "data/habitat/apartment_1_minimal/rgb/000000.png"
        if args.map_unit_to_sim_meter is None:
            args.map_unit_to_sim_meter = 1.606478038418092
        if args.output == ROOT / "outputs/habitat_dashboard":
            args.output = ROOT / "outputs/habitat_object_dashboard"
        simulation_start = (5.9657526, -1.60025, -1.7387148)
    if args.replica_cad_apt1_demo:
        replica = ROOT / "data/habitat_assets/versioned_data/replica_cad_dataset"
        args.scene = "apt_1"
        args.scene_dataset_config = replica / "replicaCAD.scene_dataset_config.json"
        args.topology_start = "lingbot_start"
        args.places = ROOT / "data/habitat/replica_cad_apt1_rgb_only_places.json"
        args.topology = ROOT / "data/habitat/replica_cad_apt1_rgb_only_topology.json"
        args.candidates = ROOT / "outputs/maps/replica_cad_apt1_instances/place_candidates.json"
        args.map_yaml = ROOT / "outputs/maps/replica_cad_apt1_rgb_only_blind/map.yaml"
        args.pointcloud = ROOT / "outputs/maps/replica_cad_apt1_rgb_only_blind/lingbot_local.ply"
        args.semantic_map = ROOT / "outputs/maps/replica_cad_apt1_instances/semantic_map.npy"
        args.instance_map = ROOT / "outputs/maps/replica_cad_apt1_instances/instance_map.npy"
        args.region_map = ROOT / "outputs/maps/replica_cad_apt1_regions_v3/region_map.npy"
        args.region_catalog = ROOT / "outputs/maps/replica_cad_apt1_regions_v3/region_catalog.json"
        args.initial_rgb = ROOT / "data/habitat/replica_cad_apt1_rgb_only/rgb/000000.png"
        if args.map_unit_to_sim_meter is None:
            args.map_unit_to_sim_meter = 1.0
        if args.output == ROOT / "outputs/habitat_dashboard":
            args.output = ROOT / "outputs/replica_cad_nav2_dashboard"
        if args.ros_domain_id is None:
            args.ros_domain_id = 31
        # Renderer anchor for the first RGB acquisition viewpoint.  It is not
        # published to Nav2 and is never used for planning or target creation.
        simulation_start = (1.6627699137, 0.1193729192, 5.0400333405)
    if args.map_unit_to_sim_meter is None:
        args.map_unit_to_sim_meter = 1.0
    if args.navigation_backend is None:
        args.navigation_backend = "nav2"
    if args.ros_domain_id is not None:
        if not 0 <= args.ros_domain_id <= 232:
            print("error: ROS domain id must be between 0 and 232", file=sys.stderr)
            return 2
        os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    if args.navigation_backend == "nav2":
        _ensure_ros_environment()
    if not 1 <= args.port <= 65535 or args.step_delay < 0 or args.realtime_factor <= 0 or args.map_unit_to_sim_meter <= 0:
        print("error: invalid port or step delay", file=sys.stderr)
        return 2
    nav2_process = None
    dashboard_lock = None
    try:
        if args.navigation_backend == "nav2":
            if args.map_yaml is None:
                raise RuntimeError("Nav2 requires --map-yaml")
        dashboard_lock = _acquire_dashboard_lock(args.output.resolve())
        # This dashboard reproduces historical schema-v1 artifacts. Formal
        # navigation nodes intentionally keep allow_legacy=False.
        places = PlaceDatabase.load(args.places, allow_legacy=True)
        topology = TopologyGraph.load(args.topology, places)
        if args.region_catalog is not None:
            places, topology = _expand_region_catalog(
                places, topology, args.region_catalog, args.topology_start
            )
        if (args.object_demo or args.replica_cad_apt1_demo) and args.candidates is not None:
            places, topology = _expand_object_demo_catalog(
                places,
                topology,
                args.candidates,
                args.topology_start,
                include_all_candidates=args.replica_cad_apt1_demo,
            )
        exploration_summary = None
        if args.object_demo or args.replica_cad_apt1_demo:
            if args.map_yaml is None or args.region_map is None:
                raise RuntimeError("Object exploration requires LingBot occupancy and region maps")
            if args.candidates is None:
                raise RuntimeError("Object exploration requires RGB object evidence")
            retarget_summary = None
            if args.object_demo:
                candidate_payload = json.loads(
                    args.candidates.read_text(encoding="utf-8")
                )
                prediction_root = Path(
                    str(candidate_payload.get("geometry", {}).get("predictions", ""))
                ).expanduser()
                if any(prediction_root.glob("frame_*.npz")):
                    places, retarget_summary = retarget_outward_regions(
                        places,
                        map_yaml=args.map_yaml,
                        region_map_path=args.region_map,
                        candidates_path=args.candidates,
                        topology_start=args.topology_start,
                    )
                else:
                    retarget_summary = {
                        "source": "precomputed_rgb_only_region_catalog",
                        "prediction_frames_loaded": False,
                    }
            if args.replica_cad_apt1_demo:
                places, topology, exploration_summary = _build_direct_nav2_topology(
                    places, args.topology_start
                )
            else:
                places, topology, exploration_summary = build_exploration_topology(
                    places,
                    map_yaml=args.map_yaml,
                    region_map_path=args.region_map,
                    topology_start=args.topology_start,
                )
            if retarget_summary is not None:
                exploration_summary["goal_refinement"] = retarget_summary
        intent_parser = create_intent_parser(
            args.llm_provider,
            places,
            allow_rule_fallback=not args.no_rule_fallback,
        )
        session = HabitatNavigationSession(
            args.scene,
            places,
            topology,
            args.topology_start,
            args.output,
            step_delay=args.step_delay,
            control_mode=args.control_mode,
            realtime_factor=args.realtime_factor,
            candidates_path=args.candidates,
            map_yaml=args.map_yaml,
            pointcloud_path=args.pointcloud,
            semantic_map_path=args.semantic_map,
            instance_map_path=args.instance_map,
            region_map_path=args.region_map,
            region_catalog_path=args.region_catalog,
            initial_rgb_path=args.initial_rgb,
            show_habitat_gt=args.show_habitat_gt,
            simulation_start=simulation_start,
            scene_dataset_config=args.scene_dataset_config,
            map_unit_to_sim_meter=args.map_unit_to_sim_meter,
            navigation_backend=args.navigation_backend,
            allow_unknown_navigation=(args.object_demo or args.replica_cad_apt1_demo),
            exploration_summary=exploration_summary,
            intent_parser=intent_parser,
        )
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        server.session = session
        if args.navigation_backend == "nav2":
            nav2_process = _start_nav2(
                args.map_yaml, places.resolve(args.topology_start).place.entrance_pose
            )
            time.sleep(0.25)
            if nav2_process.poll() is not None:
                raise RuntimeError("Nav2 launch exited during dashboard startup")
    except Exception as exc:
        _stop_nav2(nav2_process)
        if dashboard_lock is not None:
            dashboard_lock.close()
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Habitat dashboard ready: http://127.0.0.1:{args.port}")
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
        session.cancel()
        server.server_close()
        _stop_nav2(nav2_process)
        if dashboard_lock is not None:
            dashboard_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
