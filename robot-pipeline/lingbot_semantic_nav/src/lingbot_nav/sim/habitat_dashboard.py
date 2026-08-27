"""Thread-safe live state for the Habitat browser dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
import math
from pathlib import Path
import threading
import time

from ..errors import ConfigurationError
from ..intent import IntentParser, RuleBasedIntentParser
from ..mission import MissionResolver
from ..place_db import PlaceDatabase
from ..topology import TopologyGraph
from .habitat_collector import _imports
from .habitat_continuous import HabitatContinuousConfig, run_habitat_continuous_route
from .habitat_nav2 import HabitatNav2Config, run_habitat_nav2_route
from .habitat_route import HabitatRouteConfig, run_habitat_route
from .map_views import render_mapping_views, summarize_region_map


class HabitatNavigationSession:
    def __init__(
        self,
        scene: str | Path,
        places: PlaceDatabase,
        topology: TopologyGraph,
        topology_start: str,
        output_root: Path,
        *,
        map_resolution: float = 0.04,
        step_delay: float = 0.08,
        control_mode: str = "continuous",
        realtime_factor: float = 1.0,
        candidates_path: Path | None = None,
        map_yaml: Path | None = None,
        pointcloud_path: Path | None = None,
        semantic_map_path: Path | None = None,
        instance_map_path: Path | None = None,
        region_map_path: Path | None = None,
        region_catalog_path: Path | None = None,
        initial_rgb_path: Path | None = None,
        show_habitat_gt: bool = False,
        simulation_start: tuple[float, float, float] | None = None,
        scene_dataset_config: Path | None = None,
        map_unit_to_sim_meter: float = 1.0,
        navigation_backend: str = "nav2",
        allow_unknown_navigation: bool = False,
        exploration_summary: dict[str, object] | None = None,
        intent_parser: IntentParser | None = None,
    ) -> None:
        scene_path = Path(scene).expanduser()
        self.scene: str | Path = scene_path.resolve() if scene_path.is_file() else str(scene)
        self.scene_dataset_config = (
            scene_dataset_config.expanduser().resolve()
            if scene_dataset_config is not None
            else None
        )
        self.places = places
        self.topology = topology
        self.topology_start = topology_start
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.step_delay = step_delay
        self.control_mode = control_mode
        self.realtime_factor = realtime_factor
        self.simulation_start = simulation_start
        self.map_unit_to_sim_meter = map_unit_to_sim_meter
        self.navigation_backend = navigation_backend
        self.allow_unknown_navigation = allow_unknown_navigation
        self.exploration_summary = dict(exploration_summary or {})
        self.object_candidates: list[dict[str, object]] = []
        self.detected_objects: list[dict[str, object]] = []
        if candidates_path is not None:
            payload = json.loads(candidates_path.read_text(encoding="utf-8"))
            executable = {place.place_id for place in places.places}
            for item in payload.get("instances", []):
                center = item.get("center_map", {})
                instance_id = item.get("instance_id", "")
                # The dashboard is an executable navigation catalog, not a
                # detector-debug dump.  Hide unrecognized/rejected instances
                # that have no promoted target in the current place database.
                if instance_id not in executable:
                    continue
                candidate = {
                    "id": instance_id,
                    "name": item.get("name", instance_id),
                    "semantic_label": item.get("semantic_label", ""),
                    "x": center.get("x"),
                    "y": center.get("y"),
                    "status": "navigable",
                    "navigable": True,
                    "observation_count": item.get("observation_count", 0),
                    "candidate_pose_count": len(item.get("candidate_poses", [])),
                }
                place = next(
                    (value for value in places.places if value.place_id == instance_id), None
                )
                if place is not None:
                    exploration = place.metadata.get("exploration", {})
                    candidate["region_id"] = exploration.get("region_id")
                    candidate["outside_start_region"] = bool(
                        exploration.get("outside_start_region", False)
                    )
                self.detected_objects.append(candidate)
        self._intent_parser = intent_parser or RuleBasedIntentParser(places)
        self._resolver = MissionResolver(
            self._intent_parser, places, topology, topology_start
        )
        self._lock = threading.RLock()
        self._camera_condition = threading.Condition(self._lock)
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._latest_rgb: Path | None = (
            initial_rgb_path.expanduser().resolve()
            if initial_rgb_path is not None and initial_rgb_path.is_file()
            else None
        )
        self._latest_jpeg: bytes | None = None
        if self._latest_rgb is not None:
            from PIL import Image

            preview = BytesIO()
            with Image.open(self._latest_rgb) as image:
                image.convert("RGB").save(preview, format="JPEG", quality=80)
            self._latest_jpeg = preview.getvalue()
        self._camera_sequence = int(self._latest_jpeg is not None)
        self._started_at: float | None = None
        start = places.resolve(topology_start).place.entrance_pose
        self._state: dict[str, object] = {
            "state": "idle",
            "message": "输入指令，让机器人在 Habitat 公寓中导航",
            "command": "",
            "destination": "",
            "destination_name": "",
            "route": [],
            "planned_trajectory": [],
            "exploration_trajectory": [],
            "trajectory": [{"x": start.x, "y": start.y, "yaw": start.yaw}],
            "pose": {"x": start.x, "y": start.y, "yaw": start.yaw},
            "action": "start",
            "collisions": 0,
            "frame": 0,
            "terminal": False,
            "controller": control_mode,
            "linear_velocity_mps": 0.0,
            "angular_velocity_rps": 0.0,
        }
        self.topdown_path = self.output_root / "topdown.png"
        self.map_yaml = map_yaml.expanduser().resolve() if map_yaml is not None else None
        self.detected_regions: list[dict[str, object]] = []
        reference_meta = None
        reference_path = None
        if show_habitat_gt:
            reference_meta = self._create_topdown_map(map_resolution)
            reference_path = self.topdown_path
        elif self.map_yaml is None:
            raise ConfigurationError(
                "A LingBot occupancy map is required; implicit Habitat GT fallback is disabled"
            )
        self.map_meta = reference_meta or {}
        self.map_assets: dict[str, Path] = (
            {"occupancy": self.topdown_path} if reference_meta is not None else {}
        )
        if self.map_yaml is not None:
            self.map_meta, self.map_assets = render_mapping_views(
                self.map_yaml,
                self.output_root / "map_layers",
                pointcloud_path=pointcloud_path,
                semantic_map_path=semantic_map_path,
                instance_map_path=instance_map_path,
                region_map_path=region_map_path,
                reference_meta=reference_meta,
                reference_occupancy_path=reference_path,
            )
            if region_map_path is not None:
                if region_catalog_path is not None:
                    payload = json.loads(
                        region_catalog_path.expanduser().resolve().read_text(encoding="utf-8")
                    )
                    self.detected_regions = list(payload.get("regions", []))
                else:
                    self.detected_regions = summarize_region_map(
                        region_map_path.expanduser().resolve(), self.map_yaml
                    )
        place_by_id = {place.place_id: place for place in self.places.places}
        for region in self.detected_regions:
            place = place_by_id.get(str(region.get("place_id", "")))
            if place is None:
                continue
            exploration = place.metadata.get("exploration", {})
            region["source_region_id"] = region.get("id")
            region["id"] = exploration.get("region_id", region.get("id"))
            region["x"] = place.entrance_pose.x
            region["y"] = place.entrance_pose.y
            region["object_evidence"] = list(
                place.metadata.get("object_evidence", region.get("object_evidence", []))
            )
            region["goal_refined_from_rgb"] = bool(place.metadata.get("goal_refinement"))
            region["outside_start_region"] = bool(
                exploration.get("outside_start_region", False)
            )
            region["region_hops_from_start"] = exploration.get("region_hops_from_start")
    def _create_topdown_map(self, resolution: float) -> dict[str, object]:

        habitat_sim, np, _quaternion, Image = _imports()
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(self.scene)
        if self.scene_dataset_config is not None:
            sim_cfg.scene_dataset_config_file = str(self.scene_dataset_config)
        with habitat_sim.Simulator(
            habitat_sim.Configuration(sim_cfg, [habitat_sim.agent.AgentConfiguration()])
        ) as sim:
            lower, upper = sim.pathfinder.get_bounds()
            height = float(
                self.places.resolve(self.topology_start).place.metadata["habitat_y"]
            )
            navigable = sim.pathfinder.get_topdown_view(resolution, height)
        pixels = np.where(navigable, 214, 25).astype(np.uint8)
        Image.fromarray(pixels, mode="L").save(self.topdown_path)
        return {
            "asset": "/asset/topdown.png",
            "width": int(pixels.shape[1]),
            "height": int(pixels.shape[0]),
            "layers": [
                {
                    "id": "occupancy",
                    "label": "Occupancy",
                    "asset": "/asset/topdown.png",
                    "description": "Habitat navmesh occupancy",
                }
            ],
            "flip_y": False,
            "resolution": resolution,
            "bounds": {
                "min_x": float(lower[0]),
                "max_x": float(upper[0]),
                "min_z": float(lower[2]),
                "max_z": float(upper[2]),
            },
        }

    def config(self) -> dict[str, object]:
        places = []
        outside_object_names = []
        inside_object_names = []
        outside_region_names = []
        inside_region_names = []
        for place in self.places.places:
            if place.metadata.get("internal"):
                continue
            value = place.to_dict()
            if place.metadata.get("review_image"):
                value["review_asset"] = f"/asset/object-review/{place.place_id}.png"
            if str(place.metadata.get("target_type", "")).endswith("object_instance"):
                target = outside_object_names if place.metadata.get("exploration", {}).get(
                    "outside_start_region"
                ) else inside_object_names
                target.append(place.name)
            elif (
                place.metadata.get("target_type") == "semantic_region"
                and place.place_id != self.topology_start
            ):
                target = outside_region_names if place.metadata.get("exploration", {}).get(
                    "outside_start_region"
                ) else inside_region_names
                target.append(place.name)
            places.append(value)
        examples = []
        if self.exploration_summary.get("exploration_goal_place_id"):
            examples.append("扫描房间外区域")
        examples.extend(f"请带我到{name}" for name in outside_region_names)
        examples.extend(f"请带我到{name}旁边" for name in outside_object_names[:3])
        examples.extend(f"请带我到{name}旁边" for name in inside_object_names[:1])
        examples.extend(f"请带我到{name}" for name in inside_region_names[:1])
        examples = list(dict.fromkeys(examples))[:7]
        if not examples:
            examples = [
                "请带我到公寓出口",
                "请带我到前台",
                "出门左转，经过前台，到达咖啡厅",
            ]
        return {
            "mode": "habitat",
            "control_mode": self.control_mode,
            "navigation_backend": self.navigation_backend,
            "intent_parser": self._intent_parser.name,
            "scene": Path(str(self.scene)).stem,
            "map": self.map_meta,
            "places": places,
            "object_candidates": self.object_candidates,
            "detected_objects": self.detected_objects,
            "detected_regions": self.detected_regions,
            "exploration": self.exploration_summary,
            "planning_inputs": {
                "planner": "nav2",
                "occupancy": "lingbot_map_rgb_only",
                "semantic_targets": "lingbot_map+owlv2+sam2+clipseg",
                "habitat_navmesh": False,
                "habitat_depth": False,
                "habitat_semantics": False,
                "habitat_camera_poses": False,
                "simulator_role": "rgb_renderer_and_cmd_vel_kinematics_only",
            },
            "examples": examples,
        }

    def submit(self, command: str) -> dict[str, object]:
        command = command.strip()
        if not command:
            raise ValueError("指令不能为空")
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise ValueError("机器人正在运动，请等待完成或先取消")
            try:
                preview_mission = self._resolver.resolve(command)
            except Exception as exc:
                raise ValueError(str(exc)) from exc
            target = preview_mission.place.to_dict()
            is_object = str(target["metadata"].get("target_type", "")).endswith(
                "object_instance"
            )
            self._cancel.clear()
            self._started_at = time.monotonic()
            start = self.places.resolve(self.topology_start).place.entrance_pose
            exploration_trajectory = [
                {"x": start.x, "y": start.y},
                *(
                    {"x": step.place.entrance_pose.x, "y": step.place.entrance_pose.y}
                    for step in preview_mission.steps
                ),
            ]
            leaves_start_region = bool(
                preview_mission.place.metadata.get("exploration", {}).get(
                    "outside_start_region", False
                )
            )
            self._state = {
                "state": "understanding",
                "message": "正在解析指令并使用 LingBot occupancy 规划",
                "command": command,
                "destination": "",
                "destination_name": "",
                "route": [],
                "planned_trajectory": [],
                "exploration_trajectory": exploration_trajectory,
                "trajectory": [{"x": start.x, "y": start.y, "yaw": start.yaw}],
                "pose": {"x": start.x, "y": start.y, "yaw": start.yaw},
                "action": "start",
                "collisions": 0,
                "frame": 0,
                "terminal": False,
                "controller": self.control_mode,
                "linear_velocity_mps": 0.0,
                "angular_velocity_rps": 0.0,
                "target": target,
                "pipeline": [
                    {"id": "language", "name": "解析物品/区域指令", "state": "done"},
                    {
                        "id": "instance",
                        "name": (
                            "匹配房间外物品实例" if is_object and leaves_start_region else
                            "匹配房间外语义区域" if leaves_start_region else
                            "匹配物品实例" if is_object else "匹配语义区域"
                        ),
                        "state": "done",
                    },
                    {
                        "id": "exploration",
                        "name": "沿 LingBot region 邻接链扫描房间外"
                        if leaves_start_region else "确认起始区域内路线",
                        "state": "done",
                    },
                    {
                        "id": "standoff",
                        "name": "读取安全停靠点" if is_object else "读取入口点",
                        "state": "done",
                    },
                    {
                        "id": "occupancy",
                        "name": "Nav2 全局规划（LingBot occupancy）"
                        if self.navigation_backend == "nav2"
                        else "LingBot occupancy A* 规划",
                        "state": "active",
                    },
                    {
                        "id": "motion",
                        "name": "Nav2 NavigateToPose" if self.navigation_backend == "nav2" else (
                            "连续速度控制" if self.control_mode == "continuous" else "离散轮式动作"
                        ),
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
                target=self._run, args=(command, output), daemon=True
            )
            self._worker.start()
            return self._snapshot_unlocked()

    def _run(self, command: str, output: Path) -> None:
        try:
            target = self._resolver.resolve(command).place
            if self.navigation_backend == "nav2":
                if self.simulation_start is None:
                    raise RuntimeError("Nav2 Habitat backend requires simulation_start")
                result = run_habitat_nav2_route(
                    HabitatNav2Config(
                        scene=self.scene,
                        output_dir=output,
                        instruction=command,
                        simulation_start=self.simulation_start,
                        scene_dataset_config=self.scene_dataset_config,
                        map_unit_to_sim_meter=self.map_unit_to_sim_meter,
                        realtime_factor=self.realtime_factor,
                    ),
                    self._resolver,
                    progress_callback=self._on_progress,
                    cancel_check=self._cancel.is_set,
                )
            elif self.control_mode == "continuous":
                result = run_habitat_continuous_route(
                    HabitatContinuousConfig(
                        scene=self.scene,
                        output_dir=output,
                        instruction=command,
                        realtime_factor=self.realtime_factor,
                        map_yaml=self.map_yaml,
                        simulation_start=self.simulation_start,
                        map_unit_to_sim_meter=self.map_unit_to_sim_meter,
                        unknown_is_occupied=not self.allow_unknown_navigation,
                        use_habitat_navmesh=False,
                    ),
                    self._resolver,
                    progress_callback=self._on_progress,
                    cancel_check=self._cancel.is_set,
                )
            else:
                result = run_habitat_route(
                    HabitatRouteConfig(
                        scene=self.scene,
                        output_dir=output,
                        instruction=command,
                        save_depth=False,
                        step_delay=self.step_delay,
                        map_yaml=self.map_yaml,
                        simulation_start=self.simulation_start,
                        map_unit_to_sim_meter=self.map_unit_to_sim_meter,
                        unknown_is_occupied=not self.allow_unknown_navigation,
                    ),
                    self._resolver,
                    progress_callback=self._on_progress,
                    cancel_check=self._cancel.is_set,
                )
            with self._lock:
                self._state["state"] = "arrived"
                self._state["message"] = (
                    f"已到达{self._state.get('destination_name') or '目标地点'}"
                )
                self._state["terminal"] = True
                self._state["actions"] = result.get("actions", result.get("control_steps", 0))
                self._state["collisions"] = result["collisions"]
                for stage in self._state.get("pipeline", []):
                    stage["state"] = "done"
        except Exception as exc:
            with self._lock:
                cancelled = self._cancel.is_set()
                self._state["state"] = "cancelled" if cancelled else "failed"
                self._state["message"] = "导航已取消" if cancelled else str(exc)
                self._state["terminal"] = True
                for stage in self._state.get("pipeline", []):
                    if stage["state"] == "active":
                        stage["state"] = "failed"

    def _on_progress(self, update: dict[str, object]) -> None:
        with self._lock:
            if update["kind"] == "planned":
                points = update["points"]
                self._state["planned_trajectory"] = [
                    {"x": float(point[0]), "y": float(point[2])} for point in points
                ]
                self._state["destination"] = update["destination"]
                self._state["destination_name"] = update["destination_name"]
                self._state["route"] = update["route"]
                self._state["controller"] = update.get("controller", self.control_mode)
                self._state["state"] = "navigating"
                self._state["message"] = (
                    "机器人正在执行 Nav2 / LingBot occupancy 路线"
                    if str(update.get("planner", "")).startswith("nav2")
                    else "机器人正在执行 LingBot occupancy 路线"
                )
                for stage in self._state.get("pipeline", []):
                    if stage["id"] == "occupancy":
                        stage["state"] = "done"
                    elif stage["id"] == "motion":
                        stage["state"] = "active"
                return
            sample = update["sample"]
            position = sample["position"]
            map_position = sample.get("map_position", [position[0], position[2]])
            rotation = sample["rotation_xyzw"]
            yaw = float(sample.get("map_yaw", math.atan2(
                2.0 * (rotation[3] * rotation[1] + rotation[0] * rotation[2]),
                1.0 - 2.0 * (rotation[1] ** 2 + rotation[2] ** 2),
            )))
            pose = {"x": float(map_position[0]), "y": float(map_position[1]), "yaw": yaw}
            trajectory = self._state["trajectory"]
            if sample["frame"] == 0:
                trajectory[:] = [pose]
            else:
                trajectory.append(pose)
            self._state["pose"] = pose
            self._state["action"] = sample["action"]
            self._state["linear_velocity_mps"] = float(sample.get("linear_velocity_mps", 0.0))
            self._state["angular_velocity_rps"] = float(sample.get("angular_velocity_rps", 0.0))
            self._state["frame"] = sample["frame"]
            self._state["collisions"] = int(self._state["collisions"]) + int(
                sample["collided"]
            )
            self._latest_rgb = Path(str(update["rgb_path"])).resolve()
            jpeg = update.get("jpeg_bytes")
            if isinstance(jpeg, bytes):
                self._latest_jpeg = jpeg
            self._camera_sequence += 1
            self._camera_condition.notify_all()

    def cancel(self) -> dict[str, object]:
        self._cancel.set()
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                self._state["message"] = "正在停止 Habitat 机器人…"
            return self._snapshot_unlocked()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, object]:
        value = dict(self._state)
        value["trajectory"] = list(self._state["trajectory"])
        value["planned_trajectory"] = list(self._state["planned_trajectory"])
        value["exploration_trajectory"] = list(
            self._state.get("exploration_trajectory", [])
        )
        value["route"] = list(self._state["route"])
        value["pipeline"] = [dict(item) for item in self._state.get("pipeline", [])]
        value["elapsed_sec"] = (
            0.0 if self._started_at is None else time.monotonic() - self._started_at
        )
        value["camera_url"] = (
            f"/asset/camera.png?sequence={self._camera_sequence}"
            if self._latest_rgb
            else None
        )
        if value["planned_trajectory"]:
            goal = value["planned_trajectory"][-1]
            pose = value["pose"]
            value["distance_remaining"] = math.hypot(
                float(goal["x"]) - float(pose["x"]),
                float(goal["y"]) - float(pose["y"]),
            )
        else:
            value["distance_remaining"] = None
        return value

    def latest_rgb(self) -> Path | None:
        with self._lock:
            return self._latest_rgb

    def wait_for_camera(
        self, after_sequence: int, timeout: float = 10.0
    ) -> tuple[int, bytes | None]:
        """Wait for one newer in-memory JPEG without polling the filesystem."""
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

    def map_asset(self, layer_id: str) -> Path | None:
        return self.map_assets.get(layer_id)

    def review_image(self, place_id: str, project_root: Path) -> Path | None:
        place = next(
            (item for item in self.places.places if item.place_id == place_id), None
        )
        if place is None or not place.metadata.get("review_image"):
            return None
        path = (project_root / str(place.metadata["review_image"])).resolve()
        try:
            path.relative_to(project_root.resolve())
        except ValueError:
            return None
        return path if path.is_file() else None
