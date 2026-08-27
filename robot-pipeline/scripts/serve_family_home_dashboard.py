#!/usr/bin/env python3
"""Serve fail-closed navigation from the scanned family-home map bundle."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "family_home_dashboard"
DEFAULT_ARTIFACTS = ROOT / "outputs/family_home_vln"
DEFAULT_OUTPUT = ROOT / "outputs/family_home_web"
MAX_MAP_BUNDLE_BYTES = 1024 * 1024 * 1024
MAX_MAP_BUNDLE_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
MAX_MAP_BUNDLE_FILES = 20_000
sys.path.insert(0, str(ROOT))

from family_home_vln.layout import (  # noqa: E402
    ALL_DEMO_PLACES,
    ROBOT_RADIUS_M,
    SCENE_NAME,
    START_POSE,
)
from family_home_vln.household_objects import (  # noqa: E402
    DEMO_GRASP_OBJECTS,
    OBJECT_SET_SIGNATURE,
)
from family_home_vln.intent import FamilyIntentResolver  # noqa: E402
from family_home_vln.task_intent import FamilyTaskIntentResolver  # noqa: E402
from simple_room_vln.artifacts import load_lingbot_artifacts  # noqa: E402
from simple_room_vln.core import path_length, resolve_place  # noqa: E402
from g1d_dual_brain_agent.planner import (  # noqa: E402
    compile_family_home_command,
    compile_family_home_selection,
)


OBJECT_LABELS = {
    "living_room_sofa": ("沙发", "客厅"),
    "bedroom_bed": ("床", "卧室"),
    "dining_area": ("餐桌", "餐区"),
    "kitchen_counter": ("厨房操作台", "厨房"),
    "living_room_center": ("客厅中央", "客厅"),
    "media_cabinet_front": ("电视柜", "客厅"),
    "dining_table_south": ("餐桌南侧", "餐区"),
    "kitchen_entrance": ("厨房入口", "通行区"),
    "bedroom_entrance": ("卧室入口", "通行区"),
}

# These are scene-reviewed functional descriptions supplied to the constrained
# LLM catalog, not coordinate hints or a second navigation map.
PLACE_FUNCTIONS = {
    "living_room_sofa": ["休息", "坐下", "会客", "看电视"],
    "bedroom_bed": ["睡觉", "睡眠", "休息", "躺下"],
    "dining_area": ["吃饭", "用餐", "喝水"],
    "kitchen_counter": ["做饭", "准备食物", "厨房操作"],
    "living_room_center": ["活动", "站在客厅中央", "客厅中间"],
    "media_cabinet_front": ["看电视", "拿遥控器", "电视柜"],
    "dining_table_south": ["餐桌另一侧", "用餐", "取餐具"],
    "kitchen_entrance": ["进入厨房", "厨房门口", "准备做饭"],
    "bedroom_entrance": ["进入卧室", "卧室门口", "准备睡觉"],
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class FamilyHomeDashboardSession:
    def __init__(self, args: argparse.Namespace) -> None:
        self.artifacts = args.artifacts.resolve()
        self.output = args.output.resolve()
        self.live_dir = self.output / "live"
        self.output.mkdir(parents=True, exist_ok=True)
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.map_yaml = self.artifacts / "lingbot_map/map.yaml"
        self.places_json = self.artifacts / "places_formal.json"
        self.objects_json = self.artifacts / "objects_formal.json"
        self.summary_path = self.artifacts / "mapping_summary.json"
        missing = [
            path
            for path in (
                self.map_yaml,
                self.places_json,
                self.objects_json,
                self.summary_path,
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "家庭网页拒绝使用 bootstrap；请先运行 ./mobilemanibench.sh "
                "home-survey 和 home-map。缺少：" + ", ".join(str(path) for path in missing)
            )
        self.summary = _read_json(self.summary_path)
        self.place_catalog = _read_json(self.places_json)
        self.object_catalog = _read_json(self.objects_json)
        self.formal_bundle_current = (
            self.place_catalog.get("map", {}).get("household_object_set_signature")
            == OBJECT_SET_SIGNATURE
        )
        self.grid, self.places = load_lingbot_artifacts(
            self.map_yaml, self.places_json, robot_radius_m=ROBOT_RADIUS_M
        )
        if self.formal_bundle_current:
            self._build_demo_catalogs()
            self.grid, self.places = load_lingbot_artifacts(
                self.map_yaml, self.places_json, robot_radius_m=ROBOT_RADIUS_M
            )
        raw_assets = self.summary.get("assets", {})
        required = {"rgb_pointcloud", "semantic", "occupancy", "region"}
        if not required <= set(raw_assets):
            raise ValueError(
                "家庭网页要求正式 pointcloud/semantic/occupancy/region 四层，缺少："
                + ", ".join(sorted(required - set(raw_assets)))
            )
        self.map_assets = {
            key: self._bundle_path(
                raw_assets[key], self.artifacts / "map_preview" / f"{key}.png"
            )
            for key in required
        }
        missing_assets = [path for path in self.map_assets.values() if not path.is_file()]
        if missing_assets:
            raise FileNotFoundError(
                "mapping_summary 引用的图层不存在：" + ", ".join(str(path) for path in missing_assets)
            )
        self.map_asset_payloads = {
            key: path.read_bytes() for key, path in self.map_assets.items()
        }
        self.recognition = self._build_recognition_report()
        self._places_by_id = {place.place_id: place for place in self.places}
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._log_stream = None
        self._started_at: float | None = None
        self._last_command = ""
        # CLI startup supplies its explicit provider. Programmatic/test sessions
        # without that option keep the deterministic catalog parser.
        self._intent_provider = getattr(args, "intent_provider", "rule")
        self._no_rule_fallback = bool(
            getattr(args, "no_rule_fallback", False)
        )
        self._intent_resolver = getattr(args, "intent_resolver", None)
        self._task_intent_resolver = getattr(args, "task_intent_resolver", None)
        self._last_intent_resolution: dict | None = None
        self._events: deque[dict] = deque(maxlen=200)
        self._record_event("system", f"地图包已加载：{self.artifacts}")

    def _bundle_path(self, raw: str | Path | None, fallback: Path) -> Path:
        """Resolve relocatable bundle references, including stale absolute paths."""
        if raw:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = self.artifacts / candidate
            candidate = candidate.resolve()
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                # Imported manifests can retain an absolute path from another
                # server (for example /root/autodl-tmp). Fall through to the
                # relocatable paths inside this bundle.
                pass
            conventional = (self.artifacts / "map_preview" / Path(raw).name).resolve()
            if conventional.is_file():
                return conventional
            matches = [path.resolve() for path in self.artifacts.rglob(Path(raw).name)]
            if len(matches) == 1 and matches[0].is_file():
                return matches[0]
        return fallback.resolve()

    def _record_event(self, kind: str, message: str, **details: object) -> None:
        self._events.append(
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "kind": kind,
                "message": message,
                "details": details,
            }
        )

    def _build_demo_catalogs(self) -> None:
        """Create clearly labeled demo overrides without changing formal files."""

        layout_by_id = {place.place_id: place for place in ALL_DEMO_PLACES}
        demo_places = json.loads(json.dumps(self.place_catalog))
        existing_place_ids = {
            str(item.get("id", "")) for item in demo_places.get("places", [])
        }
        for place in ALL_DEMO_PLACES:
            if place.place_id in existing_place_ids:
                continue
            demo_places.setdefault("places", []).append(
                {
                    "id": place.place_id,
                    "name": place.name,
                    "aliases": list(place.aliases),
                    "status": "rejected",
                    "entrance_pose": None,
                    "target": {
                        "type": "runtime_demo_region",
                        "source_id": place.place_id,
                    },
                    "review": {
                        "status": "pending_runtime_occupancy_check",
                        "reviewer": "family_home_multi_place_extension",
                    },
                }
            )
        for item in demo_places.get("places", []):
            metadata = item.setdefault("metadata", {})
            metadata.setdefault(
                "functional_descriptions",
                PLACE_FUNCTIONS.get(str(item.get("id", "")), []),
            )
            if item.get("status") == "approved":
                metadata["availability"] = "formal_approved"
                continue
            place = layout_by_id.get(str(item.get("id", "")))
            if place is None:
                continue
            docking_xy = self._nearest_reachable_demo_pose(
                place.pose.x, place.pose.y
            )
            if docking_xy is None:
                continue
            pose = {
                "x": docking_xy[0],
                "y": docking_xy[1],
                "yaw": place.pose.yaw,
                "frame_id": "map",
            }
            item["status"] = "approved"
            item["entrance_pose"] = pose
            item["docking_candidates"] = [
                {
                    "id": "web_demo_occupancy_validated",
                    "pose": pose,
                    "checks": {
                        "clearance_m": ROBOT_RADIUS_M,
                        "footprint_radius_m": ROBOT_RADIUS_M,
                        "occupancy_status": "free",
                        "reachable": True,
                    },
                    "review": {
                        "status": "accepted",
                        "reviewer": "family_home_web_demo_occupancy_check",
                    },
                }
            ]
            item["selected_docking_candidate"] = "web_demo_occupancy_validated"
            metadata.update(
                {
                    "availability": "provisional_demo",
                    "formal_review_status": "rejected",
                    "provisional_reason": (
                        "网页演示临时开放；停靠点已在正式 occupancy 上验证可达"
                    ),
                }
            )

        demo_objects = json.loads(json.dumps(self.object_catalog))
        anchors = self._optional_json(
            self.artifacts / "semantic/semantic_metadata.json"
        ).get("anchors", {})
        approved_by_label = {
            str(item.get("source_label", "")).casefold(): item
            for item in demo_objects.get("objects", [])
            if item.get("status") == "approved"
        }
        for alias, canonical in (("sofa", "couch"), ("mug", "coffee cup")):
            target = approved_by_label.get(canonical)
            if target is not None and alias not in target.setdefault("aliases", []):
                target["aliases"].append(alias)
        # Remove historical broad aliases that made "bowl", "box" or "can"
        # resolve to the original cup after additional grasp props were added.
        coffee_cup = approved_by_label.get("coffee cup")
        if coffee_cup is not None:
            coffee_cup["aliases"] = [
                alias
                for alias in coffee_cup.get("aliases", [])
                if str(alias).casefold()
                in {"coffee cup", "杯子", "水杯", "cup", "mug"}
            ]

        # Map the audited simulator/table frame into the existing LingBot map
        # using the reviewed original cup correspondence.  New items remain
        # explicit runtime-demo entries; they do not rewrite formal scan data.
        original = next(
            (
                item
                for item in demo_objects.get("objects", [])
                if item.get("object_id") == "scan_coffee_cup_05"
            ),
            None,
        )
        if original is not None:
            original.setdefault("name", "咖啡杯")
            original.setdefault("home_place_id", "dining_area")
            original.setdefault("support_fixture_id", "dining_table")
            original.setdefault(
                "support_prim_path", "/World/FamilyHome/dining_table"
            )
            original.setdefault("support_height_above_floor_m", 0.76)
            original_anchor = original["map_position"]
            sim_to_map_x = float(original_anchor["x"]) - 1.78
            sim_to_map_y = float(original_anchor["y"]) - 3.02
            original_center_z = 0.76 + (0.044990 - (-0.034010)) / 2.0
            center_z_offset = float(original_anchor["z"]) - original_center_z
            known_ids = {
                str(item.get("object_id", ""))
                for item in demo_objects.get("objects", [])
            }
            for fixture in DEMO_GRASP_OBJECTS:
                if fixture.catalog_id in known_ids:
                    continue
                anchor_x = fixture.position_xy[0] + sim_to_map_x
                anchor_y = fixture.position_xy[1] + sim_to_map_y
                anchor_z = (
                    fixture.support_height_above_floor_m
                    + (fixture.maximum_xyz[1] - fixture.minimum_xyz[1]) / 2.0
                    + center_z_offset
                )
                bearing = float(fixture.preferred_view_bearing_rad)
                nominal_approach_x = anchor_x + math.cos(bearing) * 0.68
                nominal_approach_y = anchor_y + math.sin(bearing) * 0.68
                nominal_approach_yaw = math.atan2(
                    anchor_y - nominal_approach_y,
                    anchor_x - nominal_approach_x,
                )
                demo_objects.setdefault("objects", []).append(
                    {
                        "object_id": fixture.catalog_id,
                        "name": fixture.catalog_name,
                        "source_label": fixture.evaluation_label,
                        "aliases": list(fixture.catalog_aliases),
                        "status": "approved",
                        "availability": "runtime_grasp_demo",
                        "object_class": "portable_cup",
                        "manipulation_ready": True,
                        "home_place_id": fixture.home_place_id,
                        "support_fixture_id": fixture.support_fixture_id,
                        "support_prim_path": (
                            f"/World/FamilyHome/{fixture.support_fixture_id}"
                        ),
                        "support_height_above_floor_m": (
                            fixture.support_height_above_floor_m
                        ),
                        "map_position": {
                            "x": anchor_x,
                            "y": anchor_y,
                            "z": anchor_z,
                            "frame_id": "map",
                            "source": "reviewed_sim_support_pose_aligned_to_lingbot_map",
                        },
                        "approach": {
                            "pose": {
                                "x": nominal_approach_x,
                                "y": nominal_approach_y,
                                "yaw": nominal_approach_yaw,
                                "frame_id": "map",
                            },
                            "stand_off_m": 0.68,
                            "visibility_pose": {
                                "x": fixture.visibility_pose[0],
                                "y": fixture.visibility_pose[1],
                                "yaw": fixture.visibility_pose[2],
                                "frame_id": "map",
                            },
                            "visibility_standoff_m": (
                                fixture.visibility_standoff_m
                            ),
                            "alignment_tolerance_m": 0.06,
                            "faces_object_anchor": True,
                            "preferred_view_bearing_rad": bearing,
                            "view_bearing_source": "reviewed_support_side_alignment",
                            "footprint_radius_m": ROBOT_RADIUS_M,
                        },
                        "review": {
                            "status": "accepted_for_runtime_grasp_demo",
                            "reviewer": "family_home_multi_object_extension",
                            "reason": "cup geometry uses the verified Expert grasp contract",
                        },
                    }
                )
        provisional_labels = {"bowl", "chair", "bench", "desk"}
        for item in demo_objects.get("objects", []):
            label = str(item.get("source_label", "")).casefold()
            if item.get("status") == "approved":
                item["availability"] = "formal_approved"
                continue
            anchor = anchors.get(label)
            if label not in provisional_labels or not isinstance(anchor, list):
                continue
            item.update(
                {
                    "object_id": "demo_" + label.replace(" ", "_"),
                    "status": "approved",
                    "availability": "provisional_search_only",
                    "object_class": "demo_search_target",
                    "map_position": {
                        "x": float(anchor[0]),
                        "y": float(anchor[1]),
                        "z": 0.5,
                        "frame_id": "map",
                        "source": "sam3_map_anchor_provisional_demo",
                    },
                    "approach": {
                        "stand_off_m": 0.9,
                        "visibility_standoff_m": 1.1,
                        "alignment_tolerance_m": 0.12,
                        "preferred_view_bearing_rad": None,
                    },
                    "manipulation_ready": False,
                }
            )
            item.setdefault("review", {})["demo_override"] = (
                "临时开放实时搜索；未开放抓取，不代表正式审核通过"
            )

        self.places_json = self.output / "places_web_demo.json"
        self.objects_json = self.output / "objects_web_demo.json"
        self.places_json.write_text(
            json.dumps(demo_places, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.objects_json.write_text(
            json.dumps(demo_objects, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.place_catalog = demo_places
        self.object_catalog = demo_objects

    def _nearest_reachable_demo_pose(
        self, x: float, y: float
    ) -> tuple[float, float] | None:
        """Snap a requested region pose to the nearest reachable formal free cell."""

        center_row, center_col = self.grid.world_to_cell(x, y)
        for radius in range(0, 31):
            candidates: list[tuple[float, tuple[float, float]]] = []
            for row in range(center_row - radius, center_row + radius + 1):
                for col in range(center_col - radius, center_col + radius + 1):
                    if max(abs(row - center_row), abs(col - center_col)) != radius:
                        continue
                    cell = (row, col)
                    if not self.grid.is_free(cell):
                        continue
                    candidate = self.grid.cell_to_world(cell)
                    try:
                        self.grid.plan(
                            (START_POSE.x, START_POSE.y), candidate
                        )
                    except ValueError:
                        continue
                    candidates.append(
                        (math.dist((x, y), candidate), candidate)
                    )
            if candidates:
                candidates.sort(key=lambda item: item[0])
                return candidates[0][1]
        return None

    def _optional_json(self, path: Path | None) -> dict:
        if path is None or not path.is_file():
            return {}
        try:
            return _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    def _input_path(self, key: str, fallback: Path) -> Path:
        raw = self.summary.get("inputs", {}).get(key)
        return self._bundle_path(raw, fallback)

    def _build_recognition_report(self) -> dict:
        survey = self._optional_json(
            self._input_path("survey_manifest", self.artifacts / "survey/capture_manifest.json")
        )
        discovery = self._optional_json(
            self._input_path(
                "discovery", self.artifacts / "discovery/object_discovery.json"
            )
        )
        sam3 = self._optional_json(
            self._input_path("sam3", self.artifacts / "sam3/sam3_manifest.json")
        )
        observations = self._optional_json(
            self._input_path(
                "semantic_observations",
                self.artifacts / "semantic/sam3_observations.json",
            )
        )
        semantic_metadata = self._optional_json(
            self.artifacts / "semantic/semantic_metadata.json"
        )
        object_catalog = self.object_catalog
        reviewed_by_prompt = {
            str(item.get("source_label", "")).casefold(): item
            for item in object_catalog.get("objects", [])
        }
        raw_by_prompt = {
            str(item.get("prompt", "")).casefold(): item
            for item in sam3.get("prompts", [])
        }
        evidence_by_prompt: dict[str, list[dict]] = {}
        for item in observations.get("observations", []) if self.formal_bundle_current else []:
            prompt = str(item.get("prompt", "")).casefold()
            if (
                float(item.get("score", 0.0)) >= 0.35
                and int(item.get("point_count", 0)) >= 30
            ):
                evidence_by_prompt.setdefault(prompt, []).append(item)

        navigation_by_prompt: dict[str, dict] = {}
        for place in self.place_catalog.get("places", []):
            prompts = {
                str(place.get("target", {}).get("source_id", "")).casefold(),
                *(
                    str(item).casefold()
                    for item in place.get("metadata", {}).get("discovered_labels", [])
                ),
            }
            for prompt in prompts - {""}:
                navigation_by_prompt[prompt] = place

        objects = []
        for candidate in discovery.get("objects", []):
            prompt = str(candidate.get("label", "")).strip()
            prompt_key = prompt.casefold()
            raw = raw_by_prompt.get(prompt_key, {})
            evidence = evidence_by_prompt.get(prompt_key, [])
            anchor = semantic_metadata.get("anchors", {}).get(prompt)
            place = navigation_by_prompt.get(prompt_key)
            reviewed = reviewed_by_prompt.get(prompt_key, {})
            object_approved = bool(
                self.formal_bundle_current
                and reviewed.get("status") == "approved"
            )
            provisional_search = (
                reviewed.get("availability") == "provisional_search_only"
            )
            approved = bool(
                self.formal_bundle_current
                and place
                and place.get("status") == "approved"
            )
            scores = [float(item.get("score", 0.0)) for item in evidence]
            objects.append(
                {
                    "id": prompt_key.replace(" ", "_"),
                    "name": prompt,
                    "prompt": prompt,
                    "label_source": "Florence-2 RGB 自主生成",
                    "recognized": True,
                    "mapped": bool(evidence),
                    "searchable": object_approved,
                    "object_id": reviewed.get("object_id", ""),
                    "manipulation_ready": bool(
                        reviewed.get("manipulation_ready", False)
                    ),
                    "navigable": approved,
                    "status": (
                        "approved"
                        if approved
                        else "provisional_search_only"
                        if provisional_search
                        else "approved_for_search_and_alignment"
                        if object_approved
                        else "rejected_by_review"
                        if reviewed.get("status") == "rejected"
                        else "mapped_not_navigable"
                        if evidence
                        else "discovered_not_mapped"
                    ),
                    "discovery_frame_occurrences": int(
                        candidate.get("frame_occurrences", 0)
                    ),
                    "raw_detections": int(candidate.get("raw_detection_count", 0)),
                    "sam3_detections": int(raw.get("detections", 0)),
                    "map_observations": len(evidence),
                    "track_count": len(
                        {str(item.get("track_id", "")) for item in evidence}
                    ),
                    "mean_score": sum(scores) / len(scores) if scores else None,
                    "anchor_xy": anchor,
                    "region_id": (
                        place.get("metadata", {}).get("region_id") if place else None
                    ),
                    "review_reason": (
                        "临时开放实时搜索；尚未开放抓取"
                        if provisional_search
                        else ""
                        if approved
                        else str(reviewed.get("review", {}).get("reason", ""))
                        if reviewed
                        else "已形成地图语义，但没有对应的审核导航地点"
                        if evidence
                        else "已由 RGB 自主发现，但尚未形成合格的 SAM3 map-frame 证据"
                    ),
                    "availability": reviewed.get("availability", "unavailable"),
                }
            )

        scenes = []
        for place in self.place_catalog.get("places", []):
            place_id = str(place.get("id", ""))
            prompt = str(place.get("target", {}).get("source_id", "")).strip()
            prompt_key = prompt.casefold()
            evidence = evidence_by_prompt.get(prompt_key, [])
            recognized = bool(evidence)
            availability = place.get("metadata", {}).get(
                "availability", "formal_approved"
            )
            object_name, scene_name = OBJECT_LABELS.get(
                place_id, (place.get("name", prompt), "未分类区域")
            )
            scenes.append(
                {
                    "id": place_id,
                    "name": scene_name,
                    "status": (
                        "provisional_open"
                        if availability == "provisional_demo"
                        else "confirmed"
                        if recognized
                        else "surveyed_unconfirmed"
                    ),
                    "availability": availability,
                    "evidence": (
                        f"由“{object_name}”的 {len(evidence)} 条 map-frame 证据确认"
                        if recognized
                        else f"巡检路径已覆盖，但“{object_name}”没有形成 map-frame 语义证据"
                    ),
                }
            )

        frame_count = len(survey.get("frames", []))
        mapped_count = sum(item["mapped"] for item in objects)
        approved_count = sum(item["navigable"] for item in objects)
        searchable_count = sum(item["searchable"] for item in objects)
        return {
            "source": (
                "G1-D RGB 巡检 → Florence-2 无类别清单自主发现 → "
                "LingBot RGB-only → SAM3.1 → map-frame 审核"
            ),
            "formal_bundle_current": self.formal_bundle_current,
            "truth_boundary": discovery.get("truth_boundary", {}),
            "survey": {
                "frame_count": frame_count,
                "resolution": survey.get("camera", {}).get("resolution"),
                "rgb_only_model_input": survey.get("rgb_is_only_model_input"),
            },
            "summary": {
                "discovered_categories": len(objects),
                "mapped_categories": mapped_count,
                "approved_destinations": approved_count,
                "approved_search_objects": searchable_count,
                "semantic_regions": len(
                    semantic_metadata.get("region_labels", {})
                ),
            },
            "objects": objects,
            "scenes": scenes,
        }

    def config(self) -> dict:
        map_metadata = self.summary["map"]
        bounds = map_metadata["bounds"]
        layer_descriptions = {
            item["id"]: item.get("description", "")
            for item in map_metadata.get("layers", [])
        }
        layer_keys = (
            ("pointcloud", "rgb_pointcloud", "Point Cloud"),
            ("semantic", "semantic", "Semantic"),
            ("occupancy", "occupancy", "Occupancy"),
            ("region", "region", "Region"),
        )
        return {
            "schema_version": 1,
            "scene": SCENE_NAME,
            "mode": "isaac_family_home",
            "map": {
                "width": int(map_metadata["width"]),
                "height": int(map_metadata["height"]),
                "resolution": float(map_metadata["resolution"]),
                "flip_y": bool(map_metadata.get("flip_y", True)),
                "bounds": {
                    "min_x": float(bounds["min_x"]),
                    "max_x": float(bounds["max_x"]),
                    "min_y": float(bounds["min_z"]),
                    "max_y": float(bounds["max_z"]),
                },
                "source_status": (
                    "formal" if self.formal_bundle_current else "stale"
                ),
                "source_label": (
                    "G1-D 扫描 · LingBot / SAM3 正式制品"
                    if self.formal_bundle_current
                    else "已完成自主发现 · 正式四层等待重建"
                ),
                "formal_bundle_detected": self.formal_bundle_current,
                "robot_radius_m": ROBOT_RADIUS_M,
            },
            "layers": [
                {
                    "id": public_id,
                    "label": label,
                    "status": (
                        "formal" if self.formal_bundle_current else "stale"
                    ),
                    "asset": (
                        f"/asset/map/{asset_id}.png"
                        f"?v={self.map_assets[asset_id].stat().st_mtime_ns}"
                    ),
                    "description": layer_descriptions.get(asset_id, ""),
                }
                for public_id, asset_id, label in layer_keys
            ],
            "places": [
                {
                    "id": place.place_id,
                    "name": place.name,
                    "aliases": list(place.aliases),
                    "pose": asdict(place.pose),
                    "example": f"请带我到{place.name}",
                    "availability": next(
                        (
                            item.get("metadata", {}).get(
                                "availability", "formal_approved"
                            )
                            for item in self.place_catalog.get("places", [])
                            if item.get("id") == place.place_id
                        ),
                        "formal_approved",
                    ),
                }
                for place in self.places
                if self.formal_bundle_current
            ],
            "recognition": self.recognition,
            "bundle": {
                "root": str(self.artifacts),
                "name": self.artifacts.name,
                "import_supported": True,
                "format": "zip",
            },
            "intent": {
                "provider": self._intent_provider,
                "rule_fallback_enabled": not self._no_rule_fallback,
                "catalog_constrained": True,
                "model_generates_coordinates": False,
            },
            "dual_brain": {
                "enabled": True,
                "orchestrator": "g1d_dual_brain_agent.DualBrainExecutive",
                "navigation_brain": "VLN + reviewed occupancy/semantic map",
                "manipulation_brain": "OpenVLA-OFT adapter",
                "shared_memory": "dual_agent_world_memory.json",
            },
            "manipulation": {
                "policy": "OpenVLA-OFT G1-D v14",
                "inference_enabled": True,
                "action_chunk": 8,
                "execution_backend": "bounded_world_delta_dls_ik_simulation",
                "direct_vla_joint_control": True,
                "deployment_status": "simulation_experimental_real_robot_deferred",
            },
            "camera_stream": "/stream/camera.mjpg",
            "robot_camera_stream": "/stream/robot-camera.mjpg",
            "examples": [
                "请带我去餐厅，拿杯子，再回到客厅沙发旁",
                "请带我到客厅沙发旁",
            ],
        }

    def map_data(self) -> dict:
        return {
            "schema_version": 1,
            "source": "g1d_rgb_survey+lingbot_rgb_only+sam3.1",
            "truth_boundary": None,
            "map_yaml": str(self.map_yaml),
            "places": str(self.places_json),
        }

    def log_snapshot(self, max_lines: int = 240) -> dict:
        log_path = self.output / "isaac.log"
        lines: list[str] = []
        try:
            with log_path.open("rb") as stream:
                stream.seek(0, 2)
                size = stream.tell()
                stream.seek(max(0, size - 128 * 1024))
                text = stream.read().decode("utf-8", "replace")
                lines = text.splitlines()[-max_lines:]
        except OSError:
            pass
        state = self.snapshot()
        return {
            "schema_version": 1,
            "events": list(self._events),
            "lines": lines,
            "current": {
                "state": state.get("state"),
                "action": state.get("action"),
                "message": state.get("message"),
                "frame": state.get("frame", 0),
            },
        }

    def _idle_state(self) -> dict:
        return {
            "schema_version": 1,
            "state": "idle",
            "message": "输入导航或拿取返回任务后启动 Isaac Sim。",
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
        if process is None and state.get("state") in {"starting", "loading", "running"}:
            state["state"] = "failed"
            state["process_running"] = False
            state["message"] = (
                "Dashboard 已重启，无法继续跟踪先前的 Isaac 任务；"
                "请查看日志后重新提交任务。"
            )
        if process is not None and not running and state.get("state") in {
            "starting",
            "loading",
            "running",
        }:
            state["state"] = "failed"
            state["message"] = (
                f"Isaac Sim 进程在启动/执行期间退出，返回码 {process.returncode}。"
                "请查看下方任务日志。"
            )
        return state

    def plan(self, command: str):
        if not self.formal_bundle_current:
            raise ValueError(
                "家庭物品版本已变化；请先运行 home-map 重建并审核正式四层，当前禁止导航"
            )
        # Exact catalog labels stay deterministic and offline.  Free-form
        # functional descriptions (e.g. "可以睡觉的地方") are delegated to
        # DeepSeek, constrained to catalog IDs only.
        try:
            target = resolve_place(command, self.places)
            resolution = None
        except ValueError as exact_error:
            try:
                resolver = self._intent_resolver
                if resolver is None:
                    resolver = FamilyIntentResolver(
                        self.places_json,
                        provider=self._intent_provider,
                        allow_rule_fallback=not self._no_rule_fallback,
                    )
                    self._intent_resolver = resolver
                semantic = resolver.resolve(command)
                target = self._places_by_id.get(semantic.place_id)
                if target is None:
                    raise ValueError(
                        f"大模型返回了当前地图不存在的地点 ID：{semantic.place_id}"
                    )
                resolution = {
                    "parser": semantic.parser,
                    "confidence": semantic.confidence,
                    "intent": semantic.intent,
                    "place_id": semantic.place_id,
                }
            except Exception as semantic_error:
                raise ValueError(
                    f"{exact_error}；DeepSeek 语义匹配也未找到对应地点：{semantic_error}"
                ) from semantic_error
        path = self.grid.plan(
            (START_POSE.x, START_POSE.y),
            (target.pose.x, target.pose.y),
        )
        self._last_intent_resolution = resolution
        return target, path

    def interpret(self, command: str) -> dict:
        """Resolve a reviewed navigation or go-pick-return mission."""
        if self._intent_provider == "deepseek":
            resolver = self._task_intent_resolver
            if resolver is None:
                resolver = FamilyTaskIntentResolver(
                    self.place_catalog,
                    self.object_catalog,
                    allow_rule_fallback=not self._no_rule_fallback,
                )
                self._task_intent_resolver = resolver
            resolution = resolver.resolve(command)
            if resolution.task_type == "go_pick_return":
                mission = compile_family_home_selection(
                    command,
                    outbound_place_id=resolution.outbound_place_id,
                    object_id=resolution.object_id,
                    return_place_id=resolution.return_place_id,
                    places_catalog=self.place_catalog,
                    objects_catalog=self.object_catalog,
                    mission_id="family-home-dashboard-preview",
                )
                outbound = self._places_by_id[resolution.outbound_place_id]
                return_place = self._places_by_id[resolution.return_place_id]
                first = self.grid.plan(
                    (START_POSE.x, START_POSE.y),
                    (outbound.pose.x, outbound.pose.y),
                )
                second = self.grid.plan(
                    (outbound.pose.x, outbound.pose.y),
                    (return_place.pose.x, return_place.pose.y),
                )
                return {
                    "mode": "dual_brain_task",
                    "task": mission.mission_id,
                    "target_name": f"{resolution.object_id} → {return_place.name}",
                    "path": first + second[1:],
                    "mission": mission.to_dict(),
                    "intent_resolution": {
                        "parser": resolution.parser,
                        "confidence": resolution.confidence,
                        "task_type": resolution.task_type,
                        "outbound_place_id": resolution.outbound_place_id,
                        "object_id": resolution.object_id,
                        "return_place_id": resolution.return_place_id,
                    },
                    "steps": [
                        "NAVIGATE", "SEARCH_OBJECT", "APPROACH_AND_ALIGN",
                        "OPENVLA_PICK", "VERIFY", "RETURN",
                    ],
                }
            if (
                resolution.task_type != "vln_navigation"
                or not resolution.destination_place_id
            ):
                # Preserve the dedicated single-place DeepSeek resolver used
                # by the navigation UI and older integrations.
                target, path = self.plan(command)
                semantic = self._last_intent_resolution or {}
                return {
                    "mode": "vln_navigation",
                    "task": target.place_id,
                    "target_name": target.name,
                    "path": path,
                    "steps": ["NAVIGATE", "ARRIVE"],
                    "intent_resolution": semantic,
                }
            target = self._places_by_id[resolution.destination_place_id]
            path = self.grid.plan(
                (START_POSE.x, START_POSE.y),
                (target.pose.x, target.pose.y),
            )
            return {
                "mode": "vln_navigation",
                "task": target.place_id,
                "target_name": target.name,
                "path": path,
                "steps": ["NAVIGATE", "ARRIVE"],
                "availability": next(
                    (
                        item.get("metadata", {}).get(
                            "availability", "formal_approved"
                        )
                        for item in self.place_catalog.get("places", [])
                        if item.get("id") == target.place_id
                    ),
                    "formal_approved",
                ),
                "intent_resolution": {
                    "parser": resolution.parser,
                    "confidence": resolution.confidence,
                    "task_type": resolution.task_type,
                    "place_id": resolution.destination_place_id,
                },
            }
        compound = any(marker in command for marker in ("拿", "取", "抓"))
        if compound:
            mission = compile_family_home_command(
                command,
                places_catalog=self.place_catalog,
                objects_catalog=self.object_catalog,
                mission_id="family-home-dashboard-preview",
            )
            outbound = self._places_by_id[mission.goals[0].instruction]
            return_place = self._places_by_id[mission.goals[2].instruction]
            first = self.grid.plan(
                (START_POSE.x, START_POSE.y),
                (outbound.pose.x, outbound.pose.y),
            )
            second = self.grid.plan(
                (outbound.pose.x, outbound.pose.y),
                (return_place.pose.x, return_place.pose.y),
            )
            return {
                "mode": "dual_brain_task",
                "task": mission.mission_id,
                "target_name": f"{mission.goals[1].target_id} → {return_place.name}",
                "path": first + second[1:],
                "mission": mission.to_dict(),
                "steps": [
                    "NAVIGATE",
                    "SEARCH_OBJECT",
                    "APPROACH_AND_ALIGN",
                    "OPENVLA_PICK",
                    "VERIFY",
                    "RETURN",
                ],
            }
        target, path = self.plan(command)
        return {
            "mode": "vln_navigation",
            "task": target.place_id,
            "target_name": target.name,
            "path": path,
            "steps": ["NAVIGATE", "ARRIVE"],
            "availability": next(
                (
                    item.get("metadata", {}).get(
                        "availability", "formal_approved"
                    )
                    for item in self.place_catalog.get("places", [])
                    if item.get("id") == target.place_id
                ),
                "formal_approved",
            ),
            "intent_resolution": self._last_intent_resolution,
        }

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
        interpretation = self.interpret(command)
        self._record_event(
            "command",
            f"收到文本任务：{command}",
            mode=interpretation["mode"],
            task=interpretation["task"],
        )
        path = interpretation["path"]
        # The LLM result has already been validated against the catalog.  The
        # low-level VLN runner intentionally accepts only catalog labels, so
        # pass its resolved ID rather than asking it to re-parse free-form text.
        execution_command = (
            command
            if interpretation["mode"] == "dual_brain_task"
            else interpretation["task"]
        )
        with self._lock:
            kit_pids = self._other_kit_processes()
            if kit_pids:
                raise ValueError(
                    "检测到其他 Isaac Kit 进程："
                    + ", ".join(str(pid) for pid in kit_pids)
                    + "；请先停止其他仿真实例"
                )
            for name in ("state.json", "camera.jpg", "robot_camera.jpg", "mission.json"):
                stale = self.live_dir / name
                if stale.exists():
                    stale.unlink()
            initial = self._idle_state()
            initial.update(
                {
                    "state": "starting",
                    "message": (
                        "Agent 已拆解导航—搜索—对齐—拿取—验证—返回，"
                        "正在启动 Isaac Sim…"
                        if interpretation["mode"] == "dual_brain_task"
                        else (
                            f"指令匹配{'临时 DEMO 区域' if interpretation.get('availability') == 'provisional_demo' else '审核地点'}"
                            f"“{interpretation['target_name']}”，正在启动 Isaac Sim…"
                        )
                    ),
                    "command": command,
                    "task": interpretation["task"],
                    "target_name": interpretation["target_name"],
                    "mission_mode": interpretation["mode"],
                    "mission_steps": interpretation["steps"],
                    "intent_resolution": interpretation.get("intent_resolution"),
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
            isaac_python = Path(
                os.environ.get(
                    "G1D_ISAAC_PYTHON",
                    str(ROOT / "isaacsim/python.sh"),
                )
            )
            if not isaac_python.is_file():
                raise ValueError(f"Isaac Python launcher 不存在：{isaac_python}")
            argv = [
                str(isaac_python),
                str(ROOT / "run_g1d_simple_room_vln.py"),
                "--scene-profile",
                "family-home",
                "--headless",
                "--test",
                "--command",
                execution_command,
                "--output-dir",
                str(self.artifacts),
                "--map",
                str(self.map_yaml),
                "--places",
                str(self.places_json),
                "--live-dir",
                str(self.live_dir),
                "--live-fps",
                "15",
                "--live-resolution",
                "800x450",
                # Keep the completed state visible long enough for the
                # browser polling loop to render the arrival confirmation.
                "--arrival-hold-seconds",
                "3",
            ]
            if interpretation["mode"] == "dual_brain_task":
                mission_path = self.live_dir / "mission.json"
                mission_path.write_text(
                    json.dumps(
                        interpretation["mission"], ensure_ascii=False, indent=2
                    ) + "\n",
                    encoding="utf-8",
                )
                argv.extend(
                    [
                        "--dual-agent",
                        "--family-task",
                        "--mission-json",
                        str(mission_path),
                        "--openvla",
                        "--openvla-model",
                        str(ROOT / "checkpoints/openvla-oft-libero-combined"),
                        "--openvla-python",
                        str(ROOT / ".conda/envs/openvla-oft/bin/python"),
                        "--openvla-adapter",
                        str(
                            ROOT
                            / "checkpoints/openvla-oft-g1d-v14/g1d-family-home-oft/lora_adapter"
                        ),
                        "--openvla-action-head",
                        str(
                            ROOT
                            / "checkpoints/openvla-oft-g1d-v14/g1d-family-home-oft/action_head--latest_checkpoint.pt"
                        ),
                        "--openvla-dataset-statistics",
                        str(
                            ROOT
                            / "checkpoints/openvla-oft-g1d-v14/g1d-family-home-oft/dataset_statistics.json"
                        ),
                        "--openvla-unnorm-key",
                        "g1d_family_home_pick",
                        "--execute-openvla-actions",
                        "--objects",
                        str(self.objects_json),
                        "--live-search-frames",
                        "5",
                    ]
                )
            # Navigation always keeps the calibrated G1-D RGB camera alive:
            # the dashboard publishes it alongside the external overview.
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
            self._record_event("process", "Isaac Sim 任务进程已启动", pid=self._process.pid)
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
                self._record_event("process", "用户请求停止当前任务", pid=process.pid)
        return self.snapshot()

    def wait_for_camera(
        self, previous_mtime: int, *, stream: str = "overview"
    ) -> tuple[int, bytes | None]:
        filenames = {"overview": "camera.jpg", "robot": "robot_camera.jpg"}
        try:
            target = self.live_dir / filenames[stream]
        except KeyError as exc:
            raise ValueError(f"unsupported live camera stream: {stream}") from exc
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

    def latest_camera(self, *, stream: str = "overview") -> bytes | None:
        """Return only the most recent encoded frame; never queue old video."""
        filenames = {"overview": "camera.jpg", "robot": "robot_camera.jpg"}
        try:
            return (self.live_dir / filenames[stream]).read_bytes()
        except (KeyError, OSError):
            return None


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
        if path == "/api/logs":
            return self._send_json(self.server.session.log_snapshot())
        if path.startswith("/asset/map/") and path.endswith(".png"):
            layer_id = Path(path).stem
            body = self.server.session.map_asset_payloads.get(layer_id)
            if body is not None:
                return self._send_bytes(
                    body,
                    "image/png",
                    cache_control="public, max-age=3600, immutable",
                )
        if path == "/stream/camera.jpg":
            jpeg = self.server.session.latest_camera()
            if jpeg is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Camera frame is not ready")
                return
            return self._send_bytes(
                jpeg, "image/jpeg", cache_control="no-store, no-cache, must-revalidate"
            )
        if path == "/stream/camera.mjpg":
            return self._send_camera_stream()
        if path == "/stream/robot-camera.jpg":
            jpeg = self.server.session.latest_camera(stream="robot")
            if jpeg is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Robot camera frame is not ready")
                return
            return self._send_bytes(
                jpeg, "image/jpeg", cache_control="no-store, no-cache, must-revalidate"
            )
        if path == "/stream/robot-camera.mjpg":
            return self._send_camera_stream(stream="robot")
        self.send_error(HTTPStatus.NOT_FOUND, "Resource not found")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/cancel":
                return self._send_json(self.server.session.cancel())
            if path == "/api/map-bundle":
                return self._receive_map_bundle()
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

    def _receive_map_bundle(self) -> None:
        session = self.server.session
        if session.snapshot().get("process_running"):
            raise ValueError("任务运行期间不能切换地图，请先停止任务")
        if self.headers.get("Content-Type", "").split(";", 1)[0] not in {
            "application/zip",
            "application/octet-stream",
        }:
            raise ValueError("地图包必须是 ZIP 文件")
        size = int(self.headers.get("Content-Length", "0"))
        if not 1 <= size <= MAX_MAP_BUNDLE_BYTES:
            raise ValueError("地图包大小无效或超过 1 GiB")
        imports_root = session.output / "imported_maps"
        imports_root.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(tempfile.mkdtemp(prefix="import-", dir=imports_root))
        archive = temporary_root / "map-bundle.zip"
        remaining = size
        digest = hashlib.sha256()
        with archive.open("wb") as stream:
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("地图包上传不完整")
                stream.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
        extracted = temporary_root / "bundle"
        extracted.mkdir()
        try:
            self._extract_map_bundle(archive, extracted)
            archive.unlink()
            summaries = [
                path
                for path in extracted.rglob("mapping_summary.json")
                if "__MACOSX" not in path.parts
            ]
            if len(summaries) != 1:
                raise ValueError(
                    "ZIP 中必须且只能包含一份 mapping_summary.json；"
                    f"当前找到 {len(summaries)} 份"
                )
            artifact_root = summaries[0].parent
            args = argparse.Namespace(
                artifacts=artifact_root,
                output=session.output,
                intent_provider=session._intent_provider,
                no_rule_fallback=session._no_rule_fallback,
            )
            replacement = FamilyHomeDashboardSession(args)
            replacement._record_event(
                "map_import",
                f"网页导入地图包：{self.headers.get('X-File-Name', 'map-bundle.zip')}",
                sha256=digest.hexdigest(),
                bytes=size,
            )
            for name in ("state.json", "camera.jpg", "robot_camera.jpg", "mission.json"):
                stale = replacement.live_dir / name
                if stale.exists():
                    stale.unlink()
            with self.server.session_lock:
                if self.server.session.snapshot().get("process_running"):
                    raise ValueError("导入期间已有任务启动，地图未切换")
                self.server.session = replacement
            self._send_json(
                {
                    "ok": True,
                    "message": "地图与语义库导入完成",
                    "sha256": digest.hexdigest(),
                    "bundle_root": str(artifact_root),
                    "config": replacement.config(),
                },
                HTTPStatus.CREATED,
            )
        except (zipfile.BadZipFile, OSError) as exc:
            raise ValueError(f"地图 ZIP 无法读取：{exc}") from exc

    @staticmethod
    def _extract_map_bundle(archive: Path, destination: Path) -> None:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if len(members) > MAX_MAP_BUNDLE_FILES:
                raise ValueError("地图包文件数量超过 20000")
            expanded = sum(item.file_size for item in members)
            if expanded > MAX_MAP_BUNDLE_EXPANDED_BYTES:
                raise ValueError("地图包解压后超过 2 GiB")
            root = destination.resolve()
            for item in members:
                raw = item.filename.replace("\\", "/")
                if not raw or raw.startswith("/") or "\0" in raw:
                    raise ValueError(f"地图包包含非法路径：{raw!r}")
                target = (destination / raw).resolve()
                if target != root and root not in target.parents:
                    raise ValueError(f"地图包包含越界路径：{raw}")
                mode = (item.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise ValueError(f"地图包不允许符号链接：{raw}")
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, target.open("wb") as sink:
                    while chunk := source.read(1024 * 1024):
                        sink.write(chunk)

    def _send_camera_stream(self, *, stream: str = "overview") -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        mtime = -1
        try:
            while True:
                mtime, jpeg = self.server.session.wait_for_camera(mtime, stream=stream)
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
        *,
        cache_control: str = "no-store",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        if self.path.split("?", 1)[0] not in {
            "/api/state", "/api/logs", "/stream/camera.mjpg", "/stream/robot-camera.mjpg",
        }:
            super().log_message(fmt, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6012)
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
        help="DeepSeek 不可用时拒绝模糊指令，而不是退回别名匹配",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = FamilyHomeDashboardSession(args)
    server = ThreadingHTTPServer((args.host, args.port), FamilyHomeHandler)
    server.daemon_threads = True
    server.session = session
    server.session_lock = threading.RLock()
    print(f"Family-home dashboard: http://{args.host}:{args.port}")
    print(f"Map source: {session.config()['map']['source_label']}")
    destinations = (
        " / ".join(place.name for place in session.places)
        if session.formal_bundle_current
        else "none (formal bundle is stale)"
    )
    print("Approved destinations: " + destinations)
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
