"""ReplicaCAD household clutter used by the autonomous RGB discovery survey.

The names below are evaluation truth and asset-management metadata.  They are
never added to the USD semantic registry or passed to the perception model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLICA_CAD_ROOT = (
    ROOT
    / "lingbot_semantic_nav/data/habitat_assets/versioned_data"
    / "replica_cad_dataset"
)
PREPARED_ASSET_ROOT = ROOT / "outputs/family_home_assets"


@dataclass(frozen=True)
class HouseholdObject:
    object_id: str
    source_name: str
    evaluation_label: str
    position_xy: tuple[float, float]
    support_height_above_floor_m: float
    yaw_deg: float
    mass_kg: float
    # ReplicaCAD GLB files are Y-up. Bounds are source-space meters.
    minimum_xyz: tuple[float, float, float]
    maximum_xyz: tuple[float, float, float]
    dynamic: bool = False
    catalog_id: str = ""
    catalog_name: str = ""
    catalog_aliases: tuple[str, ...] = ()
    support_fixture_id: str = "dining_table"
    home_place_id: str = "dining_area"
    visibility_pose: tuple[float, float, float] = (1.735163245, 1.674295473, 1.8464061075)
    visibility_standoff_m: float = 1.1
    preferred_view_bearing_rad: float = -1.292969691181926

    @property
    def source_path(self) -> Path:
        return REPLICA_CAD_ROOT / "objects" / self.source_name

    @property
    def prepared_usd(self) -> Path:
        return PREPARED_ASSET_ROOT / self.object_id / f"{self.object_id}.usd"


FORMAL_SURVEY_OBJECTS = (
    HouseholdObject(
        "living_plant",
        "frl_apartment_indoor_plant_01.glb",
        "indoor plant",
        (-2.95, 1.15),
        0.0,
        20.0,
        4.00,
        (-0.256806, -0.509234, -0.243841),
        (0.260499, 0.981388, 0.246645),
    ),
    HouseholdObject(
        "bedside_lamp",
        "frl_apartment_lamp_01.glb",
        "lamp",
        (-2.48, -2.18),
        0.55,
        -15.0,
        2.00,
        (-0.226088, -0.344187, -0.211905),
        (0.229833, 0.277855, 0.214345),
    ),
    HouseholdObject(
        "media_monitor",
        "frl_apartment_monitor.glb",
        "monitor",
        (-0.20, 3.78),
        0.82,
        0.0,
        3.50,
        (-0.346054, -0.442339, -0.068302),
        (0.345627, 0.214150, 0.105224),
    ),
    HouseholdObject(
        "dining_basket",
        "frl_apartment_basket.glb",
        "basket",
        (2.80, 3.65),
        0.0,
        12.0,
        0.80,
        (-0.136252, -0.132411, -0.150227),
        (0.136229, 0.176479, 0.150212),
    ),
    HouseholdObject(
        "dining_cup",
        "frl_apartment_cup_01.glb",
        "cup",
        (1.78, 3.02),
        0.76,
        18.0,
        0.12,
        (-0.055642, -0.034010, -0.055580),
        (0.077712, 0.044990, 0.055434),
        True,
        "scan_coffee_cup_05",
        "咖啡杯",
        ("咖啡杯", "杯子", "coffee cup", "cup", "mug"),
    ),
    HouseholdObject(
        "dining_bowl",
        "frl_apartment_bowl_01.glb",
        "bowl",
        (2.24, 3.04),
        0.76,
        -12.0,
        4.50,
        (-0.126614, -0.066286, -0.126442),
        (0.126686, 0.058214, 0.126058),
    ),
    HouseholdObject(
        "bed_handbag",
        "frl_apartment_handbag.glb",
        "handbag",
        (-2.92, -2.18),
        0.55,
        -20.0,
        1.20,
        (-0.1525, -0.1278, -0.0895),
        (0.1643, 0.2696, 0.0871),
    ),
    HouseholdObject(
        "media_book",
        "frl_apartment_book_01.glb",
        "book",
        (-0.58, 3.64),
        0.82,
        8.0,
        0.85,
        (-0.074689, -0.089731, -0.013589),
        (0.072413, 0.089675, 0.013470),
    ),
    HouseholdObject(
        "media_remote",
        "frl_apartment_remote-control_01.glb",
        "remote control",
        (0.18, 3.94),
        0.82,
        72.0,
        0.10,
        (-0.023261, -0.006425, -0.088677),
        (0.023261, 0.006425, 0.088675),
    ),
    HouseholdObject(
        "kitchen_appliance",
        "frl_apartment_small_appliance_01.glb",
        "small kitchen appliance",
        (3.57, 3.78),
        0.92,
        12.0,
        2.00,
        (-0.1068, -0.1423, -0.1576),
        (0.1061, 0.2153, 0.2183),
    ),
    HouseholdObject(
        "kitchen_knife_block",
        "frl_apartment_knifeblock.glb",
        "knife block",
        (3.52, 4.08),
        0.92,
        -8.0,
        1.20,
        (-0.0476, -0.1266, -0.0769),
        (0.0473, 0.1735, 0.0720),
    ),
)

# Runtime grasp extensions are deliberately separate from the immutable RGB
# survey object set.  They are exposed by the Dashboard as clearly labelled
# demo objects with reviewed simulator poses, while the original LingBot map
# and its signature remain auditable and unchanged.
DEMO_GRASP_OBJECTS = (
    HouseholdObject(
        "dining_tall_mug",
        "frl_apartment_cup_02.glb",
        "tall mug",
        (1.48, 2.75),
        0.76,
        -18.0,
        0.22,
        (-0.054030, -0.046144, -0.053600),
        (0.075225, 0.055356, 0.053569),
        dynamic=True,
        catalog_id="demo_tall_mug_06",
        catalog_name="高杯子",
        catalog_aliases=("高杯子", "高杯", "高马克杯", "tall mug", "tall cup", "cup", "mug"),
        support_fixture_id="dining_table",
        home_place_id="dining_area",
        visibility_pose=(1.735163245, 1.674295473, 1.8464061075),
        preferred_view_bearing_rad=-1.292969691181926,
    ),
    HouseholdObject(
        "dining_wide_cup",
        "frl_apartment_cup_03.glb",
        "wide cup",
        (-0.38, 3.38),
        0.82,
        12.0,
        0.24,
        (-0.074191, -0.035406, -0.066690),
        (0.094708, 0.032094, 0.066736),
        dynamic=True,
        catalog_id="demo_wide_cup_07",
        catalog_name="宽口杯",
        catalog_aliases=("宽口杯", "宽杯子", "大杯子", "wide cup", "large cup", "cup", "mug"),
        support_fixture_id="media_cabinet",
        home_place_id="media_cabinet_front",
        visibility_pose=(-0.414836755, 2.644295473, math.pi / 2.0),
        visibility_standoff_m=0.75,
        preferred_view_bearing_rad=-math.pi / 2.0,
    ),
    HouseholdObject(
        "dining_small_mug",
        "frl_apartment_cup_05.glb",
        "small mug",
        (-0.08, 3.38),
        0.82,
        28.0,
        0.16,
        (-0.065014, -0.031006, -0.059173),
        (0.082948, 0.028194, 0.058921),
        dynamic=True,
        catalog_id="demo_small_mug_08",
        catalog_name="小杯子",
        catalog_aliases=("小杯子", "小马克杯", "小水杯", "small mug", "small cup", "cup", "mug"),
        support_fixture_id="media_cabinet",
        home_place_id="media_cabinet_front",
        visibility_pose=(-0.114836755, 2.644295473, math.pi / 2.0),
        visibility_standoff_m=0.75,
        preferred_view_bearing_rad=-math.pi / 2.0,
    ),
)

HOUSEHOLD_OBJECTS = FORMAL_SURVEY_OBJECTS + DEMO_GRASP_OBJECTS


def object_set_signature() -> str:
    payload = [
        {
            "object_id": item.object_id,
            "source_name": item.source_name,
            "position_xy": item.position_xy,
            "support_height_above_floor_m": item.support_height_above_floor_m,
            "yaw_deg": item.yaw_deg,
            "mass_kg": item.mass_kg,
            "dynamic": item.dynamic,
        }
        for item in FORMAL_SURVEY_OBJECTS
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


OBJECT_SET_SIGNATURE = object_set_signature()


def asset_manifest() -> dict:
    return {
        "schema_version": 1,
        "asset_set": "replica_cad_household_clutter_v1",
        "asset_set_signature": OBJECT_SET_SIGNATURE,
        "perception_labels_supplied_to_robot": False,
        "license_note": "ReplicaCAD local dataset; see its bundled README and LICENSE",
        "objects": [
            {
                **asdict(item),
                "source_path": str(item.source_path),
                "prepared_usd": str(item.prepared_usd),
            }
            for item in HOUSEHOLD_OBJECTS
        ],
    }


def require_prepared_assets() -> None:
    missing = [item.prepared_usd for item in HOUSEHOLD_OBJECTS if not item.prepared_usd.is_file()]
    if missing:
        raise FileNotFoundError(
            "家庭自主发现物品尚未转换；请先运行 ./mobilemanibench.sh home-assets。缺少："
            + ", ".join(str(path) for path in missing)
        )


__all__ = [
    "HOUSEHOLD_OBJECTS",
    "FORMAL_SURVEY_OBJECTS",
    "DEMO_GRASP_OBJECTS",
    "OBJECT_SET_SIGNATURE",
    "PREPARED_ASSET_ROOT",
    "HouseholdObject",
    "asset_manifest",
    "require_prepared_assets",
]
