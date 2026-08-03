"""ReplicaCAD household clutter used by the autonomous RGB discovery survey.

The names below are evaluation truth and asset-management metadata.  They are
never added to the USD semantic registry or passed to the perception model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
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

    @property
    def source_path(self) -> Path:
        return REPLICA_CAD_ROOT / "objects" / self.source_name

    @property
    def prepared_usd(self) -> Path:
        return PREPARED_ASSET_ROOT / self.object_id / f"{self.object_id}.usd"


HOUSEHOLD_OBJECTS = (
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
        0.50,
        (-0.055642, -0.034010, -0.055580),
        (0.077712, 0.044990, 0.055434),
        True,
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
        for item in HOUSEHOLD_OBJECTS
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
    "OBJECT_SET_SIGNATURE",
    "PREPARED_ASSET_ROOT",
    "HouseholdObject",
    "asset_manifest",
    "require_prepared_assets",
]
