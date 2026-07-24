"""Deterministic geometry for the isolated Hospital manipulation scene."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BoxSpec:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ManipulationSceneSpec:
    table_parts: tuple[BoxSpec, ...]
    cube: BoxSpec
    cube_mass_kg: float
    tabletop_surface_z: float

    def to_dict(self) -> dict:
        return {
            "table_parts": [part.to_dict() for part in self.table_parts],
            "cube": self.cube.to_dict(),
            "cube_mass_kg": self.cube_mass_kg,
            "tabletop_surface_z": self.tabletop_surface_z,
        }


def build_manipulation_scene(
    object_position: tuple[float, float, float],
    object_size_m: float,
    *,
    cube_mass_kg: float = 0.25,
) -> ManipulationSceneSpec:
    """Place a four-legged table below the configured manipulation cube.

    The object position remains the task catalog's source of truth.  The table
    is shifted behind the cube so the cube is close to its front edge and is
    reachable from the existing south-side docking pose.
    """

    if not 0.02 <= object_size_m <= 0.50:
        raise ValueError("object_size_m must be between 0.02 and 0.50")
    if cube_mass_kg <= 0.0:
        raise ValueError("cube_mass_kg must be positive")

    object_x, object_y, object_z = object_position
    tabletop_length = 1.00
    tabletop_depth = 0.70
    tabletop_thickness = 0.08
    front_inset = 0.10
    tabletop_surface_z = object_z - object_size_m / 2.0
    if tabletop_surface_z <= tabletop_thickness + 0.10:
        raise ValueError("object is too low to build a usable support table")

    table_center_y = object_y + tabletop_depth / 2.0 - front_inset
    tabletop_center_z = tabletop_surface_z - tabletop_thickness / 2.0
    leg_size = 0.07
    leg_height = tabletop_surface_z - tabletop_thickness
    leg_center_z = leg_height / 2.0
    leg_x_offset = tabletop_length / 2.0 - 0.09
    leg_y_offset = tabletop_depth / 2.0 - 0.09

    parts = [
        BoxSpec(
            "Tabletop",
            (object_x, table_center_y, tabletop_center_z),
            (tabletop_length, tabletop_depth, tabletop_thickness),
        )
    ]
    for x_label, x_sign in (("Left", -1.0), ("Right", 1.0)):
        for y_label, y_sign in (("Front", -1.0), ("Rear", 1.0)):
            parts.append(
                BoxSpec(
                    f"Leg{x_label}{y_label}",
                    (
                        object_x + x_sign * leg_x_offset,
                        table_center_y + y_sign * leg_y_offset,
                        leg_center_z,
                    ),
                    (leg_size, leg_size, leg_height),
                )
            )

    return ManipulationSceneSpec(
        table_parts=tuple(parts),
        cube=BoxSpec(
            "RedCube",
            object_position,
            (object_size_m, object_size_m, object_size_m),
        ),
        cube_mass_kg=cube_mass_kg,
        tabletop_surface_z=tabletop_surface_z,
    )


__all__ = ["BoxSpec", "ManipulationSceneSpec", "build_manipulation_scene"]
