"""Small dependency-free domain models shared by CLI and ROS 2 nodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping

from .errors import ConfigurationError, IntentParseError


class IntentKind(str, Enum):
    NAVIGATE = "navigate"
    GUIDE_PERSON = "guide_person"
    ESCORT = "escort"
    FOLLOW_PERSON = "follow_person"
    CANCEL = "cancel"
    UNKNOWN = "unknown"


class InteractionMode(str, Enum):
    NONE = "none"
    GUIDE = "guide"
    ESCORT = "escort"
    FOLLOW = "follow"


class RouteAction(str, Enum):
    PASS = "pass"
    ARRIVE = "arrive"


class RouteConstraint(str, Enum):
    EXIT = "exit"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    GO_STRAIGHT = "go_straight"


class PlaceStatus(str, Enum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"
    STALE = "stale"


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0
    frame_id: str = "map"

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.x, self.y, self.yaw)):
            raise ConfigurationError("Pose2D contains a non-finite value")
        if not self.frame_id:
            raise ConfigurationError("Pose2D.frame_id must not be empty")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], default_frame: str = "map") -> "Pose2D":
        try:
            return cls(
                x=float(value["x"]),
                y=float(value["y"]),
                yaw=float(value.get("yaw", 0.0)),
                frame_id=str(value.get("frame_id", default_frame)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid pose: {value!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DockingCandidate:
    candidate_id: str
    pose: Pose2D
    clearance_m: float
    footprint_radius_m: float
    occupancy_status: str
    reachable: bool
    review_status: str

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ConfigurationError("Docking candidate id must not be empty")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (self.clearance_m, self.footprint_radius_m)
        ):
            raise ConfigurationError("Docking candidate clearances must be finite and non-negative")
        if self.occupancy_status not in {"free", "occupied", "unknown", "outside"}:
            raise ConfigurationError(
                f"Unsupported docking occupancy status: {self.occupancy_status!r}"
            )
        if self.review_status not in {"pending", "accepted", "rejected"}:
            raise ConfigurationError(
                f"Unsupported docking review status: {self.review_status!r}"
            )

    @property
    def is_approved(self) -> bool:
        return (
            self.occupancy_status == "free"
            and self.reachable
            and self.review_status == "accepted"
        )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], default_frame: str = "map"
    ) -> "DockingCandidate":
        try:
            checks = value.get("checks", {})
            review = value.get("review", {})
            return cls(
                candidate_id=str(value["id"]).strip(),
                pose=Pose2D.from_mapping(value["pose"], default_frame),
                clearance_m=float(checks["clearance_m"]),
                footprint_radius_m=float(checks["footprint_radius_m"]),
                occupancy_status=str(checks["occupancy_status"]),
                reachable=bool(checks["reachable"]),
                review_status=str(review["status"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid docking candidate: {value!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.candidate_id,
            "pose": self.pose.to_dict(),
            "checks": {
                "clearance_m": self.clearance_m,
                "footprint_radius_m": self.footprint_radius_m,
                "occupancy_status": self.occupancy_status,
                "reachable": self.reachable,
            },
            "review": {"status": self.review_status},
        }


@dataclass(frozen=True)
class Place:
    place_id: str
    name: str
    aliases: tuple[str, ...]
    entrance_pose: Pose2D
    region: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: PlaceStatus = PlaceStatus.APPROVED
    map_id: str = ""
    map_sha256: str = ""
    target_type: str = ""
    source_id: str = ""
    docking_candidates: tuple[DockingCandidate, ...] = ()
    selected_docking_candidate: str = ""

    def __post_init__(self) -> None:
        if not self.place_id.strip() or not self.name.strip():
            raise ConfigurationError("Place id and name must not be empty")
        if not self.aliases:
            raise ConfigurationError(f"Place {self.place_id!r} needs at least one alias")
        if self.docking_candidates:
            candidate_ids = [item.candidate_id for item in self.docking_candidates]
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ConfigurationError(f"Place {self.place_id!r} has duplicate docking candidate ids")
            if self.selected_docking_candidate not in set(candidate_ids):
                raise ConfigurationError(
                    f"Place {self.place_id!r} selected docking candidate does not exist"
                )
            selected = next(
                item
                for item in self.docking_candidates
                if item.candidate_id == self.selected_docking_candidate
            )
            if selected.pose != self.entrance_pose:
                raise ConfigurationError(
                    f"Place {self.place_id!r} entrance pose disagrees with selected candidate"
                )
            if self.status == PlaceStatus.APPROVED and not selected.is_approved:
                raise ConfigurationError(
                    f"Approved place {self.place_id!r} must select a fully approved docking candidate"
                )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], default_frame: str = "map") -> "Place":
        try:
            aliases = tuple(str(item).strip() for item in value.get("aliases", []) if str(item).strip())
            name = str(value["name"]).strip()
            if name and name not in aliases:
                aliases = (name, *aliases)
            candidates = tuple(
                DockingCandidate.from_mapping(item, default_frame)
                for item in value.get("docking_candidates", [])
            )
            selected_id = str(value.get("selected_docking_candidate", "")).strip()
            if candidates:
                selected = next(
                    (item for item in candidates if item.candidate_id == selected_id),
                    None,
                )
                if selected is None:
                    raise ConfigurationError(
                        f"Place {value.get('id')!r} has no selected docking candidate"
                    )
                entrance_pose = selected.pose
            else:
                entrance_pose = Pose2D.from_mapping(value["entrance_pose"], default_frame)
            target = value.get("target", {})
            return cls(
                place_id=str(value["id"]).strip(),
                name=name,
                aliases=aliases,
                entrance_pose=entrance_pose,
                region=str(value.get("region", "")),
                metadata=dict(value.get("metadata", {})),
                status=PlaceStatus(str(value.get("status", "approved"))),
                map_id=str(value.get("_map_id", value.get("map_id", ""))),
                map_sha256=str(value.get("_map_sha256", value.get("map_sha256", ""))),
                target_type=str(target.get("type", value.get("target_type", ""))),
                source_id=str(target.get("source_id", value.get("source_id", ""))),
                docking_candidates=candidates,
                selected_docking_candidate=selected_id,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid place entry: {value!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.place_id,
            "name": self.name,
            "aliases": list(self.aliases),
            "entrance_pose": self.entrance_pose.to_dict(),
            "region": self.region,
            "metadata": dict(self.metadata),
            "status": self.status.value,
            "map_id": self.map_id,
            "map_sha256": self.map_sha256,
            "target": {"type": self.target_type, "source_id": self.source_id},
            "docking_candidates": [item.to_dict() for item in self.docking_candidates],
            "selected_docking_candidate": self.selected_docking_candidate,
        }


@dataclass(frozen=True)
class NavigationIntent:
    intent: IntentKind
    destination: str = ""
    interaction_mode: InteractionMode = InteractionMode.NONE
    confidence: float = 0.0
    parser: str = "unknown"
    route: tuple["NavigationStep", ...] = ()
    route_constraints: tuple[RouteConstraint, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise IntentParseError("Intent confidence must be between 0 and 1")
        if self.route and self.destination != self.route[-1].destination:
            raise IntentParseError("Intent destination must match the final route step")
        if self.route and self.route[-1].action != RouteAction.ARRIVE:
            raise IntentParseError("The final route step must use action='arrive'")
        # Intermediate steps may be either a loose pass-through waypoint or a
        # strict arrival.  "先到 A，再到 B" therefore remains two real arrival
        # goals instead of being collapsed into a decorative route through A.

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], parser: str) -> "NavigationIntent":
        try:
            intent = IntentKind(str(value.get("intent", "unknown")))
            mode = InteractionMode(str(value.get("interaction_mode", "none")))
            destination = str(value.get("destination", "") or "").strip()
            confidence = float(value.get("confidence", 0.0))
            route = tuple(NavigationStep.from_mapping(item) for item in value.get("route", []))
            constraints = tuple(
                RouteConstraint(str(item)) for item in value.get("route_constraints", [])
            )
        except (TypeError, ValueError) as exc:
            raise IntentParseError(f"Invalid structured intent: {value!r}") from exc

        if route:
            if destination and destination != route[-1].destination:
                raise IntentParseError("Structured intent destination disagrees with route")
            destination = route[-1].destination

        needs_destination = intent in {
            IntentKind.NAVIGATE,
            IntentKind.GUIDE_PERSON,
            IntentKind.ESCORT,
        }
        if needs_destination and not destination:
            raise IntentParseError(f"Intent {intent.value!r} requires a destination")
        return cls(intent, destination, mode, confidence, parser, route, constraints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "destination": self.destination,
            "interaction_mode": self.interaction_mode.value,
            "confidence": self.confidence,
            "parser": self.parser,
            "route": [step.to_dict() for step in self.route],
            "route_constraints": [item.value for item in self.route_constraints],
        }


@dataclass(frozen=True)
class NavigationStep:
    action: RouteAction
    destination: str

    def __post_init__(self) -> None:
        if not self.destination.strip():
            raise IntentParseError("Route step destination must not be empty")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NavigationStep":
        try:
            return cls(
                action=RouteAction(str(value["action"])),
                destination=str(value["destination"]).strip(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise IntentParseError(f"Invalid route step: {value!r}") from exc

    def to_dict(self) -> dict[str, str]:
        return {"action": self.action.value, "destination": self.destination}


@dataclass(frozen=True)
class MissionStep:
    action: RouteAction
    place: Place
    match_score: float
    matched_alias: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "place": self.place.to_dict(),
            "match_score": self.match_score,
            "matched_alias": self.matched_alias,
        }


@dataclass(frozen=True)
class Mission:
    command: str
    intent: NavigationIntent
    place: Place
    match_score: float
    matched_alias: str
    steps: tuple[MissionStep, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "intent": self.intent.to_dict(),
            "place": self.place.to_dict(),
            "match_score": self.match_score,
            "matched_alias": self.matched_alias,
            "steps": [step.to_dict() for step in self.steps],
        }
