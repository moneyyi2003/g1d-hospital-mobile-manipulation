"""Validated semantic-place lookup; only this layer is allowed to provide poses."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Iterable

from .errors import AmbiguousPlaceError, ConfigurationError, UnknownPlaceError
from .models import Place, PlaceStatus


_PUNCTUATION = re.compile(r"[\s\-_，。！？、,.!?;；:：'\"“”‘’()（）\[\]{}]+")


def normalize_label(value: str) -> str:
    return _PUNCTUATION.sub("", value).casefold()


@dataclass(frozen=True)
class PlaceMatch:
    place: Place
    score: float
    matched_alias: str


class PlaceDatabase:
    def __init__(
        self,
        places: Iterable[Place],
        frame_id: str = "map",
        *,
        map_id: str = "",
        map_sha256: str = "",
        schema_version: int = 2,
    ) -> None:
        self.frame_id = frame_id
        self.map_id = map_id
        self.map_sha256 = map_sha256
        self.schema_version = schema_version
        self.places = tuple(places)
        if not self.places:
            raise ConfigurationError("Semantic place database is empty")

        ids: set[str] = set()
        self._labels: list[tuple[str, str, Place]] = []
        for place in self.places:
            if place.status != PlaceStatus.APPROVED:
                raise ConfigurationError(
                    f"Only approved places may enter the navigation database: {place.place_id}"
                )
            if place.entrance_pose.frame_id != self.frame_id:
                raise ConfigurationError(
                    f"Place {place.place_id!r} frame {place.entrance_pose.frame_id!r} "
                    f"does not match catalog frame {self.frame_id!r}"
                )
            if place.place_id in ids:
                raise ConfigurationError(f"Duplicate place id: {place.place_id}")
            ids.add(place.place_id)
            for label in (place.place_id, *place.aliases):
                normalized = normalize_label(label)
                if normalized:
                    self._labels.append((normalized, label, place))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        allow_legacy: bool = False,
        expected_map_id: str = "",
        expected_map_sha256: str = "",
    ) -> "PlaceDatabase":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Cannot read place database {source}: {exc}") from exc
        schema_version = int(payload.get("schema_version", 0))
        if schema_version == 1:
            if not allow_legacy:
                raise ConfigurationError(
                    "Legacy place schema v1 is not accepted by the formal navigation chain"
                )
            frame_id = str(payload.get("frame_id", "map"))
            places = [Place.from_mapping(item, frame_id) for item in payload.get("places", [])]
            return cls(places, frame_id, schema_version=1)
        if schema_version != 2:
            raise ConfigurationError("Unsupported place database schema_version")

        map_info = payload.get("map")
        if not isinstance(map_info, dict):
            raise ConfigurationError("Place schema v2 requires a map identity object")
        frame_id = str(map_info.get("frame_id", ""))
        map_id = str(map_info.get("id", "")).strip()
        map_sha256 = str(map_info.get("sha256", "")).strip().lower()
        if frame_id != "map":
            raise ConfigurationError("Formal place catalogs must use the ROS 'map' frame")
        if not map_id:
            raise ConfigurationError("Place catalog map.id must not be empty")
        if not re.fullmatch(r"[0-9a-f]{64}", map_sha256):
            raise ConfigurationError("Place catalog map.sha256 must be a 64-character SHA-256")
        if expected_map_id and map_id != expected_map_id:
            raise ConfigurationError(
                f"Place catalog map id {map_id!r} does not match {expected_map_id!r}"
            )
        if expected_map_sha256 and map_sha256 != expected_map_sha256.casefold():
            raise ConfigurationError("Place catalog map hash does not match the active map")

        places = []
        for item in payload.get("places", []):
            if not isinstance(item, dict):
                raise ConfigurationError("Place catalog entries must be objects")
            if str(item.get("status", "")) != PlaceStatus.APPROVED.value:
                continue
            enriched = dict(item)
            enriched["_map_id"] = map_id
            enriched["_map_sha256"] = map_sha256
            places.append(Place.from_mapping(enriched, frame_id))
        return cls(
            places,
            frame_id,
            map_id=map_id,
            map_sha256=map_sha256,
            schema_version=2,
        )

    def catalog_for_prompt(self) -> list[dict[str, object]]:
        return [
            {
                "id": p.place_id,
                "name": p.name,
                "aliases": list(p.aliases),
                "region": p.region,
                "metadata": dict(p.metadata),
            }
            for p in self.places
        ]

    def resolve(
        self,
        query: str,
        *,
        minimum_score: float = 0.62,
        ambiguity_margin: float = 0.06,
    ) -> PlaceMatch:
        needle = normalize_label(query)
        if not needle:
            raise UnknownPlaceError("目标地点为空")

        candidates: dict[str, PlaceMatch] = {}
        for normalized, original, place in self._labels:
            if needle == normalized:
                score = 1.0
            elif normalized in needle or needle in normalized:
                shorter = min(len(needle), len(normalized))
                longer = max(len(needle), len(normalized))
                score = 0.82 + 0.17 * (shorter / longer)
            else:
                score = SequenceMatcher(None, needle, normalized).ratio()
            previous = candidates.get(place.place_id)
            if previous is None or score > previous.score:
                candidates[place.place_id] = PlaceMatch(place, score, original)

        ranked = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
        if not ranked or ranked[0].score < minimum_score:
            known = "、".join(place.name for place in self.places)
            raise UnknownPlaceError(f"未在语义地点库中找到“{query}”；可用地点：{known}")
        ambiguous = (
            len(ranked) > 1
            and ranked[0].score - ranked[1].score < ambiguity_margin
        )
        unique_exact = (
            len(ranked) == 1
            or (ranked[0].score == 1.0 and ranked[1].score < 1.0)
        )
        if ambiguous and not unique_exact:
            raise AmbiguousPlaceError(
                f"“{query}”同时接近“{ranked[0].place.name}”和“{ranked[1].place.name}”"
            )
        return ranked[0]
