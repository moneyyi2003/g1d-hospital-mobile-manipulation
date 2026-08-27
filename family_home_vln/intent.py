"""Constrained LLM place selection for the family-home dashboard.

The language model is only permitted to select an approved catalog ``place_id``.
Map coordinates, docking poses, and route generation remain outside the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


LINGBOT_SRC = Path(__file__).resolve().parents[1] / "lingbot_semantic_nav/src"
if str(LINGBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LINGBOT_SRC))

from lingbot_nav.errors import SemanticNavError
from lingbot_nav.intent import IntentParser, create_intent_parser
from lingbot_nav.mission import MissionResolver
from lingbot_nav.place_db import PlaceDatabase


@dataclass(frozen=True)
class FamilyPlaceResolution:
    place_id: str
    place_name: str
    parser: str
    confidence: float
    intent: str


class FamilyIntentResolver:
    """Resolve an everyday-language destination to one reviewed place ID."""

    def __init__(
        self,
        places_path: Path,
        *,
        provider: str = "deepseek",
        allow_rule_fallback: bool = True,
        parser: IntentParser | None = None,
    ) -> None:
        self.places = PlaceDatabase.load(places_path)
        self.parser = parser or create_intent_parser(
            provider,
            self.places,
            allow_rule_fallback=allow_rule_fallback,
        )
        self.missions = MissionResolver(self.parser, self.places)

    @property
    def name(self) -> str:
        return self.parser.name

    def resolve(self, command: str) -> FamilyPlaceResolution:
        try:
            mission = self.missions.resolve(command)
        except SemanticNavError as exc:
            raise ValueError(str(exc)) from exc
        return FamilyPlaceResolution(
            place_id=mission.place.place_id,
            place_name=mission.place.name,
            parser=mission.intent.parser,
            confidence=mission.intent.confidence,
            intent=mission.intent.intent.value,
        )


__all__ = ["FamilyIntentResolver", "FamilyPlaceResolution"]
