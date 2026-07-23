import unittest

from lingbot_nav.intent import RuleBasedIntentParser
from lingbot_nav.mission import MissionResolver
from lingbot_nav.errors import IntentParseError
from lingbot_nav.intent import IntentParser
from lingbot_nav.models import (
    IntentKind,
    NavigationIntent,
    NavigationStep,
    Place,
    Pose2D,
    RouteAction,
)
from lingbot_nav.place_db import PlaceDatabase
from lingbot_nav.topology import TopologyEdge, TopologyGraph


def places() -> PlaceDatabase:
    return PlaceDatabase([
        Place("a", "A点", ("A", "A点"), Pose2D(1, 0)),
        Place("b", "B点", ("B", "B点"), Pose2D(2, 0)),
        Place("c", "C点", ("C", "C点"), Pose2D(3, 0)),
    ])


class OrderedRouteTest(unittest.TestCase):
    def test_first_arrive_a_then_arrive_b(self):
        database = places()
        intent = RuleBasedIntentParser(database).parse("先到A，再到B")
        self.assertEqual(
            [(item.destination, item.action) for item in intent.route],
            [("a", RouteAction.ARRIVE), ("b", RouteAction.ARRIVE)],
        )

    def test_pass_a_then_arrive_b(self):
        database = places()
        intent = RuleBasedIntentParser(database).parse("经过A到B")
        self.assertEqual(
            [(item.destination, item.action) for item in intent.route],
            [("a", RouteAction.PASS), ("b", RouteAction.ARRIVE)],
        )

    def test_direct_b_contains_no_a(self):
        database = places()
        mission = MissionResolver(RuleBasedIntentParser(database), database).resolve("去B")
        self.assertEqual([item.place.place_id for item in mission.steps], ["b"])

    def test_configured_topology_does_not_inject_unrequested_waypoint(self):
        database = places()
        topology = TopologyGraph(
            ("a", "b", "c"),
            (TopologyEdge("c", "a"), TopologyEdge("a", "b")),
            database,
        )
        resolver = MissionResolver(
            RuleBasedIntentParser(database), database, topology, topology_start="c"
        )
        mission = resolver.resolve("去B")
        self.assertEqual([item.place.place_id for item in mission.steps], ["b"])

    def test_llm_cannot_insert_a_into_direct_b_request(self):
        class InjectingParser(IntentParser):
            def parse(self, _instruction):
                return NavigationIntent(
                    IntentKind.NAVIGATE,
                    "b",
                    confidence=0.9,
                    parser="deepseek",
                    route=(
                        NavigationStep(RouteAction.PASS, "a"),
                        NavigationStep(RouteAction.ARRIVE, "b"),
                    ),
                )

        database = places()
        resolver = MissionResolver(InjectingParser(), database)
        with self.assertRaises(IntentParseError):
            resolver.resolve("去B")


if __name__ == "__main__":
    unittest.main()
