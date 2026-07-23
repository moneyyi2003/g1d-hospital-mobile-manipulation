import json
import unittest

from lingbot_nav.errors import ProviderError
from lingbot_nav.intent import DeepSeekIntentParser
from lingbot_nav.models import Place, Pose2D
from lingbot_nav.place_db import PlaceDatabase


class FakeClient:
    def __init__(self, destination: str):
        self.destination = destination

    def post(self, _url, _api_key, _payload):
        content = {
            "intent": "navigate",
            "destination": self.destination,
            "interaction_mode": "none",
            "confidence": 0.9,
            "route": [{"action": "arrive", "destination": self.destination}],
            "route_constraints": [],
        }
        return {"choices": [{"message": {"content": json.dumps(content)}}]}


class DeepSeekGuardTest(unittest.TestCase):
    def setUp(self):
        self.places = PlaceDatabase([
            Place("cafe", "咖啡厅", ("咖啡厅", "coffee shop"), Pose2D(1, 2))
        ])

    def parser(self, destination: str) -> DeepSeekIntentParser:
        return DeepSeekIntentParser(
            self.places,
            api_key="test-key",
            model="test-model",
            base_url="https://example.invalid",
            client=FakeClient(destination),
        )

    def test_exact_catalog_id_is_accepted(self):
        self.assertEqual(self.parser("cafe").parse("去咖啡厅").destination, "cafe")

    def test_alias_is_rejected_from_remote_output(self):
        with self.assertRaises(ProviderError):
            self.parser("咖啡厅").parse("去咖啡厅")

    def test_unknown_id_is_rejected(self):
        with self.assertRaises(ProviderError):
            self.parser("hidden_waypoint").parse("去咖啡厅")


if __name__ == "__main__":
    unittest.main()
