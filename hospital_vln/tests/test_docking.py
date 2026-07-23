import tempfile
import unittest
from pathlib import Path

from hospital_vln.docking import (
    ChairInstance,
    evaluate_candidates,
    generate_candidates,
    select_candidate,
)


class HospitalDockingTest(unittest.TestCase):
    def _map(self, root: Path, *, blocked_world=()) -> Path:
        width = height = 160
        resolution = 0.1
        origin = -8.0
        pixels = bytearray([254]) * (width * height)
        for x, y in blocked_world:
            col = int((x - origin) // resolution)
            logical_row = int((y - origin) // resolution)
            pgm_row = height - 1 - logical_row
            pixels[pgm_row * width + col] = 0
        (root / "map.pgm").write_bytes(
            f"P5\n{width} {height}\n255\n".encode("ascii") + pixels
        )
        path = root / "map.yaml"
        path.write_text(
            "image: map.pgm\n"
            f"resolution: {resolution}\n"
            f"origin: [{origin}, {origin}, 0.0]\n"
            "negate: 0\n"
            "occupied_thresh: 0.65\n"
            "free_thresh: 0.196\n",
            encoding="utf-8",
        )
        return path

    def test_generates_two_sides_per_chair(self):
        chairs = (
            ChairInstance("left", "/left", -2.0, -0.5, -1.0, 0.5),
            ChairInstance("right", "/right", 1.0, -0.5, 2.0, 0.5),
        )

        values = generate_candidates(chairs)

        self.assertEqual(len(values), 4)
        self.assertEqual(
            {item[0] for item in values},
            {"left_south", "left_north", "right_south", "right_north"},
        )

    def test_occupancy_rejects_candidate_and_dynamic_block_changes_selection(self):
        chair = ChairInstance("chair", "/chair", -1.0, -0.5, 1.0, 0.5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blocked_map = self._map(root, blocked_world=((0.0, 1.3),))
            filtered = evaluate_candidates(blocked_map, (chair,))

            north = next(item for item in filtered if item.side == "north")
            south = next(item for item in filtered if item.side == "south")
            self.assertFalse(north.eligible)
            self.assertTrue(
                {"occupied_or_unknown", "insufficient_footprint_clearance"}
                & set(north.rejection_reasons)
            )
            self.assertTrue(south.eligible)

        with tempfile.TemporaryDirectory() as directory:
            open_map = self._map(Path(directory))
            candidates = evaluate_candidates(open_map, (chair,))
            first = select_candidate(candidates)
            alternate = select_candidate(
                candidates,
                blocked_candidate_ids=(first.candidate_id,),
            )

            self.assertNotEqual(first.candidate_id, alternate.candidate_id)


if __name__ == "__main__":
    unittest.main()
