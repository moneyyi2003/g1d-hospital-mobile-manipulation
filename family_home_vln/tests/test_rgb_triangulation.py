from __future__ import annotations

import unittest

import numpy as np

from family_home_vln.rgb_triangulation import triangulate_rays


class RgbTriangulationTests(unittest.TestCase):
    def test_recovers_point_from_multiple_metric_rays(self) -> None:
        expected = np.asarray([1.7, 3.0, 0.8])
        origins = [
            np.asarray([2.0, 1.8, 1.34]),
            np.asarray([2.1, 2.0, 1.34]),
            np.asarray([1.8, 2.2, 1.34]),
            np.asarray([1.6, 2.0, 1.34]),
        ]
        directions = [
            (expected - origin) / np.linalg.norm(expected - origin)
            for origin in origins
        ]
        point, residuals, condition = triangulate_rays(origins, directions)
        np.testing.assert_allclose(point, expected, atol=1.0e-8)
        self.assertLess(max(residuals), 1.0e-8)
        self.assertLess(condition, 1.0e5)

    def test_rejects_coincident_parallel_rays(self) -> None:
        origins = [np.zeros(3), np.zeros(3), np.zeros(3)]
        directions = [
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
        ]
        with self.assertRaises(ValueError):
            triangulate_rays(origins, directions)


if __name__ == "__main__":
    unittest.main()
