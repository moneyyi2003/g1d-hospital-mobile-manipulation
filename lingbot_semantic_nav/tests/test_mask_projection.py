import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from lingbot_nav.errors import ConfigurationError
from lingbot_nav.mapping.mask_projection import project_mask_to_map


@unittest.skipIf(np is None, "NumPy mapping extra is not installed")
class MaskProjectionTest(unittest.TestCase):
    def test_mask_projects_only_selected_pixels(self):
        points = np.zeros((2, 2, 3), dtype=np.float32)
        points[0, 0] = [1, 2, 3]
        points[1, 1] = [4, 5, 6]
        mask = np.asarray([[1, 0], [0, 1]], dtype=bool)
        alignment = np.eye(4)
        alignment[0, 3] = 10
        projected = project_mask_to_map(
            mask, points, alignment, scale_m_per_unit=2.0
        )
        self.assertTrue(np.allclose(projected, [[12, 4, 6], [18, 10, 12]]))

    def test_mask_and_geometry_must_be_pixel_aligned(self):
        with self.assertRaises(ConfigurationError):
            project_mask_to_map(
                np.ones((2, 2)), np.ones((3, 3, 3)), np.eye(4), scale_m_per_unit=1
            )


if __name__ == "__main__":
    unittest.main()
