import math
import unittest

import numpy as np

from lingbot_nav.mapping.alignment import fit_similarity


class SurveyAlignmentTests(unittest.TestCase):
    def test_recovers_metric_similarity(self):
        source = np.asarray(
            [[0, 0, 0], [1, 0, 0], [1, 2, 0], [-1, 1, 0], [0.5, 0.4, 0]],
            dtype=float,
        )
        angle = 0.63
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        expected_scale = 2.75
        translation = np.asarray([4.0, -1.2, 1.34])
        target = expected_scale * (source @ rotation.T) + translation

        scale, actual_rotation, actual_translation = fit_similarity(source, target)

        self.assertAlmostEqual(scale, expected_scale, places=8)
        np.testing.assert_allclose(actual_rotation, rotation, atol=1e-8)
        np.testing.assert_allclose(actual_translation, translation, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
