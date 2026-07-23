import unittest

import numpy as np

from lingbot_nav.errors import ConfigurationError
from lingbot_nav.mapping.lingbot_backend import _unbatch_sequence


class LingBotBackendShapeTests(unittest.TestCase):
    def test_unbatches_single_sequence(self):
        value = np.zeros((1, 3, 3, 8, 12), dtype=np.float32)

        result = _unbatch_sequence(
            value, name="postprocessed RGB", unbatched_ndim=4
        )

        self.assertEqual(result.shape, (3, 3, 8, 12))

    def test_keeps_already_unbatched_sequence(self):
        value = np.zeros((3, 3, 8, 12), dtype=np.float32)

        result = _unbatch_sequence(
            value, name="postprocessed RGB", unbatched_ndim=4
        )

        self.assertIs(result, value)

    def test_rejects_multiple_batches(self):
        value = np.zeros((2, 3, 3, 8, 12), dtype=np.float32)

        with self.assertRaisesRegex(ConfigurationError, "unexpected shape"):
            _unbatch_sequence(value, name="postprocessed RGB", unbatched_ndim=4)


if __name__ == "__main__":
    unittest.main()
