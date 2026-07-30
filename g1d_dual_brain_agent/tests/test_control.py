from __future__ import annotations

import unittest

from g1d_dual_brain_agent.control import ControlArbiter
from g1d_dual_brain_agent.models import ControlResource


class ControlArbiterTest(unittest.TestCase):
    def test_different_owner_cannot_take_base(self) -> None:
        controls = ControlArbiter()
        controls.acquire("vln", (ControlResource.BASE,))

        with self.assertRaisesRegex(RuntimeError, "already leased"):
            controls.acquire("vla", (ControlResource.BASE,))

    def test_stale_lease_cannot_release_new_generation(self) -> None:
        controls = ControlArbiter()
        old = controls.acquire("agent", (ControlResource.BASE,))
        with self.assertRaisesRegex(RuntimeError, "already leased"):
            controls.acquire("agent", (ControlResource.BASE,))
        controls.release(old)
        new = controls.acquire("agent", (ControlResource.BASE,))

        controls.release(old)

        self.assertTrue(controls.validate(new))

    def test_estop_revokes_lease_and_requires_safe_reset(self) -> None:
        controls = ControlArbiter()
        lease = controls.acquire("vla", (ControlResource.RIGHT_ARM,))

        controls.emergency_stop()

        self.assertFalse(controls.validate(lease))
        with self.assertRaisesRegex(ValueError, "hardware_safe"):
            controls.reset_emergency_stop(hardware_safe=False)
        controls.reset_emergency_stop(hardware_safe=True)
        replacement = controls.acquire("vln", (ControlResource.BASE,))
        self.assertTrue(controls.validate(replacement))


if __name__ == "__main__":
    unittest.main()
