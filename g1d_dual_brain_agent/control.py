"""Exclusive control leases for base, arm and hand resources."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from .models import ControlResource


@dataclass(frozen=True)
class ControlLease:
    owner: str
    resources: tuple[ControlResource, ...]
    generation: int


class ControlArbiter:
    """Prevent VLN, align and VLA phases from competing for actuators."""

    def __init__(self) -> None:
        self._owners: dict[ControlResource, tuple[str, int]] = {}
        self._generation = 0
        self._estop_latched = False
        self._lock = RLock()

    def acquire(
        self,
        owner: str,
        resources: tuple[ControlResource, ...],
    ) -> ControlLease:
        if not owner or not resources:
            raise ValueError("control lease needs owner and resources")
        if len(resources) != len(set(resources)):
            raise ValueError("control lease contains duplicate resources")
        with self._lock:
            if self._estop_latched:
                raise RuntimeError("emergency stop is latched")
            conflicts = {
                resource: current[0]
                for resource, current in self._owners.items()
                if resource in resources
            }
            if conflicts:
                details = ", ".join(
                    f"{resource.value}={current}"
                    for resource, current in sorted(
                        conflicts.items(), key=lambda item: item[0].value
                    )
                )
                raise RuntimeError(f"control resources already leased: {details}")
            self._generation += 1
            generation = self._generation
            for resource in resources:
                self._owners[resource] = (owner, generation)
            return ControlLease(owner, resources, generation)

    def validate(self, lease: ControlLease) -> bool:
        with self._lock:
            return (
                not self._estop_latched
                and all(
                    self._owners.get(resource)
                    == (lease.owner, lease.generation)
                    for resource in lease.resources
                )
            )

    def release(self, lease: ControlLease) -> None:
        with self._lock:
            for resource in lease.resources:
                if self._owners.get(resource) == (
                    lease.owner,
                    lease.generation,
                ):
                    del self._owners[resource]

    def emergency_stop(self) -> None:
        with self._lock:
            self._estop_latched = True
            self._owners.clear()

    def reset_emergency_stop(self, *, hardware_safe: bool) -> None:
        if not hardware_safe:
            raise ValueError("cannot reset emergency stop without hardware_safe")
        with self._lock:
            self._estop_latched = False

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "estop_latched": self._estop_latched,
                "generation": self._generation,
                "owners": {
                    resource.value: owner
                    for resource, (owner, _generation) in self._owners.items()
                },
            }


__all__ = ["ControlArbiter", "ControlLease"]
