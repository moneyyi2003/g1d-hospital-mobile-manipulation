"""Validated place-id topology and constrained route search."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigurationError, NoPathError
from .models import NavigationStep, RouteAction, RouteConstraint
from .place_db import PlaceDatabase


@dataclass(frozen=True)
class TopologyEdge:
    source: str
    target: str
    directive: RouteConstraint | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TopologyEdge":
        try:
            unexpected = set(value) - {"from", "to", "directive"}
            if unexpected:
                raise ConfigurationError(
                    "Topology edges may contain only from/to/directive; unexpected: "
                    + ", ".join(sorted(unexpected))
                )
            directive_value = str(value.get("directive", "")).strip()
            return cls(
                source=str(value["from"]).strip(),
                target=str(value["to"]).strip(),
                directive=RouteConstraint(directive_value) if directive_value else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid topology edge: {value!r}") from exc


class TopologyGraph:
    def __init__(
        self,
        nodes: tuple[str, ...],
        edges: tuple[TopologyEdge, ...],
        places: PlaceDatabase,
    ) -> None:
        if not nodes:
            raise ConfigurationError("Topology graph has no nodes")
        if len(set(nodes)) != len(nodes):
            raise ConfigurationError("Topology graph contains duplicate nodes")
        known_places = {place.place_id for place in places.places}
        unknown = sorted(set(nodes) - known_places)
        if unknown:
            raise ConfigurationError(
                f"Topology nodes are missing from the reviewed place database: {', '.join(unknown)}"
            )
        node_set = set(nodes)
        adjacency: dict[str, list[TopologyEdge]] = {node: [] for node in nodes}
        seen_edges: set[tuple[str, str, RouteConstraint | None]] = set()
        for edge in edges:
            if not edge.source or not edge.target:
                raise ConfigurationError("Topology edge endpoints must not be empty")
            if edge.source not in node_set or edge.target not in node_set:
                raise ConfigurationError(
                    f"Topology edge {edge.source!r}->{edge.target!r} references an unknown node"
                )
            if edge.source == edge.target:
                raise ConfigurationError("Topology self-edges are not allowed")
            identity = (edge.source, edge.target, edge.directive)
            if identity in seen_edges:
                raise ConfigurationError(f"Duplicate topology edge: {identity!r}")
            seen_edges.add(identity)
            adjacency[edge.source].append(edge)
        self.nodes = nodes
        self.edges = edges
        self._node_set = node_set
        self._adjacency = adjacency

    def has_node(self, place_id: str) -> bool:
        return place_id in self._node_set

    @classmethod
    def load(cls, path: str | Path, places: PlaceDatabase) -> "TopologyGraph":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Cannot read topology graph {source}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ConfigurationError(f"Topology graph root must be an object: {source}")
        unexpected = set(payload) - {"schema_version", "nodes", "edges"}
        if unexpected:
            raise ConfigurationError(
                "Topology graph contains unsupported fields: " + ", ".join(sorted(unexpected))
            )
        if int(payload.get("schema_version", 0)) != 1:
            raise ConfigurationError("Unsupported topology graph schema_version")
        try:
            nodes = tuple(str(item).strip() for item in payload.get("nodes", []))
            edges = tuple(TopologyEdge.from_mapping(item) for item in payload.get("edges", []))
        except TypeError as exc:
            raise ConfigurationError(f"Invalid topology graph: {source}") from exc
        return cls(nodes, edges, places)

    def plan(
        self,
        start: str,
        constraints: tuple[RouteConstraint, ...],
        destinations: tuple[str, ...],
    ) -> tuple[NavigationStep, ...]:
        if start not in self._node_set:
            raise ConfigurationError(f"Topology start place is not a graph node: {start!r}")
        if not destinations:
            raise NoPathError("Topology route has no semantic destinations")
        missing = [item for item in destinations if item not in self._node_set]
        if missing:
            raise NoPathError(
                "Route destinations are not topology nodes: " + ", ".join(missing)
            )

        initial_destination_index = 0
        if not constraints:
            while (
                initial_destination_index < len(destinations)
                and destinations[initial_destination_index] == start
            ):
                initial_destination_index += 1
        initial = (start, 0, initial_destination_index)
        frontier = deque([initial])
        previous: dict[
            tuple[str, int, int],
            tuple[tuple[str, int, int], TopologyEdge] | None,
        ] = {initial: None}
        goal: tuple[str, int, int] | None = None

        while frontier:
            state = frontier.popleft()
            node, constraint_index, destination_index = state
            if constraint_index == len(constraints) and destination_index == len(destinations):
                goal = state
                break
            for edge in self._adjacency[node]:
                next_constraint = constraint_index
                if constraint_index < len(constraints):
                    if edge.directive != constraints[constraint_index]:
                        continue
                    next_constraint += 1
                next_destination = destination_index
                if (
                    next_constraint == len(constraints)
                    and destination_index < len(destinations)
                    and edge.target == destinations[destination_index]
                ):
                    next_destination += 1
                candidate = (edge.target, next_constraint, next_destination)
                if candidate in previous:
                    continue
                previous[candidate] = (state, edge)
                frontier.append(candidate)

        if goal is None:
            constraint_text = " → ".join(item.value for item in constraints) or "none"
            destination_text = " → ".join(destinations)
            raise NoPathError(
                f"No topology route from {start!r} satisfies constraints "
                f"[{constraint_text}] and destinations [{destination_text}]"
            )

        path: list[str] = []
        cursor = goal
        while True:
            entry = previous[cursor]
            if entry is None:
                break
            parent, edge = entry
            path.append(edge.target)
            cursor = parent
        path.reverse()
        if not path:
            return (NavigationStep(RouteAction.ARRIVE, start),)
        return tuple(
            NavigationStep(
                RouteAction.ARRIVE if index == len(path) - 1 else RouteAction.PASS,
                node,
            )
            for index, node in enumerate(path)
        )
