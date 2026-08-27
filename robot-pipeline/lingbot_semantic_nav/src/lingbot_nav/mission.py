"""High-level mission resolution; no motor command is ever produced here."""

from __future__ import annotations

from .errors import IntentParseError
from .intent import IntentParser
from .models import IntentKind, Mission, MissionStep, NavigationStep, RouteAction
from .place_db import PlaceDatabase
from .place_db import normalize_label
from .topology import TopologyGraph


class MissionResolver:
    def __init__(
        self,
        parser: IntentParser,
        places: PlaceDatabase,
        topology: TopologyGraph | None = None,
        topology_start: str = "",
    ) -> None:
        self.parser = parser
        self.places = places
        self.topology = topology
        self.topology_start = ""
        if topology_start:
            self.set_topology_start(topology_start)

    def set_topology_start(self, place_id: str) -> None:
        if self.topology is None:
            raise IntentParseError("不能在未配置拓扑图时设置拓扑起点")
        if not self.topology.has_node(place_id):
            raise IntentParseError(f"拓扑起点不在图中：{place_id}")
        self.topology_start = place_id

    def resolve(self, command: str) -> Mission:
        intent = self.parser.parse(command)
        if intent.intent == IntentKind.CANCEL:
            raise IntentParseError("这是取消指令，不应生成新的导航目标")
        if intent.intent == IntentKind.FOLLOW_PERSON:
            raise IntentParseError("follow_person 需要动态人员跟随模块，静态地点导航不执行")
        if intent.intent == IntentKind.UNKNOWN:
            raise IntentParseError("无法从指令中确定一个已知目标地点")
        requested_steps = intent.route or (
            NavigationStep(RouteAction.ARRIVE, intent.destination),
        )
        # The final destination may be inferred from a functional phrase, but
        # every intermediate semantic waypoint must be explicitly grounded in
        # the user's words. This blocks an LLM from returning a valid yet
        # unrequested A in a direct "去 B" command.
        if len(requested_steps) > 1:
            normalized_command = normalize_label(command)
            for requested in requested_steps[:-1]:
                place = self.places.resolve(requested.destination).place
                labels = (place.place_id, place.name, *place.aliases)
                if not any(
                    normalize_label(label) in normalized_command
                    for label in labels
                    if normalize_label(label)
                ):
                    raise IntentParseError(
                        f"中间地点 {place.place_id!r} 未在用户指令中明确出现，拒绝隐式绕行"
                    )
        # A topology is consulted only for direction constraints explicitly
        # present in the command.  Merely configuring a graph must never inject
        # semantic waypoints into a direct request such as "去 B".
        if self.topology is not None and intent.route_constraints:
            if not self.topology_start:
                raise IntentParseError("已配置拓扑图，但没有设置当前拓扑地点")
            requested_steps = self.topology.plan(
                self.topology_start,
                intent.route_constraints,
                tuple(step.destination for step in requested_steps),
            )
        elif intent.route_constraints:
            constraints = "、".join(item.value for item in intent.route_constraints)
            raise IntentParseError(
                f"检测到尚未支持的方向路线约束：{constraints}；"
                "需要门/走廊拓扑地图，不能将该约束静默忽略"
            )
        steps: list[MissionStep] = []
        for requested in requested_steps:
            match = self.places.resolve(requested.destination)
            steps.append(
                MissionStep(
                    requested.action,
                    match.place,
                    match.score,
                    match.matched_alias,
                )
            )
        final = steps[-1]
        return Mission(
            command,
            intent,
            final.place,
            final.match_score,
            final.matched_alias,
            tuple(steps),
        )
