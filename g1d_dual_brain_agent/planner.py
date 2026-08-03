"""Mission loading and conservative compatibility compilation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from g1d_agent.router import RuleTaskPlanner

from .models import GoalKind, Mission, TaskGoal


_PICK_MARKERS = ("拿起", "取下", "抓起", "拿", "取", "抓")
_RETURN_MARKERS = ("再回到", "然后回到", "返回到", "回到", "返回")


def _catalog_items(
    catalog: Mapping[str, Any],
    key: str,
) -> list[Mapping[str, Any]]:
    items = catalog.get(key, [])
    if not isinstance(items, list):
        raise ValueError(f"{key} catalog must be a list")
    return [item for item in items if isinstance(item, Mapping)]


def _item_id(item: Mapping[str, Any]) -> str:
    return str(item.get("id") or item.get("object_id") or "").strip()


def _item_terms(item: Mapping[str, Any]) -> tuple[str, ...]:
    values = [
        _item_id(item),
        str(item.get("name", "")).strip(),
        str(item.get("source_label", "")).strip(),
        *[str(value).strip() for value in item.get("aliases", [])],
    ]
    return tuple(
        sorted(
            {value.casefold() for value in values if value},
            key=len,
            reverse=True,
        )
    )


def _match_catalog_item(
    text: str,
    items: Sequence[Mapping[str, Any]],
    *,
    kind: str,
) -> Mapping[str, Any]:
    normalized = text.casefold()
    matches: list[tuple[int, Mapping[str, Any]]] = []
    for item in items:
        matching_terms = [
            term for term in _item_terms(item) if term and term in normalized
        ]
        if matching_terms:
            matches.append((len(matching_terms[0]), item))
    if not matches:
        allowed = sorted(
            {
                str(item.get("name") or item.get("source_label") or _item_id(item))
                for item in items
                if _item_id(item)
            }
        )
        raise ValueError(
            f"无法从“{text.strip()}”匹配审核{kind}；当前可用："
            + "、".join(allowed)
        )
    matches.sort(key=lambda value: value[0], reverse=True)
    best_length = matches[0][0]
    best = {
        _item_id(item): item
        for length, item in matches
        if length == best_length
    }
    if len(best) != 1:
        raise ValueError(
            f"“{text.strip()}”匹配到多个审核{kind}："
            + "、".join(sorted(best))
        )
    return next(iter(best.values()))


def _approved_places(catalog: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        item
        for item in _catalog_items(catalog, "places")
        if str(item.get("status", "")).casefold() == "approved"
        and _item_id(item)
    ]


def _approved_objects(catalog: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        item
        for item in _catalog_items(catalog, "objects")
        if str(item.get("status", "")).casefold() == "approved"
        and _item_id(item)
    ]


def _split_long_instruction(instruction: str) -> tuple[str, str, str]:
    """Return the leading destination, pick phrase and return destination.

    The parser deliberately recognizes a small auditable grammar. Semantic
    fuzzy matching remains the responsibility of the existing reviewed-place
    resolver; this function only identifies the three task phases.
    """

    text = instruction.strip()
    if not text:
        raise ValueError("instruction cannot be empty")
    pick_hits = [
        (text.find(marker), marker)
        for marker in _PICK_MARKERS
        if text.find(marker) >= 0
    ]
    if not pick_hits:
        raise ValueError("家庭长任务必须包含“拿/取/抓”之一")
    pick_index, pick_marker = min(pick_hits, key=lambda value: value[0])
    before_pick = text[:pick_index].strip(" ，,；;。")
    after_pick = text[pick_index + len(pick_marker) :].strip()

    return_hits = [
        (after_pick.find(marker), marker)
        for marker in _RETURN_MARKERS
        if after_pick.find(marker) >= 0
    ]
    if not return_hits:
        raise ValueError("家庭长任务必须包含“回到/返回”以及审核返回地点")
    return_index, return_marker = min(return_hits, key=lambda value: value[0])
    object_phrase = after_pick[:return_index].strip(" ，,；;。然后再")
    return_phrase = after_pick[
        return_index + len(return_marker) :
    ].strip(" ，,；;。")
    if not before_pick or not object_phrase or not return_phrase:
        raise ValueError(
            "家庭长任务格式应类似：带我去客厅，拿起杯子，再回到沙发旁"
        )
    return before_pick, object_phrase, return_phrase


def compile_family_home_command(
    instruction: str,
    *,
    places_catalog: Mapping[str, Any],
    objects_catalog: Mapping[str, Any],
    mission_id: str = "",
) -> Mission:
    """Compile a reviewed family ``go -> pick -> return`` instruction.

    No coordinate is inferred from language. Both navigation destinations and
    the manipulation object must resolve to approved scan-derived catalog
    entries. A non-manipulation-ready object can still be compiled so the live
    backend can report the precise safety blocker; readiness is never promoted
    by this language layer.
    """

    outbound_phrase, object_phrase, return_phrase = _split_long_instruction(
        instruction
    )
    outbound = _match_catalog_item(
        outbound_phrase,
        _approved_places(places_catalog),
        kind="地点",
    )
    target = _match_catalog_item(
        object_phrase,
        _approved_objects(objects_catalog),
        kind="对象",
    )
    return_place = _match_catalog_item(
        return_phrase,
        _approved_places(places_catalog),
        kind="地点",
    )
    if not mission_id:
        digest = hashlib.sha256(instruction.strip().encode("utf-8")).hexdigest()[
            :12
        ]
        mission_id = f"family-home-{digest}"
    object_id = _item_id(target)
    outbound_id = _item_id(outbound)
    return_id = _item_id(return_place)
    goals = (
        TaskGoal(
            goal_id="outbound-navigation",
            kind=GoalKind.NAVIGATE,
            instruction=outbound_id,
            success_condition=f"机器人到达审核地点 {outbound_id}",
            metadata={
                "phase": "outbound",
                "resolved_place_id": outbound_id,
                "source_phrase": outbound_phrase,
            },
        ),
        TaskGoal(
            goal_id="pick-object",
            kind=GoalKind.INTERACT,
            instruction=f"拿起 {object_id}",
            target_id=object_id,
            action="pick",
            success_condition=(
                f"{object_id} 相对支撑面抬升并由右手稳定保持"
            ),
            metadata={
                "phase": "pick",
                "resolved_object_id": object_id,
                "source_phrase": object_phrase,
                "catalog_manipulation_ready": bool(
                    target.get("manipulation_ready", False)
                ),
            },
        ),
        TaskGoal(
            goal_id="return-navigation",
            kind=GoalKind.NAVIGATE,
            instruction=return_id,
            payload_object_id=object_id,
            success_condition=(
                f"机器人携带 {object_id} 到达审核地点 {return_id}"
            ),
            metadata={
                "phase": "return",
                "resolved_place_id": return_id,
                "requires_carried_object_id": object_id,
                "source_phrase": return_phrase,
            },
        ),
    )
    return Mission(
        mission_id,
        instruction.strip(),
        goals,
        maximum_transitions=48,
        maximum_attempts_per_skill=2,
    )


def load_mission(path: Path) -> Mission:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Mission.from_dict(payload)


def compile_command(
    instruction: str,
    *,
    object_id: str = "",
    region_hint: str = "",
    mission_id: str = "",
) -> Mission:
    """Reuse the old auditable language router, then emit v2 task goals.

    Object identity is deliberately not guessed from free text. Interaction
    commands need an object ID from the reviewed scene memory/catalog.
    """

    old_plan = RuleTaskPlanner().plan(instruction)
    if not mission_id:
        digest = hashlib.sha256(
            f"{instruction}\0{object_id}\0{region_hint}".encode("utf-8")
        ).hexdigest()[:12]
        mission_id = f"command-{digest}"
    if old_plan.route == "vln":
        goals = (
            TaskGoal(
                goal_id="navigate-1",
                kind=GoalKind.NAVIGATE,
                instruction=instruction.strip(),
                success_condition=old_plan.steps[0].success_condition,
                metadata={"compiled_from": old_plan.to_dict()},
            ),
        )
    else:
        if not object_id.strip():
            raise ValueError(
                "操作任务必须用 --object-id 或 mission JSON "
                "指定审核对象 ID；Agent 不会从自由文本猜坐标或对象身份"
            )
        manipulation = old_plan.steps[-1]
        goals = (
            TaskGoal(
                goal_id="interact-1",
                kind=GoalKind.INTERACT,
                instruction=instruction.strip(),
                target_id=object_id.strip(),
                action=str(manipulation.metadata.get("skill", "manipulation")),
                region_hint=region_hint.strip(),
                success_condition=manipulation.success_condition,
                metadata={"compiled_from": old_plan.to_dict()},
            ),
        )
    return Mission(mission_id, instruction.strip(), goals)


__all__ = [
    "compile_command",
    "compile_family_home_command",
    "load_mission",
]
