"""Intent parsers, including local rules and provider-specific HTTP adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
import socket
import time
from typing import Any
from urllib import error, request

from .config import load_dotenv
from .errors import ConfigurationError, IntentParseError, ProviderError
from .models import (
    InteractionMode,
    IntentKind,
    NavigationIntent,
    NavigationStep,
    RouteAction,
    RouteConstraint,
)
from .place_db import PlaceDatabase, normalize_label


INTENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [item.value for item in IntentKind],
        },
        "destination": {"type": "string"},
        "interaction_mode": {
            "type": "string",
            "enum": [item.value for item in InteractionMode],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "route": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": [item.value for item in RouteAction]},
                    "destination": {"type": "string"},
                },
                "required": ["action", "destination"],
                "additionalProperties": False,
            },
        },
        "route_constraints": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [item.value for item in RouteConstraint],
            },
        },
    },
    "required": [
        "intent",
        "destination",
        "interaction_mode",
        "confidence",
        "route",
        "route_constraints",
    ],
    "additionalProperties": False,
}


class IntentParser(ABC):
    name = "base"

    @abstractmethod
    def parse(self, instruction: str) -> NavigationIntent:
        raise NotImplementedError


class RuleBasedIntentParser(IntentParser):
    """Deterministic parser used by tests, demos, and API fallback."""

    name = "rule"
    _cancel_words = ("取消", "停止", "停下", "终止任务", "cancel", "stop")
    _follow_words = ("跟着我", "跟随我", "follow me")
    _guide_words = ("带我", "带我们", "领我", "引导我", "guide me")
    _escort_words = ("护送", "带客人", "送他", "送她", "escort")
    _route_constraints = (
        (RouteConstraint.EXIT, ("出门", "离开房间", "exit")),
        (RouteConstraint.TURN_LEFT, ("左转", "向左转", "turn left")),
        (RouteConstraint.TURN_RIGHT, ("右转", "向右转", "turn right")),
        (RouteConstraint.GO_STRAIGHT, ("直行", "一直走", "go straight")),
    )

    def __init__(self, places: PlaceDatabase) -> None:
        self.places = places

    def parse(self, instruction: str) -> NavigationIntent:
        normalized = normalize_label(instruction)
        if not normalized:
            raise IntentParseError("指令为空")
        if any(normalize_label(word) in normalized for word in self._cancel_words):
            return NavigationIntent(IntentKind.CANCEL, "", InteractionMode.NONE, 1.0, self.name)
        if any(normalize_label(word) in normalized for word in self._follow_words):
            return NavigationIntent(
                IntentKind.FOLLOW_PERSON,
                "",
                InteractionMode.FOLLOW,
                0.98,
                self.name,
            )

        hits: list[tuple[int, int, str, str, str]] = []
        for place in self.places.places:
            for alias in (place.place_id, *place.aliases):
                token = normalize_label(alias)
                if not token:
                    continue
                start = normalized.find(token)
                while start >= 0:
                    hits.append((start, start + len(token), token, place.place_id, alias))
                    start = normalized.find(token, start + 1)
        if not hits:
            return NavigationIntent(IntentKind.UNKNOWN, "", InteractionMode.NONE, 0.0, self.name)

        # Keep the longest alias at each textual position and preserve command order.
        selected_spans: list[tuple[int, int, str]] = []
        for start, end, token, _, _ in sorted(hits, key=lambda item: (item[0], -(item[1] - item[0]))):
            if any(start < kept_end and end > kept_start for kept_start, kept_end, _ in selected_spans):
                continue
            selected_spans.append((start, end, token))
        selected_spans.sort()

        destinations: list[str] = []
        for start, end, token in selected_spans:
            candidates = [item for item in hits if item[:3] == (start, end, token)]
            place_ids = {item[3] for item in candidates}
            # Preserve a shared alias so PlaceDatabase performs its normal ambiguity rejection.
            destination = candidates[0][4] if len(place_ids) > 1 else candidates[0][3]
            if not destinations or destinations[-1] != destination:
                destinations.append(destination)

        if any(normalize_label(word) in normalized for word in self._escort_words):
            kind, mode = IntentKind.ESCORT, InteractionMode.ESCORT
        elif any(normalize_label(word) in normalized for word in self._guide_words):
            kind, mode = IntentKind.GUIDE_PERSON, InteractionMode.GUIDE
        else:
            kind, mode = IntentKind.NAVIGATE, InteractionMode.NONE
        route_steps = []
        for index, destination in enumerate(destinations):
            if index == len(destinations) - 1:
                action = RouteAction.ARRIVE
            else:
                start, end, _ = selected_spans[index]
                next_start = selected_spans[index + 1][0]
                prefix = normalized[max(0, start - 6):start]
                bridge = normalized[end:next_start]
                if any(token in prefix for token in ("经过", "途经", "路过", "via", "pass")):
                    action = RouteAction.PASS
                elif (
                    any(token in prefix for token in ("先到", "先去", "先前往", "到", "去"))
                    and any(token in bridge for token in ("再", "然后", "之后", "以后", "then"))
                ):
                    action = RouteAction.ARRIVE
                else:
                    action = RouteAction.PASS
            route_steps.append(NavigationStep(action, destination))
        route = tuple(route_steps)
        constraint_hits: list[tuple[int, RouteConstraint]] = []
        for constraint, phrases in self._route_constraints:
            positions = [
                normalized.find(normalize_label(phrase))
                for phrase in phrases
                if normalize_label(phrase) in normalized
            ]
            if positions:
                constraint_hits.append((min(positions), constraint))
        constraints = tuple(item[1] for item in sorted(constraint_hits, key=lambda item: item[0]))
        return NavigationIntent(
            kind,
            destinations[-1],
            mode,
            0.96,
            self.name,
            route,
            constraints,
        )


class _JsonHttpClient:
    def __init__(self, timeout_seconds: float = 20.0, retries: int = 2) -> None:
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def post(self, url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "lingbot-semantic-nav/0.1",
            },
        )
        for attempt in range(self.retries + 1):
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                detail = exc.read(1000).decode("utf-8", errors="replace")
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if retryable and attempt < self.retries:
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise ProviderError(f"LLM HTTP {exc.code}: {detail}") from exc
            except (error.URLError, socket.timeout, TimeoutError, json.JSONDecodeError) as exc:
                if attempt < self.retries:
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise ProviderError(f"LLM request failed: {exc}") from exc
        raise ProviderError("LLM request failed after retries")


class _RemoteIntentParser(IntentParser):
    def __init__(
        self,
        places: PlaceDatabase,
        *,
        api_key: str,
        model: str,
        base_url: str,
        client: _JsonHttpClient | None = None,
    ) -> None:
        if not api_key:
            raise ConfigurationError(f"{self.name} API key is missing")
        self.places = places
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = client or _JsonHttpClient()

    def _system_prompt(self) -> str:
        catalog = json.dumps(self.places.catalog_for_prompt(), ensure_ascii=False)
        return (
            "你是移动机器人的高层指令解析器。只解析意图，不规划路径、不生成坐标、不控制速度。"
            "必须输出 JSON，字段严格为 intent、destination、interaction_mode、confidence、route、route_constraints。"
            "route 是按用户明确提及的顺序排列，禁止添加用户未要求的地点。"
            "‘经过A到B’中A用 action=pass；‘先到A再到B’中A和B都用 action=arrive。"
            "只说‘去B’时 route 只能包含B，不能为了展示路线而补充A或其他途经点。"
            "destination 等于最后一步的地点 id；route 中的 destination 也只能使用地点 id。"
            "route_constraints 只记录 exit/turn_left/turn_right/go_straight，没有方向要求时为空数组。"
            "没有目标时 destination 输出空字符串且 route 输出空数组。"
            "根据地点名称、别名和日常功能描述做语义归一，不要求用户逐字说出目录词；"
            "例如‘吃饭的地方’应匹配用餐区，‘睡觉的地方’应匹配卧室。"
            "地点 metadata 是可信的场景知识；可以根据用户想做的事情、需要的服务和典型物品，"
            "推断一个最终地点。例如有‘坐下、等待医生、休息’功能的地点应响应"
            "‘找个能坐着等医生的地方’，有‘咨询、问路、登记’功能的地点应响应"
            "‘我想找工作人员问点事情’。这种功能推断只允许用于最终目的地，不能虚构中间地点。"
            "只能选择地点库已有的 id；语义上没有对应地点时，intent=unknown，destination=\"\"。"
            "intent 可选 navigate/guide_person/escort/follow_person/cancel/unknown；"
            "interaction_mode 可选 none/guide/escort/follow。"
            f"地点库={catalog}"
        )

    def _decode_intent(self, content: str) -> NavigationIntent:
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.name} returned invalid JSON") from exc
        intent = NavigationIntent.from_mapping(value, self.name)
        allowed = {place.place_id for place in self.places.places}
        returned = [step.destination for step in intent.route]
        if intent.destination:
            returned.append(intent.destination)
        invalid = sorted({item for item in returned if item not in allowed})
        if invalid:
            raise ProviderError(
                f"{self.name} returned non-catalog place ids: {', '.join(invalid)}"
            )
        return intent


class DeepSeekIntentParser(_RemoteIntentParser):
    """DeepSeek V4 adapter using its OpenAI-compatible Chat Completions API."""

    name = "deepseek"

    def parse(self, instruction: str) -> NavigationIntent:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": instruction},
            ],
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "max_tokens": 300,
            "stream": False,
        }
        response = self.client.post(
            f"{self.base_url}/chat/completions", self.api_key, payload
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("DeepSeek response has no message content") from exc
        if not content:
            raise ProviderError("DeepSeek returned empty content")
        return self._decode_intent(content)


class OpenAIResponsesIntentParser(_RemoteIntentParser):
    """Future provider adapter using OpenAI Responses + Structured Outputs."""

    name = "openai"

    def parse(self, instruction: str) -> NavigationIntent:
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": instruction},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "navigation_intent",
                    "strict": True,
                    "schema": INTENT_JSON_SCHEMA,
                }
            },
            "store": False,
            "max_output_tokens": 300,
        }
        response = self.client.post(f"{self.base_url}/responses", self.api_key, payload)
        content = self._extract_output_text(response)
        return self._decode_intent(content)

    @staticmethod
    def _extract_output_text(response: dict[str, Any]) -> str:
        direct = response.get("output_text")
        if isinstance(direct, str) and direct:
            return direct
        for item in response.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        return text
        raise ProviderError("OpenAI response has no output_text")


class FallbackIntentParser(IntentParser):
    def __init__(self, primary: IntentParser, fallback: IntentParser) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}+{fallback.name}-fallback"

    def parse(self, instruction: str) -> NavigationIntent:
        try:
            return self.primary.parse(instruction)
        except ProviderError:
            return self.fallback.parse(instruction)


def create_intent_parser(
    provider: str,
    places: PlaceDatabase,
    *,
    allow_rule_fallback: bool | None = None,
) -> IntentParser:
    load_dotenv()
    provider = provider.strip().casefold()
    rule = RuleBasedIntentParser(places)
    if provider == "rule":
        return rule
    if allow_rule_fallback is None:
        allow_rule_fallback = os.getenv("LLM_ALLOW_RULE_FALLBACK", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    try:
        if provider == "deepseek":
            parser: IntentParser = DeepSeekIntentParser(
                places,
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            )
        elif provider == "openai":
            parser = OpenAIResponsesIntentParser(
                places,
                api_key=os.getenv("OPENAI_API_KEY", ""),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            )
        else:
            raise ConfigurationError(f"Unknown LLM provider: {provider}")
    except ConfigurationError:
        if allow_rule_fallback:
            return rule
        raise
    return FallbackIntentParser(parser, rule) if allow_rule_fallback else parser
