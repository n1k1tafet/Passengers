"""Шина сообщений и базовый агент.

Обмен между агентами реализован явно: каждый вызов проходит через шину и
попадает в упорядоченный журнал. Журнал сохраняется вместе с рекомендацией —
именно он предъявляется инженеру как «логика принятия решения».

Шина СИНХРОННАЯ и ОДНОПОТОЧНАЯ намеренно: любая параллельность внесла бы
недетерминированность порядка сообщений, а ТЗ требует воспроизводимости
«на одних и тех же данных».
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class Message:
    seq: int
    sender: str
    recipient: str
    topic: str
    payload: Any
    elapsed_ms: float = 0.0

    MAX_LIST_IN_TRACE = 8

    def as_dict(self) -> dict:
        p = self.payload
        if isinstance(p, (list, tuple)):
            items = list(p)
            head = [(x.as_dict(compact=True) if _compactable(x)
                     else (x.as_dict() if hasattr(x, "as_dict") else x))
                    for x in items[:self.MAX_LIST_IN_TRACE]]
            p = {"всего вариантов": len(items),
                 "показаны первые": len(head), "варианты": head}
        elif hasattr(p, "as_dict"):
            p = p.as_dict()
        elif isinstance(p, dict):
            p = {k: (v.as_dict() if hasattr(v, "as_dict") else v) for k, v in p.items()}
        return {"seq": self.seq, "sender": self.sender, "recipient": self.recipient,
                "topic": self.topic, "elapsed_ms": round(self.elapsed_ms, 1), "payload": p}


class Bus:
    """Журналируемая шина запрос/ответ."""

    def __init__(self):
        self.log: list[Message] = []
        self._seq = 0
        self._agents: dict[str, "Agent"] = {}

    def register(self, agent: "Agent") -> None:
        self._agents[agent.name] = agent
        agent.bus = self

    def _next(self) -> int:
        self._seq += 1
        return self._seq

    def request(self, sender: str, recipient: str, topic: str, **kwargs) -> Any:
        self.log.append(Message(self._next(), sender, recipient, topic + ".request",
                                {k: _short(v) for k, v in kwargs.items()}))
        agent = self._agents[recipient]
        t0 = time.perf_counter()
        result = agent.handle(topic, **kwargs)
        dt = (time.perf_counter() - t0) * 1000.0
        self.log.append(Message(self._next(), recipient, sender, topic + ".response",
                                result, elapsed_ms=dt))
        return result

    def note(self, sender: str, topic: str, payload: Any) -> None:
        """Информационное сообщение без ответа (например, конфликт целей)."""
        self.log.append(Message(self._next(), sender, "*", topic, payload))

    def trace(self) -> list[dict]:
        return [m.as_dict() for m in self.log]

    def trace_json(self) -> str:
        return json.dumps(self.trace(), ensure_ascii=False, indent=1, default=str)


def _compactable(x: Any) -> bool:
    try:
        import inspect
        return "compact" in inspect.signature(x.as_dict).parameters
    except Exception:
        return False


def _short(v: Any) -> Any:
    """Компактное представление аргумента для журнала."""
    if hasattr(v, "as_dict"):
        return v.as_dict()
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, (pd.DataFrame, pd.Series)):
        return f"<{type(v).__name__} shape={getattr(v, 'shape', None)}>"
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {k: _short(x) for k, x in list(v.items())[:20]}
    if isinstance(v, (list, tuple)):
        return [_short(x) for x in list(v)[:20]]
    return f"<{type(v).__name__}>"


class Agent:
    """Базовый агент. Роль и краткое описание попадают в отчёт и на дашборд."""

    name: str = "agent"
    role: str = ""

    def __init__(self, ctx: "AgentContext"):
        self.ctx = ctx
        self.bus: Bus | None = None

    def handle(self, topic: str, **kwargs) -> Any:
        method = getattr(self, "on_" + topic.replace(".", "_"), None)
        if method is None:
            raise KeyError(f"{self.name}: нет обработчика для темы '{topic}'")
        return method(**kwargs)


@dataclass
class AgentContext:
    """Общий контекст: конфигурация, модели, справочники. Только чтение."""
    config: Any
    bundle: Any
    plant: Any
    store: Any
    extras: dict = field(default_factory=dict)


def state_hash(state, extra: dict | None = None) -> str:
    """Устойчивый отпечаток состояния — основа проверки воспроизводимости.

    Значения округляются до 6 знаков: одинаковое состояние процесса даёт
    одинаковый хэш, а значит и одинаковую рекомендацию. Тест
    ``tests/test_determinism.py`` это проверяет.
    """
    payload = {
        "t": state.t.isoformat(),
        "tags": {k: (None if v is None or not np.isfinite(v) else round(float(v), 6))
                 for k, v in sorted(state.tags.items())},
        "lab": {f"{k[0]}:{k[1]}@{k[2]}": round(float(v.value), 6)
                for k, v in sorted(state.lab.items())},
    }
    if extra:
        payload["extra"] = extra
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
