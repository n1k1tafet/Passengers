"""Контракты обмена между агентами.

Все сообщения — неизменяемые датаклассы с методом ``as_dict``. Это даёт три
вещи, которые прямо требует ТЗ:

* **явный обмен информацией** — видно, кто что передал и в каком виде;
* **проверяемость** — весь обмен сериализуется в JSON и сохраняется вместе
  с рекомендацией, поэтому логику решения можно воспроизвести post factum;
* **детерминизм** — агенты не держат скрытого состояния между циклами,
  результат зависит только от среза данных и конфигурации.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


def _r(v, nd=3):
    if v is None:
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), nd)
    return v


class Severity(str, Enum):
    OK = "ok"
    WARN = "warn"
    ALARM = "alarm"


# --------------------------------------------------------------------- данные
@dataclass
class DataIssue:
    code: str
    detail: str
    severity: Severity
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"code": self.code, "detail": self.detail,
                "severity": self.severity.value, "tags": self.tags}


@dataclass
class DataQualityReport:
    t: pd.Timestamp
    usable: bool
    score: float                       # 0..1 — интегральная пригодность данных
    issues: list[DataIssue] = field(default_factory=list)
    freshness: dict[str, float] = field(default_factory=dict)   # источник -> возраст, ч
    cross_check: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"t": self.t.isoformat(), "usable": self.usable, "score": _r(self.score),
                "issues": [i.as_dict() for i in self.issues],
                "freshness_h": {k: _r(v, 2) for k, v in self.freshness.items()},
                "cross_check": {k: _r(v) for k, v in self.cross_check.items()}}


# --------------------------------------------------------------------- сырьё
@dataclass
class FeedReport:
    t: pd.Timestamp
    sulfur_wt: float                   # сера сырья гидроочистки, % масс.
    sulfur_source: str                 # LIMS | ВАК-АВТ | опорное значение
    sulfur_age_h: float
    sulfur_confidence: float           # 0..1
    sulfur_prev_wt: float              # сера сырья, уже отработавшая в реакторе
    unrealized_fraction: float         # доля изменения сырья, ещё не дошедшая до продукта
    t95_c: float
    d15: float
    avt_diesel_flow_t_h: float
    trend_note: str = ""

    def as_dict(self) -> dict:
        return {"t": self.t.isoformat(), "sulfur_wt": _r(self.sulfur_wt, 4),
                "sulfur_source": self.sulfur_source, "sulfur_age_h": _r(self.sulfur_age_h, 1),
                "sulfur_confidence": _r(self.sulfur_confidence),
                "sulfur_prev_wt": _r(self.sulfur_prev_wt, 4),
                "unrealized_fraction": _r(self.unrealized_fraction, 3),
                "t95_c": _r(self.t95_c, 1), "d15": _r(self.d15, 1),
                "avt_diesel_flow_t_h": _r(self.avt_diesel_flow_t_h, 1),
                "trend_note": self.trend_note}


# ------------------------------------------------------------------- качество
@dataclass
class QualityItem:
    param: str
    ru_name: str
    value: float
    unit: str
    source: str
    age_h: float
    forecast: float | None = None
    lo: float | None = None
    hi: float | None = None
    limit: float | None = None
    limit_kind: str = ""               # "max" | "min"
    hard: bool = True
    exceed_prob: float | None = None   # вероятность, что ЛАБОРАТОРИЯ покажет нарушение
    margin: float | None = None        # запас до предела в единицах показателя
    risk_process: float | None = None  # вероятность нарушения по самому процессу
    risk_detail: dict | None = None    # разложение неопределённости

    def as_dict(self) -> dict:
        d = {k: _r(v) if isinstance(v, float) else v for k, v in self.__dict__.items()}
        if self.risk_detail:
            d["risk_detail"] = {k: _r(v) for k, v in self.risk_detail.items()}
        return d


@dataclass
class QualityReport:
    t: pd.Timestamp
    grade: str
    items: list[QualityItem]
    confidence: float
    horizon_min: int
    worst_param: str | None = None
    worst_prob: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"t": self.t.isoformat(), "grade": self.grade,
                "items": [i.as_dict() for i in self.items],
                "confidence": _r(self.confidence), "horizon_min": self.horizon_min,
                "worst_param": self.worst_param, "worst_prob": _r(self.worst_prob),
                "notes": self.notes}


# ------------------------------------------------------------------ надёжность
@dataclass
class ReliabilityReport:
    t: pd.Timestamp
    severity_index: float              # 0..1
    severity: Severity
    factors: dict[str, float] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    catalyst_days_left: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"t": self.t.isoformat(), "severity_index": _r(self.severity_index),
                "severity": self.severity.value,
                "factors": {k: _r(v) for k, v in self.factors.items()},
                "bounds": {k: [_r(v[0]), _r(v[1])] for k, v in self.bounds.items()},
                "catalyst_days_left": _r(self.catalyst_days_left, 0),
                "notes": self.notes}


# ------------------------------------------------------------------- варианты
@dataclass
class OptionMetrics:
    quality_margin: float              # минимальный относительный запас по жёстким пределам
    worst_exceed_prob: float
    throughput_t_h: float
    cost_rub_h: float                   # переменные затраты
    severity_index: float
    feasible: bool                      # уровень 1: точечный прогноз в пределах
    margin_rub_h: float = float("nan")  # выручка минус переменные затраты
    violated: list[str] = field(default_factory=list)
    compliant: bool = True              # уровень 2: риск нарушения в допуске

    def as_dict(self) -> dict:
        return {k: ([str(x) for x in v] if isinstance(v, list) else _r(v))
                for k, v in self.__dict__.items()}


@dataclass
class Option:
    option_id: str
    action: Any                        # models.process_model.Action
    metrics: OptionMetrics
    predictions: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    rank: int = 0
    pareto: bool = False
    rationale: str = ""

    def as_dict(self, compact: bool = False) -> dict:
        """``compact`` — краткая форма для журнала обмена.

        Оптимизатор перебирает сотни вариантов; полный прогноз качества по
        каждому раздул бы журнал до мегабайт и сделал бы его нечитаемым.
        В журнал идёт краткая форма, в рекомендацию — полная.
        """
        base = {"option_id": self.option_id, "action": self.action.as_dict(),
                "action_label": self.action.label(), "metrics": self.metrics.as_dict(),
                "score": _r(self.score), "rank": self.rank, "pareto": self.pareto,
                "rationale": self.rationale}
        if compact:
            q = self.predictions.get("quality")
            if q is not None:
                s = next((i for i in q.items if i.param == "Mg.Sulfur"), None)
                if s is not None:
                    base["сера_прогноз"] = _r(s.forecast, 2)
                    base["риск"] = _r(s.exceed_prob, 3)
            return base
        base["predictions"] = {k: (v.as_dict() if hasattr(v, "as_dict") else _r(v))
                               for k, v in self.predictions.items()}
        return base


# -------------------------------------------------------------------- блендинг
@dataclass
class BlendComponent:
    name: str
    share: float
    sulfur_mg_kg: float
    t95_c: float
    cetane: float
    density: float
    cost_rub_t: float

    def as_dict(self) -> dict:
        return {k: _r(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


@dataclass
class BlendPlan:
    grade: str
    feasible: bool
    components: list[BlendComponent] = field(default_factory=list)
    additive_share: float = 0.0
    blended: dict[str, float] = field(default_factory=dict)
    cost_rub_t: float = float("nan")
    message: str = ""

    def as_dict(self) -> dict:
        return {"grade": self.grade, "feasible": self.feasible,
                "components": [c.as_dict() for c in self.components],
                "additive_share": _r(self.additive_share, 5),
                "blended": {k: _r(v) for k, v in self.blended.items()},
                "cost_rub_t": _r(self.cost_rub_t, 1), "message": self.message}


# -------------------------------------------------------------- рекомендация
@dataclass
class SetpointPlan:
    """План выхода на целевую уставку, если одного шага не хватает."""
    target_sulfur: float
    required_temp_c: float
    current_temp_c: float
    delta_total_c: float
    steps: list[dict] = field(default_factory=list)
    reachable: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        return {"target_sulfur": _r(self.target_sulfur, 2),
                "required_temp_c": _r(self.required_temp_c, 2),
                "current_temp_c": _r(self.current_temp_c, 2),
                "delta_total_c": _r(self.delta_total_c, 2),
                "steps": self.steps, "reachable": self.reachable, "note": self.note}


@dataclass
class Recommendation:
    t: pd.Timestamp
    decision: str                      # "hold" | "act" | "refuse"
    headline: str
    action_label: str
    option: Option | None
    alternatives: list[Option] = field(default_factory=list)
    constraint_table: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    expected_effect: dict[str, Any] = field(default_factory=dict)
    blend: BlendPlan | None = None
    setpoint_plan: SetpointPlan | None = None
    state_hash: str = ""

    def as_dict(self) -> dict:
        return {"t": self.t.isoformat(), "decision": self.decision,
                "headline": self.headline, "action_label": self.action_label,
                "option": self.option.as_dict() if self.option else None,
                "alternatives": [o.as_dict() for o in self.alternatives],
                "constraint_table": self.constraint_table,
                "confidence": _r(self.confidence), "reasons": self.reasons,
                "expected_effect": {k: (v.as_dict() if hasattr(v, "as_dict") else _r(v))
                                    for k, v in self.expected_effect.items()},
                "blend": self.blend.as_dict() if self.blend else None,
                "setpoint_plan": self.setpoint_plan.as_dict() if self.setpoint_plan else None,
                "state_hash": self.state_hash}
