"""Сборка мультиагентной системы и запуск цикла принятия решения.

Точка входа для всего остального кода: CLI, сценарии, дашборд и тесты
используют ``DarkFactorySystem``.

Важное свойство: система НЕ хранит состояния между циклами. Всё, что нужно,
приходит из среза ``AsOfStore.snapshot(t)`` и из предобученного набора моделей
``model_bundle``. Поэтому один и тот же срез всегда даёт один и тот же ответ —
свойство проверяется тестом ``tests/test_determinism.py``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, REPORTS, load_config
from .featurestore import AsOfStore
from .io.ingest import load_processed
from .models import va_formulas as va
from .models.bias import apply_online_bias
from .models.fit import (TAG_DP, TAG_H2, TAG_LOAD, TAG_PRESS, TAG_QUENCH,
                         TAG_S_ONLINE, TAG_TEMP, TRAIN_END, ModelBundle,
                         _online_sulfur)
from .agents.base import AgentContext, Bus
from .agents.blending import BlendingAgent, Tank
from .agents.data_quality import DataQualityAgent
from .agents.economics import EconomicsAgent
from .agents.explainer import ExplainerAgent
from .agents.feedstock import FeedstockAgent
from .agents.optimizer import OptimizerAgent
from .agents.orchestrator import OrchestratorAgent
from .agents.quality import QualityAgent
from .agents.reliability import ReliabilityAgent

REGIME_TAGS = [TAG_TEMP, TAG_LOAD, TAG_PRESS, TAG_H2, TAG_DP, TAG_QUENCH]


@dataclass
class CycleResult:
    recommendation: object
    trace: list[dict]
    state_summary: dict

    def to_json(self) -> str:
        return json.dumps({"recommendation": self.recommendation.as_dict(),
                           "state": self.state_summary, "trace": self.trace},
                          ensure_ascii=False, indent=1, default=str)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path


class DarkFactorySystem:
    """Мультиагентная система управления производством дизельного топлива."""

    def __init__(self, store: AsOfStore, bundle: ModelBundle, extras: dict | None = None):
        self.store = store
        self.bundle = bundle
        self.config = load_config()
        self.plant = bundle.to_plant_model()
        self.ctx = AgentContext(config=self.config, bundle=bundle, plant=self.plant,
                                store=store, extras=extras or {})
        self.agents = {
            "data": DataQualityAgent(self.ctx),
            "feed": FeedstockAgent(self.ctx),
            "quality": QualityAgent(self.ctx),
            "reliability": ReliabilityAgent(self.ctx),
            "economics": EconomicsAgent(self.ctx),
            "optimizer": OptimizerAgent(self.ctx),
            "blending": BlendingAgent(self.ctx),
            "explainer": ExplainerAgent(self.ctx),
        }
        self.orchestrator = OrchestratorAgent(self.ctx)
        self.ctx.extras["agents"] = self.agents
        self.ctx.extras["reliability_fn"] = \
            lambda st, action: self.agents["reliability"].on_assess(st, action)

    # ----------------------------------------------------------------- сборка
    @classmethod
    def build(cls, processed_dir: Path | None = None) -> "DarkFactorySystem":
        store = AsOfStore.load(processed_dir)
        bundle = ModelBundle.load()
        extras = _build_extras(store, bundle)
        return cls(store, bundle, extras)

    # ------------------------------------------------------------------ цикл
    def run_at(self, t, grade: str = "DT_SUMMER", horizon_min: int | None = None,
               overrides: dict | None = None, lab_overrides: dict | None = None,
               tanks: list[Tank] | None = None, plan_blend: bool = True) -> CycleResult:
        state = self.store.snapshot(t, overrides=overrides, lab_overrides=lab_overrides)
        bus = Bus()
        for a in self.agents.values():
            bus.register(a)
        bus.register(self.orchestrator)
        self.ctx.extras["tanks"] = tanks      # None => состав по умолчанию
        # актуальная коррекция смещения серы, вычисленная причинно (available_at <= t)
        bias_series = self.ctx.extras.get("sulfur_bias_series")
        if bias_series is not None:
            self.ctx.extras["sulfur_bias"] = float(bias_series.asof(state.t)) \
                if len(bias_series) else 0.0
        rec = self.orchestrator.run_cycle(state, grade=grade, horizon_min=horizon_min,
                                          plan_blend=plan_blend)
        return CycleResult(recommendation=rec, trace=bus.trace(),
                           state_summary=state.summary())

    def agent_roster(self) -> list[dict]:
        rows = [{"name": a.name, "role": a.role} for a in self.agents.values()]
        rows.append({"name": self.orchestrator.name, "role": self.orchestrator.role})
        return rows


# ---------------------------------------------------------------------------
def _build_extras(store: AsOfStore, bundle: ModelBundle) -> dict:
    """Предрасчёты, общие для всех циклов (статистика режима, сигмы ВАК, смещение)."""
    tele, flags, lab = load_processed()
    train = tele.loc[:TRAIN_END]

    # --- статистика нормального режима для расстояния Махаланобиса
    sub = train[REGIME_TAGS].rolling(6, min_periods=3).mean().dropna()
    mu = sub.mean().to_numpy()
    sd = sub.std().to_numpy()
    z = ((sub - sub.mean()) / (sub.std() + 1e-9))
    corr = np.corrcoef(z.to_numpy().T)
    corr = corr + np.eye(len(REGIME_TAGS)) * 1e-6
    regime_stats = {"tags": REGIME_TAGS, "mu": mu, "sd": sd, "corr": corr}

    # --- разброс ВАК по показателям (для интервалов «прочих» параметров)
    hist = va.evaluate_history(tele, lab)
    param_sigma: dict[str, float] = {}
    for name, (_, stream, param, _fn) in va.SPECS.items():
        if stream != "HT_PRODUCT":
            continue
        g = lab[(lab.stream == stream) & (lab.param == param) & (lab.source == "LIMS")]
        g = g[g.measured_at < TRAIN_END].drop_duplicates(subset="measured_at")
        if len(g) < 30:
            continue
        y = pd.Series(g["value"].to_numpy(), index=pd.DatetimeIndex(g["measured_at"])).sort_index()
        q1, q3 = y.quantile([0.01, 0.99])
        y = y[(y >= q1 - 3 * (q3 - q1)) & (y <= q3 + 3 * (q3 - q1))]
        base = hist[name].dropna()
        pred = pd.Series(np.interp(y.index.view("i8"), base.index.view("i8"),
                                   base.to_numpy()), index=y.index)
        r = (y - pred - bundle.va_bias.get(name, 0.0)).dropna()
        if len(r) >= 30:
            qq1, qq3 = np.quantile(r, [0.25, 0.75])
            param_sigma[param] = float((qq3 - qq1) / 1.349)
    # показатели без ВАК — берём разброс самой лаборатории (консервативно)
    for param in ("CetaneNumber", "FlashPoint"):
        g = lab[(lab.stream == "HT_PRODUCT") & (lab.param == param)
                & (lab.measured_at < TRAIN_END)]
        if len(g) >= 20:
            qq1, qq3 = np.quantile(g["value"], [0.25, 0.75])
            param_sigma[param] = float(max((qq3 - qq1) / 1.349, 0.3))

    # --- причинная коррекция смещения поточной серы по ЛИМС
    s_on = _online_sulfur(tele, lab).dropna()
    lp = lab[(lab.stream == "HT_PRODUCT") & (lab.param == "Mg.Sulfur")
             & (lab.source == "LIMS")]
    lp = lp[(lp.value > 0.2) & (lp.value < 60)].drop_duplicates(subset="measured_at")
    lp = lp.sort_values("measured_at")
    truth = pd.Series(lp["value"].to_numpy(), index=pd.DatetimeIndex(lp["measured_at"]))
    bias_series = apply_online_bias(s_on, truth, pd.DatetimeIndex(lp["available_at"]),
                                    halflife_hours=48.0, max_abs=4.0, forget_hours=720.0)
    return {"regime_stats": regime_stats, "param_sigma": param_sigma,
            "sulfur_bias_series": bias_series, "sulfur_bias": 0.0}
