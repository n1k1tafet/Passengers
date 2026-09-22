"""Развёртка решения по входным условиям — доказательство «мягкости» сценариев.

Эксперт потребовал: «Сценарии не должны быть жёсткими, т.е. должна быть
возможность изменения входных данных и результаты расчёта должны меняться в
зависимости как от качества сырья, так и от задания на качество продукции».

Этот модуль проверяет требование напрямую: он прогоняет ОДИН И ТОТ ЖЕ момент
времени через полный мультиагентный цикл при разных значениях серы сырья и
разных марках товарного топлива и печатает, как меняются решение, требуемая
температура и рецептура блендинга.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .system import DarkFactorySystem


def feed_sulfur_sweep(system: DarkFactorySystem, at: str,
                      values: list[float] | None = None,
                      grade: str = "DT_SUMMER") -> pd.DataFrame:
    values = values or [0.70, 0.85, 1.00, 1.15, 1.30, 1.45, 1.60, 1.80]
    rows = []
    for v in values:
        res = system.run_at(at, grade=grade,
                            lab_overrides={("HT_FEED", "Mass.Sulfur"): v})
        rec = res.recommendation
        q = rec.option.predictions["quality"] if rec.option else None
        s = next((i for i in q.items if i.param == "Mg.Sulfur"), None) if q else None
        sp = rec.setpoint_plan
        # прогноз ПРИ БЕЗДЕЙСТВИИ — чистое влияние сырья, без наложения действия
        hold_pred = _hold_forecast(system, at, grade, v)
        rows.append({
            "сера сырья, % масс.": v,
            "решение": rec.decision,
            "действие": rec.action_label,
            "ΔT, °C": rec.option.action.d_temp_c if rec.option else np.nan,
            "прогноз при бездействии, мг/кг": hold_pred,
            "прогноз серы, мг/кг": s.forecast if s else np.nan,
            "риск, %": round(100 * s.exceed_prob, 1) if s and s.exceed_prob is not None else np.nan,
            "требуемая компенсация, °C": sp.delta_total_c if sp else np.nan,
            "маржа, ₽/ч": rec.option.metrics.margin_rub_h if rec.option else np.nan,
        })
    return pd.DataFrame(rows)


def _hold_forecast(system, at: str, grade: str, feed_value: float) -> float:
    """Прогноз серы без вмешательства — изолирует влияние сырья от действия."""
    from .models.process_model import Action
    state = system.store.snapshot(at, lab_overrides={("HT_FEED", "Mass.Sulfur"): feed_value})
    qa = system.agents["quality"]
    qa.reset_cache()
    feed = system.agents["feed"].on_assess(state)
    rep = qa.on_assess(state, grade, feed, Action(), system.config.horizon_min)
    item = next((i for i in rep.items if i.param == "Mg.Sulfur"), None)
    return float(item.forecast) if item else float("nan")


def grade_sweep(system: DarkFactorySystem, at: str,
                grades: list[str] | None = None) -> pd.DataFrame:
    grades = grades or ["HT_PRODUCT", "DT_SUMMER", "DT_WINTER"]
    rows = []
    for g in grades:
        res = system.run_at(at, grade=g)
        rec = res.recommendation
        bl = rec.blend
        rows.append({
            "марка": g,
            "название": system.config.grade(g)["title"],
            "решение": rec.decision,
            "действие": rec.action_label,
            "рецептура допустима": None if bl is None else bl.feasible,
            "сера смеси": None if bl is None or not bl.feasible else round(bl.blended["Mg.Sulfur"], 2),
            "ЦЧ смеси": None if bl is None or not bl.feasible else round(bl.blended["CetaneNumber"], 2),
            "присадка, % об.": None if bl is None or not bl.feasible else round(100 * bl.additive_share, 3),
            "цена смеси, ₽/т": None if bl is None or not bl.feasible else round(bl.cost_rub_t),
        })
    return pd.DataFrame(rows)
