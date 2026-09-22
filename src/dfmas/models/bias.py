"""Онлайн-коррекция смещения виртуального анализатора по лаборатории.

Приём стандартный для НПЗ: виртуальный анализатор даёт форму кривой, а
лаборатория — абсолютный уровень. Коррекция — экспоненциально сглаженное
смещение ``b(t) = EWMA(lab - base)``, которое обновляется ТОЛЬКО в моменты,
когда результат анализа стал доступен (``available_at``), и ограничено по
модулю, чтобы единичный промах лаборатории не «увёл» модель.

Почему это важно для задачи: ТЗ требует приоритета ЛИМС над ПАК и ВАК
(«лабораторный результат считается контрольным фактом»). Коррекция — это и
есть механизм, который такой приоритет реализует численно, а не декларативно.

Проверка на выданных данных (reports/02_va_validation.md) показывает
сокращение MAE в 2–6 раз для половины показателей.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BiasState:
    value: float = 0.0
    n_updates: int = 0
    last_update: pd.Timestamp | None = None
    last_residual: float = float("nan")


class OnlineBiasCorrector:
    """EWMA-коррекция смещения с ограничением и «забыванием» по времени."""

    def __init__(self, halflife_hours: float = 24.0, max_abs: float = 5.0,
                 forget_hours: float = 240.0):
        self.halflife = float(halflife_hours)
        self.max_abs = float(max_abs)
        self.forget_hours = float(forget_hours)
        self.state = BiasState()

    def update(self, t: pd.Timestamp, residual: float) -> float:
        """Поглощает новый лабораторный результат (residual = lab - base)."""
        if not np.isfinite(residual):
            return self.state.value
        if self.state.last_update is None:
            w = 1.0
        else:
            dt_h = max(0.0, (t - self.state.last_update).total_seconds() / 3600.0)
            w = 1.0 - 0.5 ** (dt_h / self.halflife) if self.halflife > 0 else 1.0
            w = float(np.clip(w, 0.05, 1.0))
        new = (1 - w) * self.state.value + w * float(residual)
        self.state = BiasState(value=float(np.clip(new, -self.max_abs, self.max_abs)),
                               n_updates=self.state.n_updates + 1,
                               last_update=t, last_residual=float(residual))
        return self.state.value

    def value_at(self, t: pd.Timestamp) -> float:
        """Текущая коррекция; затухает к нулю, если лаборатория давно молчит."""
        if self.state.last_update is None:
            return 0.0
        age_h = (t - self.state.last_update).total_seconds() / 3600.0
        if age_h <= 0:
            return self.state.value
        decay = 0.5 ** (age_h / self.forget_hours) if self.forget_hours > 0 else 1.0
        return float(self.state.value * decay)

    @property
    def confidence(self) -> float:
        """0..1 — насколько коррекция обоснована числом обновлений."""
        return float(1.0 - 0.5 ** (self.state.n_updates / 3.0))


def apply_online_bias(base: pd.Series, lab: pd.Series, lab_available: pd.Series,
                      halflife_hours: float = 24.0, max_abs: float = 5.0,
                      forget_hours: float = 240.0) -> pd.Series:
    """Прогоняет коррекцию по истории строго в хронологическом порядке.

    ``base``           — ряд ВАК на сетке телеметрии;
    ``lab``            — лабораторные значения (индекс = момент ОТБОРА);
    ``lab_available``  — момент, когда результат стал доступен.

    Возвращает ряд коррекции b(t) на сетке ``base``. Гарантия отсутствия
    утечки: b(t) обновляется только событиями с ``available_at <= t``.
    """
    corr = OnlineBiasCorrector(halflife_hours, max_abs, forget_hours)
    events = pd.DataFrame({"measured_at": lab.index, "value": lab.to_numpy(),
                           "available_at": pd.DatetimeIndex(lab_available)})
    events = events.sort_values("available_at")
    base_idx = base.index
    out = np.zeros(len(base_idx))
    ei, n = 0, len(events)
    for i, t in enumerate(base_idx):
        while ei < n and events["available_at"].iloc[ei] <= t:
            ev = events.iloc[ei]
            b0 = base.asof(ev["measured_at"])
            if np.isfinite(b0):
                corr.update(ev["available_at"], float(ev["value"]) - float(b0))
            ei += 1
        out[i] = corr.value_at(t)
    return pd.Series(out, index=base_idx, name="bias")
