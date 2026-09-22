"""Конформные интервалы прогноза и риск нарушения спецификации.

Зачем
-----
ТЗ просит оценку неопределённости, а жёсткое ограничение по сере (<= 10 мг/кг)
требует не точечного прогноза, а ВЕРОЯТНОСТИ выхода за предел. Точечная оценка
«будет 9.6» бесполезна, если разброс модели ±1.5.

Метод — split conformal prediction с блочным разбиением по времени.
Выбран потому, что:
  * не требует предположения о нормальности остатков (они асимметричны);
  * даёт гарантированное покрытие при обмене (exchangeability) внутри блока;
  * калибруется на ОТЛОЖЕННОМ по времени куске, то есть честно.

Риск нарушения считается как доля калибровочных остатков, при которых
прогноз + остаток нарушил бы предел. Это эмпирическая вероятность, а не
результат подгонки распределения.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConformalCalibration:
    """Калибровочная выборка остатков (прогноз минус факт) для одного показателя."""
    param: str
    horizon_min: int
    residuals: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    n: int = 0

    @classmethod
    def from_residuals(cls, param: str, horizon_min: int, residuals) -> "ConformalCalibration":
        r = np.asarray(residuals, dtype=float)
        r = r[np.isfinite(r)]
        return cls(param=param, horizon_min=horizon_min, residuals=np.sort(r), n=int(r.size))

    def interval(self, point: float, level: float = 0.90) -> tuple[float, float]:
        if self.n < 20 or not np.isfinite(point):
            return (float("nan"), float("nan"))
        a = (1.0 - level) / 2.0
        lo, hi = np.quantile(self.residuals, [a, 1.0 - a])
        # residual = факт - прогноз, поэтому интервал факта = точка + квантили
        return (float(point + lo), float(point + hi))

    def exceed_probability(self, point: float, limit: float, side: str = "upper") -> float:
        """Эмпирическая вероятность нарушить предел при данном точечном прогнозе."""
        if self.n < 20 or not np.isfinite(point):
            return float("nan")
        realised = point + self.residuals
        return float((realised > limit).mean() if side == "upper" else (realised < limit).mean())

    def sigma(self) -> float:
        if self.n < 5:
            return float("nan")
        q1, q3 = np.quantile(self.residuals, [0.25, 0.75])
        return float((q3 - q1) / 1.349)

    def as_dict(self) -> dict:
        return {"param": self.param, "horizon_min": self.horizon_min, "n": self.n,
                "sigma": None if not np.isfinite(self.sigma()) else round(self.sigma(), 4),
                "q05": None if self.n < 20 else round(float(np.quantile(self.residuals, .05)), 4),
                "q95": None if self.n < 20 else round(float(np.quantile(self.residuals, .95)), 4)}


def widen(cal: ConformalCalibration, factor: float) -> ConformalCalibration:
    """Расширяет интервал, когда модель применяется вне области калибровки.

    Используется агентом качества: если рекомендация выводит режим за пределы
    наблюдавшегося диапазона, интервал расширяется пропорционально величине
    экстраполяции, и риск нарушения растёт — система становится осторожнее.
    """
    return ConformalCalibration(param=cal.param, horizon_min=cal.horizon_min,
                                residuals=cal.residuals * float(max(1.0, factor)),
                                n=cal.n)
