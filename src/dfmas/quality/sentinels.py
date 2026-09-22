"""Обнаружение непригодных значений в архиве историка.

Зачем отдельный модуль
----------------------
В выданных файлах «плохие» значения не помечены NaN — историк подставляет
числовые коды. Если их не убрать, любая модель будет обучаться на мусоре, а
агент качества — выдавать рекомендации по несуществующему режиму.

Найдено три механизма порчи данных (см. reports/01_tag_audit.md):

1. **Жёсткий сентинел 307.0.** Подтверждён экспертом («307 — это выброс»).
   Встречается в 24 из 26 тегов установки 24-2000 и в 69 из 72 тегов АВТ.
   Тег D10 (плотность нефти) состоит из него на 99.995 %.
2. **Цифровые состояния прибора** — целые значения (251, 252, 240, 213, 10,
   99.991 ...), которые повторяются подозрительно часто и лежат у границы
   диапазона. Детектируются автоматически, а не списком.
3. **Залипание (flatline)** — прибор отдаёт одно и то же значение часами.

Каждое значение получает флаг качества, а не молча удаляется: агент данных
должен уметь объяснить оператору, ЧТО именно испорчено.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HARD_SENTINELS = (307.0,)

#: Коды качества значения
GOOD, SENTINEL, DIGITAL_STATE, FLATLINE, OUT_OF_RANGE, MISSING = 0, 1, 2, 3, 4, 5
FLAG_NAMES = {GOOD: "good", SENTINEL: "sentinel", DIGITAL_STATE: "digital_state",
              FLATLINE: "flatline", OUT_OF_RANGE: "out_of_range", MISSING: "missing"}


@dataclass
class TagProfile:
    """Паспорт тега, построенный по обучающему отрезку истории."""
    tag: str
    digital_states: tuple[float, ...]
    lo: float               # нижняя модельная граница (P0.5 чистых значений)
    hi: float               # верхняя модельная граница (P99.5 чистых значений)
    median: float
    mad: float              # робастный разброс
    good_share: float

    def to_dict(self) -> dict:
        return {"tag": self.tag, "digital_states": list(self.digital_states),
                "lo": self.lo, "hi": self.hi, "median": self.median,
                "mad": self.mad, "good_share": self.good_share}


def detect_digital_states(values: np.ndarray, min_share: float = 0.003,
                          quantile_band: float = 0.005) -> tuple[float, ...]:
    """Ищет «цифровые состояния» прибора без заранее заданного списка.

    Значение считается цифровым состоянием, если оно
      * повторяется чаще ``min_share`` доли выборки,
      * целое (историки кодируют статусы целыми),
      * и при этом лежит за пределами центральной части распределения.

    Такой критерий находит 251/252/240/213/10, не трогая нормальные уставки.
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return ()
    v = v[~np.isin(v, HARD_SENTINELS)]
    if v.size == 0:
        return ()
    lo, hi = np.quantile(v, [quantile_band, 1 - quantile_band])
    vals, counts = np.unique(np.round(v, 6), return_counts=True)
    share = counts / v.size
    mask = (share >= min_share) & (np.abs(vals - np.round(vals)) < 1e-9) & ((vals >= hi) | (vals <= lo))
    return tuple(float(x) for x in vals[mask])


def build_profile(series: pd.Series, tag: str) -> TagProfile:
    v = series.to_numpy(dtype=float)
    states = detect_digital_states(v)
    bad = ~np.isfinite(v) | np.isin(v, HARD_SENTINELS) | np.isin(v, states)
    clean = v[~bad]
    if clean.size < 10:
        return TagProfile(tag, states, np.nan, np.nan, np.nan, np.nan, 0.0)
    lo, hi = np.quantile(clean, [0.005, 0.995])
    med = float(np.median(clean))
    mad = float(np.median(np.abs(clean - med)) * 1.4826)
    return TagProfile(tag, states, float(lo), float(hi), med, mad,
                      float(clean.size / v.size))


def flag_series(series: pd.Series, profile: TagProfile,
                flatline_samples: int = 18) -> pd.Series:
    """Возвращает Series кодов качества той же длины, что и вход."""
    v = series.to_numpy(dtype=float)
    flags = np.full(v.shape, GOOD, dtype=np.int8)
    flags[~np.isfinite(v)] = MISSING
    flags[np.isin(v, HARD_SENTINELS)] = SENTINEL
    if profile.digital_states:
        flags[np.isin(v, profile.digital_states)] = DIGITAL_STATE
    if np.isfinite(profile.lo):
        span = profile.hi - profile.lo
        pad = 0.10 * span if span > 0 else 1.0
        oor = (v < profile.lo - pad) | (v > profile.hi + pad)
        flags[(flags == GOOD) & oor] = OUT_OF_RANGE
    # залипание: одно и то же значение >= flatline_samples подряд (векторно)
    if flatline_samples and v.size > flatline_samples:
        same = np.r_[False, np.diff(v) == 0]
        n = v.size
        pos = np.arange(n)
        run = pos - np.maximum.accumulate(np.where(~same, pos, 0))
        ends = run >= (flatline_samples - 1)
        # «растянуть» отметку назад на длину окна
        stuck = (pd.Series(ends[::-1]).rolling(flatline_samples, min_periods=1)
                 .max().to_numpy()[::-1] > 0)
        flags[(flags == GOOD) & stuck] = FLATLINE
    return pd.Series(flags, index=series.index, name=f"{series.name}__flag")


def clean_frame(df: pd.DataFrame, profiles: dict[str, TagProfile],
                flatline_samples: int = 18) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (очищенные значения, матрица флагов качества)."""
    values, flags = {}, {}
    for col in df.columns:
        prof = profiles.get(col)
        if prof is None:
            prof = build_profile(df[col], col)
            profiles[col] = prof
        fl = flag_series(df[col], prof, flatline_samples)
        flags[col] = fl.to_numpy()
        v = df[col].to_numpy(dtype=float).copy()
        v[fl.to_numpy() != GOOD] = np.nan
        values[col] = v
    return (pd.DataFrame(values, index=df.index),
            pd.DataFrame(flags, index=df.index).astype(np.int8))
