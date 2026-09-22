"""Идентификация транспортного запаздывания MV -> CV.

Эксперт сообщил, что задержка между изменением технологического параметра и
показателем качества составляет от 0 до 3 часов, а величину «желательно найти
самостоятельно». Здесь это и делается.

Метод
-----
Работаем с ПРИРАЩЕНИЯМИ, а не с уровнями: ряды нестационарны (сезонность,
смены режима, дрейф катализатора), и корреляция уровней даёт ложные пики.
Для каждого лага τ считаем корреляцию Δ_h MV(t-τ) и Δ_h CV(t), где h — окно
усреднения. Пик |корреляции| даёт оценку τ. Доверительный интервал — блочным
бутстрапом (блоки по неделе), чтобы учесть автокорреляцию.

Устойчивость: если пик не отличается от шума (перекрытие ДИ с нулём), лаг
помечается как неидентифицируемый и модель отклика для этой пары не строится.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STEP_MIN = 10          # шаг телеметрии


@dataclass
class DeadTime:
    mv: str
    cv: str
    lag_min: int
    corr: float
    lag_lo_min: int
    lag_hi_min: int
    identifiable: bool
    n: int

    def as_dict(self) -> dict:
        return {"mv": self.mv, "cv": self.cv, "lag_min": self.lag_min,
                "corr": round(float(self.corr), 4),
                "ci_min": [self.lag_lo_min, self.lag_hi_min],
                "identifiable": self.identifiable, "n": int(self.n)}


def identify(mv: pd.Series, cv: pd.Series, max_lag_min: int = 180,
             diff_window_min: int = 60, n_boot: int = 120,
             block_days: int = 7, seed: int = 0) -> DeadTime:
    h = max(1, diff_window_min // STEP_MIN)
    x = mv.rolling(h, min_periods=max(1, h // 2)).mean()
    y = cv.rolling(h, min_periods=max(1, h // 2)).mean()
    dx, dy = x.diff(h), y.diff(h)
    lags = list(range(0, max_lag_min // STEP_MIN + 1))
    d = pd.DataFrame({"dx": dx, "dy": dy}).dropna()
    if len(d) < 2000:
        return DeadTime(mv.name, cv.name, 0, 0.0, 0, 0, False, len(d))

    def corr_at(frame: pd.DataFrame, lag: int) -> float:
        a = frame["dx"].shift(lag)
        m = a.notna() & frame["dy"].notna()
        if m.sum() < 500:
            return 0.0
        aa, bb = a[m].to_numpy(), frame["dy"][m].to_numpy()
        sa, sb = aa.std(), bb.std()
        return 0.0 if sa == 0 or sb == 0 else float(np.corrcoef(aa, bb)[0, 1])

    cors = np.array([corr_at(d, l) for l in lags])
    best = int(np.nanargmax(np.abs(cors)))

    rng = np.random.default_rng(seed)
    blocks = [g for _, g in d.groupby(pd.Grouper(freq=f"{block_days}D")) if len(g) > 200]
    boot_lags = []
    if len(blocks) >= 5:
        for _ in range(n_boot):
            pick = rng.integers(0, len(blocks), len(blocks))
            sample = pd.concat([blocks[i] for i in pick])
            c = np.array([corr_at(sample, l) for l in lags])
            boot_lags.append(lags[int(np.nanargmax(np.abs(c)))])
    if boot_lags:
        lo, hi = np.percentile(boot_lags, [5, 95])
    else:
        lo = hi = lags[best]
    # Лаг считается идентифицированным, если бутстрап даёт УЗКИЙ пик
    # (разброс не более 60 мин) и сама связь заметна (|r| >= 0.05).
    # Пороги вынесены сюда намеренно: они определяют, каким управляющим
    # воздействиям система вообще доверяет.
    ci_width_min = (hi - lo) * STEP_MIN
    ident = bool(ci_width_min <= 60 and abs(cors[best]) >= 0.05)
    return DeadTime(str(mv.name), str(cv.name), lags[best] * STEP_MIN,
                    float(cors[best]), int(lo * STEP_MIN), int(hi * STEP_MIN),
                    ident, len(d))


def identify_matrix(tele: pd.DataFrame, mvs: list[str], cvs: list[str],
                    **kw) -> pd.DataFrame:
    rows = [identify(tele[m], tele[c], **kw).as_dict() for m in mvs for c in cvs
            if m in tele.columns and c in tele.columns]
    return pd.DataFrame(rows)
