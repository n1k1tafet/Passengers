"""Виртуальные анализаторы качества (ВАК) из листа «ВАК» справочника.

Формулы приведены ровно в том виде, в каком их ПОДТВЕРДИЛ ЭКСПЕРТ после
вопроса команд. Исправления эксперта:

  * ``T90``: коэффициент при F15 применяется к ``F15/2000``
    (в исходном файле деление потеряно, из-за чего прямая подстановка
    давала T90 ≈ 166 286 вместо ~340 °C);
  * ``T50``: коэффициент при T6 равен 0.471 (в файле было 0.8052);
  * ``CloudPoint``: первое слагаемое ``0.0002*F22`` (в файле множитель потерян);
  * ``CFPP``: первое слагаемое по ``T23``, а не по ``T6``;
  * ``T95``: коэффициент при T6 равен 0.50 (в файле 0.62259);
  * ``AVT6:240-350:CFPP``: лишняя скобка, верно ``F65/F32 + F30``.

Ссылки вида ``LIMS:24-2000.Pipeline.*`` трактуются как лабораторные
показатели трубопровода СЫРЬЯ установки 24-2000, то есть точки отбора
«Гидроочистка, точка 1» (поток ``HT_FEED``). Значения берутся через
``ProcessState`` — то есть только те, что уже опубликованы (см. featurestore).

Каждый ВАК возвращает ``VAResult`` с признаком применимости: если хотя бы
один вход непригоден (сентинел, пропуск, устаревший анализ), результат
помечается ``valid=False`` и НЕ используется для принятия решения.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

H = "H_"   # префикс тегов установки 24-2000
A = "A_"   # префикс тегов АВТ


@dataclass
class VAResult:
    name: str
    stream: str
    param: str
    value: float
    valid: bool
    inputs: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "stream": self.stream, "param": self.param,
                "value": None if not np.isfinite(self.value) else round(float(self.value), 3),
                "valid": self.valid, "missing": self.missing, "note": self.note}


class _Resolver:
    """Достаёт входы формулы из состояния процесса и запоминает пропуски."""

    def __init__(self, state, prefix: str, smooth_min: int = 60,
                 max_lab_age_h: float = 72.0):
        self.state, self.prefix = state, prefix
        self.smooth_min, self.max_lab_age_h = smooth_min, max_lab_age_h
        self.used: dict[str, float] = {}
        self.missing: list[str] = []

    def tag(self, name: str) -> float:
        full = self.prefix + name
        v = self.state.mean(full, self.smooth_min)
        if not np.isfinite(v):
            v = self.state.get(full)
        if not np.isfinite(v):
            self.missing.append(full)
            return float("nan")
        self.used[full] = float(v)
        return float(v)

    def lab(self, stream: str, param: str) -> float:
        r = self.state.lab_value(stream, param)
        if r is None:
            self.missing.append(f"LIMS:{stream}:{param}")
            return float("nan")
        if r.age_hours > self.max_lab_age_h:
            self.missing.append(f"LIMS:{stream}:{param}(устарел {r.age_hours:.0f} ч)")
            return float("nan")
        self.used[f"LIMS:{stream}:{param}"] = float(r.value)
        return float(r.value)


# ---------------------------------------------------------------------------
#  24-2000 — гидроочищенное дизельное топливо (поток HT_PRODUCT)
# ---------------------------------------------------------------------------

def _ht_T90(r: _Resolver) -> float:
    T12, F15, W7, T23 = r.tag("T12"), r.tag("F15"), r.tag("W7"), r.tag("T23")
    F1, F26 = r.tag("F1"), r.tag("F26")
    return (162.998 + 0.12945 * T12 + 59.57 * (F15 / 2000.0) + 0.00036 * W7
            + 0.26366 * T23 - 424.72638 * F1 / F26)


def _ht_T50(r: _Resolver) -> float:
    return 44.625 + 10.0224 * r.tag("P13") + 0.06981 * r.tag("F9") + 0.471 * r.tag("T6")


def _ht_I250(r: _Resolver) -> float:
    return (84.585 - 0.21172 * r.tag("T5") + 0.12137 * r.tag("T11")
            - 0.00014 * r.tag("F25") + 0.56248 * r.tag("F14")
            - 0.16317 * r.tag("T23") + 0.20272 * r.tag("T16"))


def _ht_D15(r: _Resolver) -> float:
    return (667.881 + 0.15417 * r.lab("HT_FEED", "D15") + 0.00005 * r.tag("F22")
            + 0.10774 * r.tag("T11"))


def _ht_cloud(r: _Resolver) -> float:
    return (0.0002 * r.tag("F22") + 0.0021 * r.tag("W7") + 0.00008 * r.tag("F25")
            - 0.30656 * r.tag("F1") + 0.12018 * r.tag("T6") + 0.01916 * r.tag("F9")
            - 48.254 - 0.05249 * r.tag("T16") + 0.00011)


def _ht_T95(r: _Resolver) -> float:
    return (0.03814 * r.tag("F9") - 9.201 - 0.00002 * r.tag("F2")
            + 0.50 * r.tag("T6") + 0.48321 * r.lab("HT_FEED", "95%.T"))


def _ht_CFPP(r: _Resolver) -> float:
    return (0.22088 * r.tag("T23") - 102.375 - 47.75834 * r.tag("P8")
            + 0.03862 * r.tag("F9") + 43.60207 * r.tag("W7") + 43.81849 * r.tag("P24"))


def _ht_IBP(r: _Resolver) -> float:
    return (137.762 - 0.0653 * r.tag("F26") + 0.00011 * r.tag("F22")
            + 5.78137 * r.tag("P13") - 34.58028 * r.tag("P24") - 0.00993 * r.tag("F14")
            - 0.99962 * r.tag("W4") + 0.32232 * r.tag("T23") - 0.09406 * r.tag("T16"))


# ---------------------------------------------------------------------------
#  ЭЛОУ-АВТ-6 — дизельная фракция 240-350 °C (сырьё гидроочистки)
# ---------------------------------------------------------------------------

def _avt_D15(r: _Resolver) -> float:
    F30, F32, T66, T33 = r.tag("F30"), r.tag("F32"), r.tag("T66"), r.tag("T33")
    denom = F32 + F30
    return 791.22872 - 5.30294 * (F30 / denom) + 0.52755 * T66 - 0.15629 * T33


def _avt_T50(r: _Resolver) -> float:
    return (283.177 - 0.01685 * r.tag("F7") + 0.06248 * r.tag("F30")
            + 0.22048 * r.tag("F34") - 0.25816 * r.tag("F45")
            - 0.12159 * r.tag("F59") + 0.01221 * r.tag("F63"))


def _avt_EBP(r: _Resolver) -> float:
    return (813.883 + 2.66463 * r.tag("F30") - 0.20239 * r.tag("T33")
            - 3.65888 * r.tag("F36") - 14.08235 * r.tag("T37")
            - 1.32603 * r.tag("T40") + 14.60206 * r.tag("T58"))


def _avt_CFPP(r: _Resolver) -> float:
    # Скобки исправлены экспертом: F65/F32 + F30
    return (31.40363 - 0.06784 * r.tag("T33") + 17.411 * r.tag("P67")
            - 8.11544 * r.tag("P4") - 0.47309 * (r.tag("F65") / r.tag("F32") + r.tag("F30")))


SPECS: dict[str, tuple[str, str, str, Callable[[_Resolver], float]]] = {
    # name                 (prefix, stream,       param,         fn)
    "24-2000:GODT:T90":     (H, "HT_PRODUCT", "90%.T",      _ht_T90),
    "24-2000:GODT:T50":     (H, "HT_PRODUCT", "50%.T",      _ht_T50),
    "24-2000:GODT:I250":    (H, "HT_PRODUCT", "I250",       _ht_I250),
    "24-2000:GODT:D15":     (H, "HT_PRODUCT", "D15",        _ht_D15),
    "24-2000:GODT:CloudPoint": (H, "HT_PRODUCT", "CloudPoint", _ht_cloud),
    "24-2000:GODT:T95":     (H, "HT_PRODUCT", "95%.T",      _ht_T95),
    "24-2000:GODT:CFPP":    (H, "HT_PRODUCT", "CFPP",       _ht_CFPP),
    "24-2000:GODT:IBP":     (H, "HT_PRODUCT", "IBP.T",      _ht_IBP),
    "AVT6:240-350:D15":     (A, "HT_FEED",    "D15",        _avt_D15),
    "AVT6:240-350:T50":     (A, "HT_FEED",    "50%.T",      _avt_T50),
    "AVT6:240-350:EBP":     (A, "HT_FEED",    "EBP.T",      _avt_EBP),
    "AVT6:240-350:CFPP":    (A, "HT_FEED",    "CFPP",       _avt_CFPP),
}


def evaluate(state, names: list[str] | None = None, smooth_min: int = 60) -> dict[str, VAResult]:
    """Считает все (или указанные) ВАК на состоянии ``state``."""
    out: dict[str, VAResult] = {}
    for name in (names or list(SPECS)):
        prefix, stream, param, fn = SPECS[name]
        r = _Resolver(state, prefix, smooth_min=smooth_min)
        try:
            val = float(fn(r))
        except Exception as exc:              # деление на ноль и т. п.
            val, r.missing = float("nan"), r.missing + [f"ошибка расчёта: {exc}"]
        ok = bool(np.isfinite(val)) and not r.missing
        out[name] = VAResult(name=name, stream=stream, param=param, value=val,
                             valid=ok, inputs=r.used, missing=r.missing,
                             note="" if ok else "не все входы доступны")
    return out


# ---------------------------------------------------------------------------
#  Пакетный расчёт ВАК на всей истории
# ---------------------------------------------------------------------------
class _FrameResolver:
    """Тот же интерфейс, что и ``_Resolver``, но возвращает Series.

    Формулы выше — чистая арифметика, поэтому один и тот же код считает и
    одну точку, и всю историю. Это гарантирует, что офлайн-валидация
    проверяет РОВНО ту формулу, которая работает в онлайне.
    """

    def __init__(self, tele: pd.DataFrame, lab_asof: dict[tuple[str, str], pd.Series],
                 prefix: str):
        self.tele, self.lab_asof, self.prefix = tele, lab_asof, prefix
        self.used, self.missing = {}, []

    def tag(self, name: str) -> pd.Series:
        col = self.prefix + name
        if col not in self.tele.columns:
            self.missing.append(col)
            return pd.Series(np.nan, index=self.tele.index)
        return self.tele[col]

    def lab(self, stream: str, param: str) -> pd.Series:
        s = self.lab_asof.get((stream, param))
        if s is None:
            self.missing.append(f"LIMS:{stream}:{param}")
            return pd.Series(np.nan, index=self.tele.index)
        return s


def build_lab_asof(lab: pd.DataFrame, index: pd.DatetimeIndex,
                   max_age_h: float | None = None) -> dict[tuple[str, str], pd.Series]:
    """Для каждой пары (поток, показатель) — ряд «последнее ДОСТУПНОЕ значение».

    Используется ``available_at`` (отбор + задержка публикации), поэтому
    сдвиг вперёд гарантирован и утечки нет.
    """
    out = {}
    for key, g in lab.groupby(["stream", "param"], sort=False):
        g = g.sort_values("available_at")
        s = pd.Series(g["value"].to_numpy(), index=pd.DatetimeIndex(g["available_at"]))
        s = s[~s.index.duplicated(keep="last")]
        joined = s.reindex(s.index.union(index)).ffill().reindex(index)
        if max_age_h is not None:
            age = (pd.Series(index, index=index) -
                   pd.Series(s.index, index=s.index).reindex(
                       s.index.union(index)).ffill().reindex(index))
            joined = joined.where(age <= pd.Timedelta(hours=max_age_h))
        out[key] = joined
    return out


def evaluate_history(tele: pd.DataFrame, lab: pd.DataFrame,
                     names: list[str] | None = None,
                     smooth_min: int = 60) -> pd.DataFrame:
    """Считает ВАК на всей истории (для валидации и обучения гибридных моделей)."""
    sm = max(1, smooth_min // 10)
    tele_s = tele.rolling(sm, min_periods=max(1, sm // 2)).mean()
    lab_asof = build_lab_asof(lab, tele.index)
    cols = {}
    for name in (names or list(SPECS)):
        prefix, stream, param, fn = SPECS[name]
        r = _FrameResolver(tele_s, lab_asof, prefix)
        with np.errstate(all="ignore"):
            cols[name] = fn(r)
    return pd.DataFrame(cols, index=tele.index)
