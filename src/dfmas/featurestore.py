"""Витрина «на момент времени» (as-of feature store).

ЗАЧЕМ ЭТО ГЛАВНЫЙ МОДУЛЬ ПРОЕКТА
--------------------------------
ТЗ требует отсутствия временной утечки и хранения «возраста анализа».
Мы решаем это не дисциплиной разработчика, а конструкцией: каждое значение
качества несёт ``available_at`` — момент, когда оно физически могло стать
известным оператору (для ЛИМС это отбор пробы + 4 ч, подтверждено экспертом).

Единственный способ получить данные в системе — вызвать ``snapshot(t)``.
Он возвращает только то, что было доступно к моменту ``t``. Ни агенты, ни
модели не имеют доступа к полным таблицам, поэтому «заглянуть в будущее»
невозможно даже случайно. Backtest и онлайн-режим используют один и тот же
код — значит, поведение в бэктесте и на потоке совпадает по построению.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, load_config
from .quality.sentinels import FLAG_NAMES

# сколько назад смотрим по умолчанию, формируя состояние процесса
DEFAULT_WINDOW = pd.Timedelta(hours=6)


@dataclass(frozen=True)
class LabReading:
    """Один доступный результат анализа со всей метаинформацией."""
    stream: str
    param: str
    value: float
    unit: str
    source: str            # LIMS | PAK
    measured_at: pd.Timestamp
    available_at: pd.Timestamp
    age_hours: float       # возраст ОТНОСИТЕЛЬНО момента среза, по отбору пробы
    overridden: bool = False   # значение подменено сценарием

    def as_dict(self) -> dict:
        return {"stream": self.stream, "param": self.param, "value": float(self.value),
                "unit": self.unit, "source": self.source,
                "measured_at": self.measured_at.isoformat(),
                "available_at": self.available_at.isoformat(),
                "age_hours": round(float(self.age_hours), 2),
                "overridden": self.overridden}


@dataclass
class ProcessState:
    """Полное состояние процесса на момент ``t`` — вход мультиагентного цикла."""
    t: pd.Timestamp
    tags: dict[str, float]                 # последнее валидное значение тега
    tag_age_min: dict[str, float]          # сколько минут этому значению
    tag_flag: dict[str, str]               # код качества последнего сырого отсчёта
    window: pd.DataFrame                   # окно телеметрии (очищенное)
    # Ключ — (поток, показатель, ИСТОЧНИК). Источник обязателен: иначе частый
    # ПАК затирает редкий, но контрольный результат ЛИМС, и требование ТЗ
    # «лабораторный результат считается контрольным фактом» ломается молча.
    lab: dict[tuple[str, str, str], LabReading]
    missing_tags: list[str] = field(default_factory=list)
    # Кэш агрегатов по окну. Оптимизатор перебирает сотни вариантов, и каждый
    # заново спрашивает одни и те же средние — без кэша на это уходит ~80 %
    # времени цикла. Срез неизменяем, поэтому кэш безопасен.
    _agg_cache: dict = field(default_factory=dict, repr=False)

    # ---- удобные геттеры -------------------------------------------------
    def get(self, tag: str, default: float = float("nan")) -> float:
        v = self.tags.get(tag, default)
        return default if v is None or (isinstance(v, float) and np.isnan(v)) else v

    def lab_value(self, stream: str, param: str,
                  source: str | None = None) -> LabReading | None:
        """Анализ по потоку и показателю.

        ``source=None`` — самый СВЕЖИЙ по моменту отбора (обычно ПАК);
        ``source="LIMS"`` — именно лабораторный результат, контрольный факт.
        """
        if source is not None:
            return self.lab.get((stream, param, source))
        cands = [v for (st, pr, _src), v in self.lab.items()
                 if st == stream and pr == param]
        if not cands:
            return None
        return min(cands, key=lambda r: (r.age_hours, 0 if r.source == "LIMS" else 1))

    def lab_is_overridden(self, stream: str, param: str) -> bool:
        return any(v.overridden for (st, pr, _s), v in self.lab.items()
                   if st == stream and pr == param)

    def mean(self, tag: str, minutes: int = 60) -> float:
        """Среднее тега за последние ``minutes`` — сглаживание шума приборов."""
        key = ("mean", tag, minutes)
        hit = self._agg_cache.get(key)
        if hit is not None:
            return hit
        if tag not in self.window.columns:
            self._agg_cache[key] = float("nan")
            return float("nan")
        s = self.window[tag].loc[self.t - pd.Timedelta(minutes=minutes):self.t]
        val = float(s.mean()) if s.notna().any() else float("nan")
        self._agg_cache[key] = val
        return val

    def slope_per_hour(self, tag: str, minutes: int = 120) -> float:
        """Скорость изменения тега (ед./ч) — нужна агенту качества для тренда."""
        key = ("slope", tag, minutes)
        if key in self._agg_cache:
            return self._agg_cache[key]
        if tag not in self.window.columns:
            self._agg_cache[key] = float("nan")
            return float("nan")
        s = self.window[tag].loc[self.t - pd.Timedelta(minutes=minutes):self.t].dropna()
        if len(s) < 4:
            self._agg_cache[key] = float("nan")
            return float("nan")
        x = (s.index - s.index[0]).total_seconds().to_numpy() / 3600.0
        a = float(np.polyfit(x, s.to_numpy(), 1)[0])
        self._agg_cache[key] = a
        return a

    def summary(self) -> dict:
        return {"t": self.t.isoformat(),
                "n_tags": len(self.tags),
                "n_missing": len(self.missing_tags),
                "lab": {f"{k[0]}:{k[1]}@{k[2]}": v.as_dict()
                        for k, v in sorted(self.lab.items())}}


class AsOfStore:
    """Хранилище, отдающее данные строго «как их видно на момент t»."""

    def __init__(self, telemetry: pd.DataFrame, flags: pd.DataFrame, lab: pd.DataFrame):
        self.telemetry = telemetry.sort_index()
        self.flags = flags.reindex(self.telemetry.index)
        lab = lab.copy()
        lab["available_at"] = pd.to_datetime(lab["available_at"])
        lab["measured_at"] = pd.to_datetime(lab["measured_at"])
        self.lab = lab.sort_values("available_at").reset_index(drop=True)
        self._lab_by_key = {k: g.reset_index(drop=True) for k, g in
                            self.lab.groupby(["stream", "param", "source"], sort=False)}

    # -------------------------------------------------------------- loading
    @classmethod
    def load(cls, processed_dir: Path | None = None) -> "AsOfStore":
        d = Path(processed_dir or DATA_PROCESSED)
        missing = [f for f in ("telemetry.parquet", "telemetry_flags.parquet", "lab.parquet")
                   if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(
                "Витрина данных не собрана: не хватает "
                + ", ".join(missing)
                + ".\n\nЧто сделать:\n"
                "  1) положите data_1.rar в корень проекта\n"
                "     (телеметрия 24-2000 и АВТ не входит в архив решения — 340 МБ);\n"
                "  2) выполните:  make data && make fit\n\n"
                "Файлы ЛИМС, ПАК и справочника тегов уже лежат в data/raw/.")
        return cls(pd.read_parquet(d / "telemetry.parquet"),
                   pd.read_parquet(d / "telemetry_flags.parquet"),
                   pd.read_parquet(d / "lab.parquet"))

    # ------------------------------------------------------------- snapshot
    def snapshot(self, t: pd.Timestamp | str, window: pd.Timedelta = DEFAULT_WINDOW,
                 tags: list[str] | None = None,
                 overrides: dict[str, float] | None = None,
                 lab_overrides: dict[tuple[str, str], float] | None = None) -> ProcessState:
        """Состояние процесса на момент ``t``.

        ``overrides`` / ``lab_overrides`` позволяют сценариям подменить любое
        значение (например, поднять серу в сырье) — именно так реализованы
        «мягкие» тестовые сценарии из ТЗ: те же агенты, другие входные данные.
        """
        t = pd.Timestamp(t)
        cols = tags or list(self.telemetry.columns)
        win = self.telemetry.loc[t - window:t, cols]
        flg = self.flags.loc[t - window:t, cols] if not self.flags.empty else None

        values, ages, flag_names, missing = {}, {}, {}, []
        for c in cols:
            s = win[c].dropna()
            if s.empty:
                values[c] = float("nan"); ages[c] = float("inf"); missing.append(c)
            else:
                values[c] = float(s.iloc[-1])
                ages[c] = float((t - s.index[-1]).total_seconds() / 60.0)
            if flg is not None and not flg[c].empty:
                flag_names[c] = FLAG_NAMES.get(int(flg[c].iloc[-1]), "?")
        if overrides:
            win = win.copy()
            for k, v in overrides.items():
                values[k] = float(v)
                ages[k] = 0.0 if np.isfinite(v) else float("inf")
                if np.isfinite(v):
                    if k in missing:
                        missing.remove(k)
                elif k not in missing:
                    missing.append(k)
                # Подмена применяется ко ВСЕМУ окну: неисправный прибор врёт не
                # один отсчёт, а всё время, и сглаживание по часу должно это
                # видеть. Иначе сценарий «отказ датчика» не воспроизводится.
                if k in win.columns:
                    win[k] = float(v)
                else:
                    win[k] = float(v)

        lab_now: dict[tuple[str, str, str], LabReading] = {}
        for key, g in self._lab_by_key.items():
            pos = g["available_at"].searchsorted(t, side="right") - 1
            if pos < 0:
                continue
            row = g.iloc[pos]
            lab_now[key] = LabReading(
                stream=row["stream"], param=row["param"], value=float(row["value"]),
                unit=row["unit"], source=row["source"],
                measured_at=row["measured_at"], available_at=row["available_at"],
                age_hours=float((t - row["measured_at"]).total_seconds() / 3600.0))
        if lab_overrides:
            # Подмена задаёт ТЕКУЩЕЕ качество сырья — такое, каким его показал бы
            # виртуальный анализатор на стороне АВТ (без четырёхчасовой задержки
            # лаборатории). Поэтому возраст равен нулю, а эффект изменения ещё
            # не дошёл до продукта: именно это и проверяет сценарий
            # «утяжелилось сырьё». Если бы значение подставлялось с возрастом
            # 4 ч, его влияние уже содержалось бы в показании поточного
            # анализатора продукта, и добавлять его второй раз было бы ошибкой.
            for key, v in lab_overrides.items():
                stream, param = key[0], key[1]
                base = lab_now.get((stream, param, "LIMS")) or \
                    next((r for (st, pr, _s), r in lab_now.items()
                          if st == stream and pr == param), None)
                lab_now[(stream, param, "LIMS")] = LabReading(
                    stream=stream, param=param, value=float(v),
                    unit=base.unit if base else "?", source="СЦЕНАРИЙ (текущее сырьё)",
                    measured_at=t, available_at=t, age_hours=0.0, overridden=True)

        return ProcessState(t=t, tags=values, tag_age_min=ages, tag_flag=flag_names,
                            window=win, lab=lab_now, missing_tags=missing)

    # --------------------------------------------------------------- future
    def future_value(self, tag: str, t: pd.Timestamp, horizon_min: int) -> float:
        """ТОЛЬКО для офлайн-оценки качества прогноза. В рабочем контуре не используется.

        Вынесено отдельным методом с говорящим именем, чтобы любое обращение
        к будущему было видно при чтении кода и на ревью.
        """
        tt = pd.Timestamp(t) + pd.Timedelta(minutes=horizon_min)
        if tag not in self.telemetry.columns:
            return float("nan")
        s = self.telemetry[tag].loc[:tt].dropna()
        return float(s.iloc[-1]) if len(s) else float("nan")
