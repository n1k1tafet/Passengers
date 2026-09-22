"""Модель процесса: что произойдёт с качеством, если изменить режим.

Архитектура прогноза
--------------------
Прогноз каждого показателя = ТЕКУЩИЙ ФАКТ + ПРИРАЩЕНИЕ ОТ ДЕЙСТВИЯ.

Такое разделение принципиально. Абсолютный прогноз «с нуля» по историческим
данным объясняет малую долю дисперсии (см. reports/03_models.md), а вот
ПРИРАЩЕНИЕ от управляющего воздействия оценивается гораздо надёжнее и именно
оно нужно для сравнения вариантов. Поэтому:

  * уровень берётся из измерения (ПАК/поточный анализатор) с коррекцией по
    ЛИМС — то есть из факта, а не из модели;
  * приращение считается моделью отклика;
  * динамика учитывается явно: до истечения запаздывания эффекта нет,
    дальше он нарастает по апериодическому звену первого порядка.

Источники приращения
--------------------
  сера        — кинетика ГДС (hds.py), монотонная и обратимая;
  остальные   — аналитические производные формул ВАК (они линейны по тегам),
                то есть коэффициенты, подтверждённые экспертом.

Такой подход не изобретает зависимостей, которых нет в выданных материалах.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import va_formulas as va
from .conformal import ConformalCalibration
from .hds import HDSParams, dlnS_dT, required_temperature, sulfur_out

#: Постоянная времени контура «режим -> качество», мин. [ДОП]
#: Эксперт задал только диапазон запаздывания 0..3 ч; постоянная времени
#: принята равной половине верхней границы и вынесена в константу, чтобы
#: её можно было изменить одним местом.
TIME_CONSTANT_MIN = 60.0


def dynamic_fraction(horizon_min: float, dead_time_min: float,
                     tau_min: float = TIME_CONSTANT_MIN) -> float:
    """Доля установившегося эффекта, реализованная к моменту ``horizon_min``."""
    t = float(horizon_min) - float(dead_time_min)
    if t <= 0:
        return 0.0
    return float(1.0 - np.exp(-t / max(tau_min, 1e-6)))


@dataclass
class Prediction:
    param: str
    point: float
    lo: float
    hi: float
    horizon_min: int
    method: str
    components: dict = field(default_factory=dict)
    exceed_prob: float | None = None
    note: str = ""

    def as_dict(self) -> dict:
        f = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 3)
        return {"param": self.param, "point": f(self.point), "lo": f(self.lo),
                "hi": f(self.hi), "horizon_min": self.horizon_min,
                "method": self.method, "exceed_prob": f(self.exceed_prob),
                "components": {k: f(v) if isinstance(v, (int, float)) else v
                               for k, v in self.components.items()},
                "note": self.note}


@dataclass
class Action:
    """Управляющее воздействие в ФИЗИЧЕСКИХ единицах.

    ``d_temp_c``  — изменение температуры на входе в реактор Р-202, °C
    ``d_load_pct``— изменение загрузки установки по сырью, % от текущей
    ``d_press_mpa``— изменение давления на входе в реактор, МПа
    ``d_h2_pct``  — изменение расхода свежего ВСГ, % от текущего
    """
    d_temp_c: float = 0.0
    d_load_pct: float = 0.0
    d_press_mpa: float = 0.0
    d_h2_pct: float = 0.0

    def is_hold(self, eps: float = 1e-9) -> bool:
        return (abs(self.d_temp_c) < eps and abs(self.d_load_pct) < eps
                and abs(self.d_press_mpa) < eps and abs(self.d_h2_pct) < eps)

    def as_dict(self) -> dict:
        return {"d_temp_c": round(self.d_temp_c, 3), "d_load_pct": round(self.d_load_pct, 3),
                "d_press_mpa": round(self.d_press_mpa, 4), "d_h2_pct": round(self.d_h2_pct, 3)}

    def label(self) -> str:
        if self.is_hold():
            return "режим без изменений"
        parts = []
        if abs(self.d_temp_c) >= 0.05:
            parts.append(f"температура реактора {self.d_temp_c:+.1f} °C")
        if abs(self.d_load_pct) >= 0.05:
            parts.append(f"загрузка {self.d_load_pct:+.1f} %")
        if abs(self.d_press_mpa) >= 0.005:
            parts.append(f"давление {self.d_press_mpa:+.2f} МПа")
        if abs(self.d_h2_pct) >= 0.5:
            parts.append(f"расход ВСГ {self.d_h2_pct:+.1f} %")
        return ", ".join(parts) if parts else "режим без изменений"


class PlantModel:
    """Единая модель установки, используемая всеми агентами."""

    def __init__(self, hds: HDSParams, dead_times: dict[str, int],
                 calibrations: dict[str, ConformalCalibration],
                 va_bias: dict[str, float] | None = None,
                 mv_ranges: dict[str, tuple[float, float]] | None = None,
                 empirical_dlnS_dT: float | None = None):
        self.hds = hds
        self.dead_times = dead_times            # показатель -> запаздывание, мин
        self.calibrations = calibrations        # показатель -> конформная калибровка
        self.va_bias = va_bias or {}            # имя ВАК -> смещение
        self.mv_ranges = mv_ranges or {}
        self.empirical_dlnS_dT = empirical_dlnS_dT

    # ------------------------------------------------------------------ сера
    def predict_sulfur(self, s_now: float, s_feed_wt: float, lhsv: float,
                       p_mpa: float, t_reactor_c: float, h2_oil: float,
                       action: Action, horizon_min: int,
                       s_feed_prev_wt: float | None = None,
                       unrealized_feed_fraction: float = 0.0) -> Prediction:
        """Прогноз серы продукта через ``horizon_min`` при заданном действии.

        Учитываются ДВА источника изменения:
          * управляющее воздействие (что сделает оператор);
          * ещё не проявившееся изменение качества сырья — возмущение,
            которое придёт само. Без него сценарий «утяжелилось сырьё»
            не воспроизводится: текущий замер серы в продукте отражает
            сырьё, прошедшее реактор dead_time назад.
        """
        dt = self.dead_times.get("Mg.Sulfur", 80)
        frac = dynamic_fraction(horizon_min, dt)

        s_base = sulfur_out(s_feed_wt * 10_000.0, t_reactor_c, lhsv, p_mpa,
                            self.hds, h2_oil)
        lhsv_new = lhsv * (1.0 + action.d_load_pct / 100.0)
        h2_new = h2_oil * (1.0 + action.d_h2_pct / 100.0) / max(1e-9, 1.0 + action.d_load_pct / 100.0)
        s_act = sulfur_out(s_feed_wt * 10_000.0, t_reactor_c + action.d_temp_c,
                           lhsv_new, p_mpa + action.d_press_mpa, self.hds, h2_new)
        # приращение в логарифме — модель мультипликативна
        dln_model = float(np.log(max(s_act, 1e-6)) - np.log(max(s_base, 1e-6)))
        # эмпирическая проверка знака и масштаба по температуре
        if self.empirical_dlnS_dT is not None and abs(action.d_temp_c) > 1e-9:
            dln_emp = self.empirical_dlnS_dT * action.d_temp_c
            spread = abs(dln_model - dln_emp)
        else:
            spread = 0.0
        # возмущение по сырью
        dln_feed = 0.0
        if (s_feed_prev_wt is not None and unrealized_feed_fraction > 1e-3
                and abs(s_feed_prev_wt - s_feed_wt) > 1e-9):
            s_prev = sulfur_out(s_feed_prev_wt * 10_000.0, t_reactor_c, lhsv, p_mpa,
                                self.hds, h2_oil)
            dln_feed = float(np.log(max(s_base, 1e-6)) - np.log(max(s_prev, 1e-6)))
            dln_feed *= float(unrealized_feed_fraction) * frac

        point = float(max(s_now, 1e-6) * np.exp(dln_model * frac + dln_feed))

        cal = self.calibrations.get("Mg.Sulfur")
        if cal is not None and cal.n >= 20:
            extrap = self._extrapolation_factor(action)
            widen_f = extrap * (1.0 + 2.0 * spread)
            from .conformal import widen as _w
            c = _w(cal, widen_f)
            lo, hi = c.interval(point, 0.90)
            note = ""
            if widen_f > 1.05:
                note = (f"интервал расширен x{widen_f:.2f}: выход за наблюдавшийся "
                        f"диапазон и/или расхождение кинетики с идентификацией")
        else:
            lo = hi = float("nan"); note = "нет калибровки интервала"
        return Prediction(param="Mg.Sulfur", point=point, lo=lo, hi=hi,
                          horizon_min=horizon_min, method="кинетика ГДС + динамика",
                          components={"s_now": s_now, "dln_steady": dln_model,
                                      "dln_feed_disturbance": dln_feed,
                                      "realised_fraction": frac, "dead_time_min": dt,
                                      "s_steady": s_now * float(np.exp(dln_model))},
                          note=note)

    def risk_decomposition(self, point: float, limit: float) -> dict:
        """Раскладывает риск нарушения на технологическую и измерительную части.

        Возвращает:
          ``p_total``      — вероятность, что ЛАБОРАТОРНЫЙ результат превысит предел;
          ``p_process``    — вероятность, что истинный уровень процесса превысит предел;
          ``p_measurement``— вклад расхождения «лаборатория — поточный анализатор»;
          ``target``       — уровень серы, при котором ``p_total`` опускается до порога.
        """
        total = self.calibrations.get("Mg.Sulfur")
        meas = self.calibrations.get("Mg.Sulfur.measurement")
        out = {"p_total": float("nan"), "p_process": float("nan"),
               "sigma_total": float("nan"), "sigma_measurement": float("nan"),
               "sigma_process": float("nan"), "target_for_10pct": float("nan")}
        if total is None or total.n < 20:
            return out
        out["p_total"] = total.exceed_probability(point, limit, "upper")
        out["sigma_total"] = total.sigma()
        q90 = float(np.quantile(total.residuals, 0.90))
        out["target_for_10pct"] = float(limit - q90)
        if meas is not None and meas.n >= 20:
            sm = meas.sigma()
            out["sigma_measurement"] = sm
            out["sigma_process"] = float(np.sqrt(max(out["sigma_total"] ** 2 - sm ** 2, 0.0)))
            sp = out["sigma_process"]
            if sp > 1e-6:
                from math import erf, sqrt
                z = (limit - point) / sp
                out["p_process"] = float(1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
            else:
                out["p_process"] = 0.0
        return out

    def _extrapolation_factor(self, action: Action) -> float:
        """Во сколько раз расширить интервал при выходе за историю."""
        f = 1.0
        for key, val in (("temp", action.d_temp_c), ("load", action.d_load_pct),
                         ("press", action.d_press_mpa), ("h2", action.d_h2_pct)):
            rng = self.mv_ranges.get(key)
            if rng is None or abs(val) < 1e-12:
                continue
            allowed = max(abs(rng[0]), abs(rng[1]))
            if allowed > 0 and abs(val) > allowed:
                f *= 1.0 + (abs(val) / allowed - 1.0)
        return float(min(f, 4.0))

    # --------------------------------------------------- прочие показатели
    @staticmethod
    def va_sensitivity(state, va_name: str, tag: str, eps: float = 1.0) -> float:
        """Производная ВАК по тегу — численно, но по ТОЙ ЖЕ формуле, что в онлайне."""
        base = va.evaluate(state, [va_name])[va_name].value
        st2 = copy.copy(state)
        st2.tags = dict(state.tags)
        st2.window = state.window.copy()
        st2._agg_cache = {}
        if tag in st2.window.columns:
            st2.window[tag] = st2.window[tag] + eps
        st2.tags[tag] = st2.tags.get(tag, np.nan) + eps
        bumped = va.evaluate(st2, [va_name])[va_name].value
        if not (np.isfinite(base) and np.isfinite(bumped)):
            return float("nan")
        return float((bumped - base) / eps)
