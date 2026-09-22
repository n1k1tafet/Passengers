"""Агент качества: текущее и прогнозное качество, риск выхода за спецификацию.

Иерархия источников строго по ТЗ: ЛИМС -> ПАК -> ВАК.
  * если есть свежий лабораторный результат — он считается контрольным фактом
    и используется как уровень;
  * поточный анализатор даёт оперативный уровень, скорректированный по
    последним лабораторным результатам (models/bias.py);
  * виртуальный анализатор применяется там, где ни ЛИМС, ни ПАК недоступны,
    и тоже с коррекцией смещения.

Прогноз = текущий уровень + приращение от действия (models/process_model.py).
Риск нарушения = эмпирическая вероятность по конформной калибровке, а не
«сигма из нормального распределения».

Почему агент не пытается предсказать серу «с нуля»: на выданной истории
событие «лаборатория > 10 мг/кг» предсказывается поточным анализатором с
ROC-AUC 0.73, и добавление всех остальных 96 тегов не улучшает ни MAE, ни AUC
(reports/03_models.md). Расхождение «лаборатория — анализатор» превышает
наблюдаемую вариацию процесса, поэтому честный продукт — не точечный прогноз,
а риск с интервалом.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.lims import RU_NAMES
from ..models import va_formulas as va
from ..models.process_model import Action, dynamic_fraction
from .base import Agent
from .contracts import QualityItem, QualityReport

#: Какой ВАК отвечает за показатель продукта, если нет ЛИМС/ПАК
VA_FOR_PARAM = {
    "95%.T": "24-2000:GODT:T95",
    "90%.T": "24-2000:GODT:T90",
    "50%.T": "24-2000:GODT:T50",
    "CloudPoint": "24-2000:GODT:CloudPoint",
    "CFPP": "24-2000:GODT:CFPP",
    "D15": "24-2000:GODT:D15",
    "IBP.T": "24-2000:GODT:IBP",
    "I250": "24-2000:GODT:I250",
}

#: Чувствительность показателей к температуре реактора, ед./°C.
#: Значения получены как производные подтверждённых экспертом формул ВАК по
#: температурному тегу T6/T23 (они линейны, поэтому производная = коэффициент),
#: и дополнительно ограничены здравым смыслом. Помечены как допущение [ДОП].
DQDT = {
    "95%.T": 0.50,        # из формулы T95: 0.50 * T6
    "90%.T": 0.26366,     # из формулы T90: 0.26366 * T23
    "50%.T": 0.471,       # из формулы T50: 0.471 * T6
    "CloudPoint": 0.12018,
    "CFPP": 0.22088,
    "D15": -0.25,         # [ДОП] рост глубины гидрирования снижает плотность
    "CetaneNumber": 0.05, # [ДОП] глубже гидрирование -> выше ЦЧ
    "FlashPoint": -0.30,  # [ДОП]
}


class QualityAgent(Agent):
    name = "quality"
    role = "Агент качества — прогноз показателей, риск нарушения спецификации, уверенность"

    def __init__(self, ctx):
        super().__init__(ctx)
        # Кэш уровней показателей внутри одного среза: уровень не зависит от
        # варианта действия, а оптимизатор перебирает сотни вариантов.
        self._level_cache: dict[tuple, tuple] = {}

    def reset_cache(self) -> None:
        self._level_cache.clear()

    # ------------------------------------------------------------ интерфейс
    def on_assess(self, state, grade: str, feed, action: Action | None = None,
                  horizon_min: int | None = None) -> QualityReport:
        cfg = self.ctx.config
        horizon_min = int(horizon_min or cfg.horizon_min)
        action = action or Action()
        limits = cfg.grade(grade)["limits"]
        items: list[QualityItem] = []
        notes: list[str] = []

        s_item = self._sulfur_item(state, feed, action, horizon_min, limits.get("Mg.Sulfur"))
        items.append(s_item)

        for param, lim in limits.items():
            if param == "Mg.Sulfur":
                continue
            items.append(self._generic_item(state, param, lim, action, horizon_min))

        hard_items = [i for i in items if i.hard and i.exceed_prob is not None]
        worst = max(hard_items, key=lambda i: i.exceed_prob) if hard_items else None
        conf = self._confidence(state, feed, items)
        for i in items:
            if i.source.startswith("ВАК"):
                notes.append(f"{i.ru_name}: нет свежего анализа, используется "
                             f"виртуальный анализатор с коррекцией смещения")
        return QualityReport(t=state.t, grade=grade, items=items, confidence=conf,
                             horizon_min=horizon_min,
                             worst_param=worst.param if worst else None,
                             worst_prob=float(worst.exceed_prob) if worst else 0.0,
                             notes=notes)

    # --------------------------------------------------------------- сера
    def _sulfur_item(self, state, feed, action: Action, horizon_min: int, lim) -> QualityItem:
        plant, bundle = self.ctx.plant, self.ctx.bundle
        ref = bundle.reference
        key = (id(state), state.t, "__sulfur__")
        if key in self._level_cache:
            s_now, src, age = self._level_cache[key]
        else:
            s_now, src, age = self.current_sulfur(state)
            self._level_cache[key] = (s_now, src, age)
        t_react = state.mean(bundle.meta["tags"]["temp"], 60)
        load = state.mean(bundle.meta["tags"]["load"], 60)
        press = state.mean(bundle.meta["tags"]["press"], 60) * ref["scale_press"]
        h2 = state.mean(bundle.meta["tags"]["h2"], 60)
        if not np.isfinite(t_react):
            t_react = ref["t_reactor_c"]
        if not np.isfinite(load) or load <= 0:
            load = ref["load"]
        if not np.isfinite(press) or press <= 0:
            press = ref["p_mpa"]
        lhsv = ref["lhsv"] * load / max(ref["load"], 1e-9)
        h2_oil = 250.0 * (h2 / max(ref["h2"], 1e-9)) * (ref["load"] / max(load, 1e-9)) \
            if np.isfinite(h2) else 250.0

        pred = plant.predict_sulfur(s_now=s_now, s_feed_wt=feed.sulfur_wt, lhsv=lhsv,
                                    p_mpa=press, t_reactor_c=t_react, h2_oil=h2_oil,
                                    action=action, horizon_min=horizon_min,
                                    s_feed_prev_wt=getattr(feed, "sulfur_prev_wt", None),
                                    unrealized_feed_fraction=getattr(
                                        feed, "unrealized_fraction", 0.0))
        cal = plant.calibrations.get("Mg.Sulfur")
        limit = float(lim["max"]) if lim else None
        prob = cal.exceed_probability(pred.point, limit, "upper") if (cal and limit) else None
        detail = plant.risk_decomposition(pred.point, limit) if limit else None
        return QualityItem(param="Mg.Sulfur", ru_name=RU_NAMES["Mg.Sulfur"], value=s_now,
                           unit="мг/кг", source=src, age_h=age, forecast=pred.point,
                           lo=pred.lo, hi=pred.hi, limit=limit, limit_kind="max",
                           hard=bool(lim.get("hard", True)) if lim else True,
                           exceed_prob=prob,
                           risk_process=detail.get("p_process") if detail else None,
                           risk_detail=detail,
                           margin=(limit - pred.point) if limit is not None else None)

    def current_sulfur(self, state) -> tuple[float, str, float]:
        """Слитая оценка серы продукта: ЛИМС -> ПАК/поточный -> ВАК."""
        bundle = self.ctx.bundle
        lims = state.lab_value("HT_PRODUCT", "Mg.Sulfur", source="LIMS")
        online_tag = state.mean(bundle.meta["tags"]["sulfur"], 60)
        pak = state.lab_value("HT_PRODUCT", "Mg.Sulfur", source="PAK")
        pak_v = pak.value if pak is not None else np.nan
        vals = [v for v in (online_tag, pak_v) if np.isfinite(v) and 0.2 < v < 60]
        bias = self.ctx.extras.get("sulfur_bias", 0.0)
        # Свежий лабораторный результат — контрольный факт (правило ТЗ).
        if lims is not None and lims.age_hours <= 4.5:
            return float(lims.value), "ЛИМС (контрольный факт)", float(lims.age_hours)
        if vals:
            v = float(np.mean(vals) + bias)
            src = "ПАК + поточный анализатор, коррекция по ЛИМС" if len(vals) > 1 \
                else "поточный анализатор, коррекция по ЛИМС"
            return v, src, float(state.tag_age_min.get(bundle.meta["tags"]["sulfur"], 0) / 60.0)
        if lims is not None:
            return float(lims.value), "ЛИМС (устаревший)", float(lims.age_hours)
        return float("nan"), "нет источника", float("inf")

    # ------------------------------------------------- остальные показатели
    def _generic_item(self, state, param: str, lim: dict, action: Action,
                      horizon_min: int) -> QualityItem:
        value, src, age = self._level(state, param)
        dt = self.ctx.bundle.dead_times.get("temp", 80)
        frac = dynamic_fraction(horizon_min, dt)
        d = DQDT.get(param, 0.0) * action.d_temp_c * frac
        forecast = value + d if np.isfinite(value) else float("nan")

        kind = "max" if "max" in lim else "min"
        limit = float(lim.get("max", lim.get("min")))
        sigma = self.ctx.extras.get("param_sigma", {}).get(param, np.nan)
        lo = hi = float("nan"); prob = None
        if np.isfinite(forecast) and np.isfinite(sigma) and sigma > 0:
            lo, hi = forecast - 1.645 * sigma, forecast + 1.645 * sigma
            z = (limit - forecast) / sigma
            prob = float(1.0 - _phi(z)) if kind == "max" else float(_phi(z))
        margin = (limit - forecast) if kind == "max" else (forecast - limit)
        # для диапазона (например, плотность 820..845) учитываем обе границы
        if "min" in lim and "max" in lim and np.isfinite(forecast):
            margin = min(float(lim["max"]) - forecast, forecast - float(lim["min"]))
            if np.isfinite(sigma) and sigma > 0:
                prob = float((1.0 - _phi((float(lim["max"]) - forecast) / sigma))
                             + _phi((float(lim["min"]) - forecast) / sigma))
            limit = float(lim["max"]); kind = "range"
        return QualityItem(param=param, ru_name=RU_NAMES.get(param, param), value=value,
                           unit=lim.get("unit", ""), source=src, age_h=age,
                           forecast=forecast, lo=lo, hi=hi, limit=limit, limit_kind=kind,
                           hard=bool(lim.get("hard", True)), exceed_prob=prob, margin=margin)

    def _level(self, state, param: str) -> tuple[float, str, float]:
        key = (id(state), state.t, param)
        if key in self._level_cache:
            return self._level_cache[key]
        out = self._level_uncached(state, param)
        self._level_cache[key] = out
        return out

    def _level_uncached(self, state, param: str) -> tuple[float, str, float]:
        stale = float(self.ctx.config.specs["staleness_hours"]["LIMS"])
        lr = state.lab_value("HT_PRODUCT", param, source="LIMS") \
            or state.lab_value("HT_PRODUCT", param)
        if lr is not None and lr.age_hours <= stale:
            return float(lr.value), f"ЛИМС ({lr.age_hours:.0f} ч)", float(lr.age_hours)
        name = VA_FOR_PARAM.get(param)
        if name:
            res = va.evaluate(state, [name])[name]
            if res.valid and np.isfinite(res.value):
                return (float(res.value + self.ctx.bundle.va_bias.get(name, 0.0)),
                        f"ВАК {name} + коррекция", 0.0)
        if lr is not None:
            return float(lr.value), f"ЛИМС (устаревший, {lr.age_hours:.0f} ч)", float(lr.age_hours)
        return float("nan"), "нет источника", float("inf")

    # -------------------------------------------------------- уверенность
    def _confidence(self, state, feed, items) -> float:
        """Уверенность как СРЕДНЕЕ ГЕОМЕТРИЧЕСКОЕ частных оценок.

        Произведение независимых штрафов занижает результат тем сильнее, чем
        больше показателей в спецификации, — это артефакт, а не свойство
        данных. Среднее геометрическое даёт величину, сравнимую между
        режимами и марками топлива.
        """
        parts = []
        parts.append(0.45 + 0.55 * float(feed.sulfur_confidence))     # свежесть сырья
        n_va = sum(1 for i in items if i.source.startswith("ВАК"))
        parts.append(float(np.clip(1.0 - 0.07 * n_va, 0.4, 1.0)))     # доля виртуальных
        n_missing = sum(1 for i in items if not np.isfinite(i.value))
        parts.append(float(np.clip(1.0 - 0.35 * n_missing, 0.1, 1.0)))
        s = next((i for i in items if i.param == "Mg.Sulfur"), None)
        if s is not None and s.lo is not None and np.isfinite(s.lo) and np.isfinite(s.hi):
            width = float(s.hi - s.lo)
            ref_width = 2.0 * 1.645 * 2.0          # ориентир: sigma = 2 мг/кг
            parts.append(float(np.clip(ref_width / max(width, 1e-6), 0.25, 1.0)))
        parts = [max(p, 1e-6) for p in parts]
        return float(np.clip(np.exp(np.mean(np.log(parts))), 0.0, 1.0))


def _phi(z: float) -> float:
    """Функция стандартного нормального распределения без scipy."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(float(z) / sqrt(2.0)))
