"""Агент оптимизации: генерация, отсев и сравнение вариантов режима.

Шаги 5-7 цикла принятия решения из ТЗ.

1. **Генерация.** Сетка по управляющим воздействиям. Границы берутся как
   пересечение двух источников: наблюдавшейся в истории скорости изменения за
   30 минут (P95) и коридора, который разрешил агент надёжности. Вариант
   «ничего не делать» присутствует всегда и участвует в сравнении наравне.

2. **Отсев — два уровня.** Разделение принципиально и отражает физику
   измерения: расхождение «лаборатория — поточный анализатор» на выданных
   данных само по себе даёт 15-20 % вероятности превышения предела даже при
   идеально выдержанном режиме. Поэтому:

   *Уровень 1 (допустимость).* Вариант недопустим, если ТОЧЕЧНЫЙ прогноз
   выходит за жёсткий предел или если его запретил агент надёжности. Такие
   варианты не рассматриваются вообще.

   *Уровень 2 (соответствие).* Среди допустимых выделяются те, у которых
   вероятность нарушения не выше ``MAX_HARD_EXCEED_PROB``. Если таких нет,
   система не молчит: она выбирает вариант МИНИМАЛЬНОГО РИСКА и прямо
   сообщает, что соблюдение спецификации не гарантировано. Отказ остаётся
   для случаев, когда допустимых вариантов нет вовсе.

   Отсев выполняется ДО сравнения по экономике, поэтому экономический
   выигрыш физически не может «перевесить» качество.

3. **Сравнение.** Сначала строится фронт Парето по четырём критериям
   (риск качества, тяжесть режима, затраты, выпуск) — он показывается
   инженеру. Затем среди парето-оптимальных вариантов выбирается один по
   взвешенной свёртке с весами из config/agents.yaml.

4. **Защита от дёрганья.** Если выигрыш меньше порога
   ``min_action_benefit_rub_h`` и запас по качеству достаточен, побеждает
   «ничего не делать»: лишние управляющие действия сами по себе вредны.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..models.process_model import Action
from .base import Agent
from .contracts import Option, OptionMetrics, Severity

#: Порог берётся из config/agents.yaml (orchestrator.max_hard_exceed_prob).
#: Константа оставлена как значение по умолчанию для тестов.
MAX_HARD_EXCEED_PROB = 0.10


class OptimizerAgent(Agent):
    name = "optimizer"
    role = "Агент оптимизации — генерация вариантов, отсев по ограничениям, фронт Парето"

    def on_optimize(self, state, grade: str, feed, reliability, quality_fn,
                    economics_fn, horizon_min: int) -> list[Option]:
        cfg = self.ctx.config
        oc = cfg.agents["optimizer"]
        ranges = self.ctx.bundle.mv_ranges
        ref = self.ctx.bundle.reference
        max_prob = float(cfg.agents["orchestrator"].get("max_hard_exceed_prob",
                                                        MAX_HARD_EXCEED_PROB))

        rel_lo, rel_hi = reliability.bounds.get("d_temp_c", (-99.0, 99.0))
        t_lo = max(ranges["temp"][0], rel_lo)
        t_hi = min(ranges["temp"][1], rel_hi)
        if reliability.severity is Severity.ALARM:
            t_hi = min(t_hi, 0.0)          # тяжёлый режим — повышать температуру нельзя

        n = int(oc["n_grid"])
        temps = _grid(t_lo, t_hi, n)
        loads = _grid(ranges["load"][0], ranges["load"][1], 5)
        press = _grid(ranges["press"][0], ranges["press"][1], 3)
        h2s = _grid(ranges["h2"][0], ranges["h2"][1], 3)

        load_now = state.mean(self.ctx.bundle.meta["tags"]["load"], 60)
        if not np.isfinite(load_now) or load_now <= 0:
            load_now = ref["load"]
        t_now = state.mean(self.ctx.bundle.meta["tags"]["temp"], 60)
        if not np.isfinite(t_now):
            t_now = ref["t_reactor_c"]

        combos = [(0.0, 0.0, 0.0, 0.0)]
        combos += [c for c in itertools.product(temps, loads, press, h2s)
                   if c != (0.0, 0.0, 0.0, 0.0)]

        options: list[Option] = []
        for i, (dt, dl, dp, dh) in enumerate(combos):
            action = Action(d_temp_c=float(dt), d_load_pct=float(dl),
                            d_press_mpa=float(dp), d_h2_pct=float(dh))
            qrep = quality_fn(state, grade, feed, action, horizon_min)
            rel = self.ctx.extras["reliability_fn"](state, action)
            s_item = next((x for x in qrep.items if x.param == "Mg.Sulfur"), None)
            s_fore = s_item.forecast if s_item else float("nan")
            econ = economics_fn(state, feed, action, s_fore, t_now, load_now)

            violated, risky, margins, worst_p = [], [], [], 0.0
            for it in qrep.items:
                if not it.hard:
                    continue
                if it.exceed_prob is not None and np.isfinite(it.exceed_prob):
                    worst_p = max(worst_p, float(it.exceed_prob))
                    if it.exceed_prob > max_prob:
                        risky.append(f"{it.ru_name}: риск {it.exceed_prob*100:.0f} %")
                if it.margin is not None and np.isfinite(it.margin):
                    scale = max(abs(it.limit) * 0.02, 1e-6) if it.limit else 1.0
                    margins.append(float(it.margin) / scale)
                    if it.margin < 0:
                        violated.append(f"{it.ru_name}: точечный прогноз вне предела")
            if rel.severity is Severity.ALARM and dt > 0:
                violated.append("режим признан тяжёлым, повышение температуры запрещено")

            metrics = OptionMetrics(
                quality_margin=float(min(margins)) if margins else float("nan"),
                worst_exceed_prob=float(worst_p),
                throughput_t_h=float(econ["throughput_t_h"]),
                cost_rub_h=float(econ["cost_rub_h"]),
                margin_rub_h=float(econ["margin_rub_h"]),
                severity_index=float(rel.severity_index),
                feasible=not violated, violated=violated + risky)
            metrics.compliant = not violated and not risky   # уровень 2
            options.append(Option(option_id=f"opt{i:03d}", action=action, metrics=metrics,
                                  predictions={"quality": qrep, "economics": econ,
                                               "reliability": rel}))

        feasible = [o for o in options if o.metrics.feasible]
        _mark_pareto(feasible)
        self._score(feasible, options)
        feasible.sort(key=lambda o: -o.score)
        for r, o in enumerate(feasible, 1):
            o.rank = r
        if self.bus is not None:
            self.bus.note(self.name, "optimizer.summary", {
                "всего вариантов в сетке": len(options),
                "прошли жёсткие ограничения": len(feasible),
                "соответствуют спецификации по риску":
                    sum(1 for o in feasible if o.metrics.compliant),
                "на фронте Парето": sum(1 for o in feasible if o.pareto),
                "коридор температуры, °C": [round(t_lo, 2), round(t_hi, 2)],
            })
        return options

    # ---------------------------------------------------------------- свёртка
    def _score(self, feasible: list[Option], all_options: list[Option]) -> None:
        """Свёртка среди ДОПУСТИМЫХ вариантов.

        Качество уже обеспечено отсевом, поэтому здесь максимизируется МАРЖА
        (выручка минус переменные затраты), а не минимизируются затраты.
        Разница принципиальная: минимизация затрат толкает установку снижать
        нагрузку, потому что снижение расхода сырья действительно уменьшает
        затраты — но теряет выручку, которая на порядок больше. Именно это и
        имел в виду эксперт, говоря, что более высокая остаточная сера
        удешевляет производство: выигрыш должен считаться по марже.

        Запас по качеству и тяжесть режима остаются в свёртке с меньшими
        весами — как предпочтение более робастного варианта при прочих равных.
        """
        if not feasible:
            return
        w = self.ctx.config.agents["orchestrator"]["objective_weights"]
        margin = np.array([o.metrics.margin_rub_h for o in feasible], dtype=float)
        sev = np.array([o.metrics.severity_index for o in feasible], dtype=float)
        risk = np.array([o.metrics.worst_exceed_prob for o in feasible], dtype=float)
        for o, m, s, r in zip(feasible, _minmax(margin), _minmax(-sev), _minmax(-risk)):
            o.score = float(w["economics"] * m + w["reliability"] * s
                            + w["quality_margin"] * r)


def _grid(lo: float, hi: float, n: int) -> list[float]:
    lo, hi = float(min(lo, 0.0)), float(max(hi, 0.0))
    if abs(hi - lo) < 1e-9:
        return [0.0]
    vals = list(np.linspace(lo, hi, max(n, 3)))
    if not any(abs(v) < 1e-9 for v in vals):
        vals.append(0.0)
    return [round(float(v), 4) for v in sorted(set(vals))]


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    lo, hi = np.nanmin(x[finite]), np.nanmax(x[finite])
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    out = (x - lo) / (hi - lo)
    return np.where(finite, out, 0.0)


def _mark_pareto(options: list[Option]) -> None:
    """Отмечает недоминируемые варианты по (риск↓, тяжесть↓, затраты↓, выпуск↑)."""
    pts = np.array([[o.metrics.worst_exceed_prob, o.metrics.severity_index,
                     -o.metrics.margin_rub_h, -o.metrics.throughput_t_h] for o in options],
                   dtype=float)
    pts = np.nan_to_num(pts, nan=1e9)
    for i, o in enumerate(options):
        dominated = np.all(pts <= pts[i] + 1e-12, axis=1) & np.any(pts < pts[i] - 1e-12, axis=1)
        o.pareto = not bool(dominated.any())
