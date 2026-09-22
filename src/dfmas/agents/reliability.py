"""Агент надёжности: тяжесть режима и риск для оборудования и катализатора.

Прямой разметки отказов в выданных данных нет, поэтому, как разрешает ТЗ,
используются ПРОКСИ-ПРИЗНАКИ с явно описанными допущениями. Каждый фактор
нормирован в 0..1 относительно наблюдавшегося диапазона НОРМАЛЬНОЙ работы,
итоговый индекс — взвешенная сумма (веса в config/agents.yaml).

Факторы
-------
``wabt``       Тяжесть температурного режима. Чем выше температура слоя
               относительно опорной, тем быстрее дезактивируется катализатор.
               Правило Аррениуса: +12 °C сокращают ресурс цикла вдвое [ДОП].
``dp_reactor`` Перепад давления на реакторе Р-202 (тег W10). Рост перепада —
               классический признак закоксовывания и загрязнения слоя.
``quench``     Отношение расхода квенча к расходу сырья. Падение квенча при
               высокой температуре означает потерю управления экзотермой.
``h2_to_oil``  Кратность циркуляции водорода. Её снижение ускоряет
               коксообразование — общепринятый признак тяжёлого режима.

Агент возвращает не только индекс, но и ГРАНИЦЫ для оптимизатора: если режим
уже тяжёлый, коридор допустимого повышения температуры сужается. Так
надёжность попадает в оптимизацию как ограничение, а не как пожелание.
"""
from __future__ import annotations

import numpy as np

from .base import Agent
from .contracts import ReliabilityReport, Severity


def _norm(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x) or hi <= lo:
        return float("nan")
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.5))


class ReliabilityAgent(Agent):
    name = "reliability"
    role = "Агент надёжности — тяжесть режима, ресурс катализатора, границы для оптимизации"

    def on_assess(self, state, action=None) -> ReliabilityReport:
        cfg = self.ctx.config
        rc = cfg.agents["reliability"]
        ref = self.ctx.bundle.reference
        tags = self.ctx.bundle.meta["tags"]
        econ = cfg.economics["hydrotreater"]
        d_temp = getattr(action, "d_temp_c", 0.0) if action else 0.0

        t_now = state.mean(tags["temp"], 60)
        if not np.isfinite(t_now):
            t_now = ref["t_reactor_c"]
        t_eff = t_now + d_temp
        load = state.mean(tags["load"], 60)
        dp = state.mean(tags["dp"], 60)
        quench = state.mean(tags["quench"], 60)
        h2 = state.mean(tags["h2"], 60)

        factors = {
            "wabt": _norm(t_eff, ref["t_reactor_c"] - 2.0, ref["temp_p95"]),
            "dp_reactor": _norm(dp, ref["dp_median"], ref["dp_p95"]),
            "quench": _norm(-(quench / max(load, 1e-9)),
                            -(ref["quench_median"] / max(ref["load"], 1e-9)) * 1.15,
                            -(ref["quench_median"] / max(ref["load"], 1e-9)) * 0.70),
            "h2_to_oil": _norm(-(h2 / max(load, 1e-9)),
                               -(ref["h2"] / max(ref["load"], 1e-9)) * 1.20,
                               -(ref["h2"] / max(ref["load"], 1e-9)) * 0.75),
        }
        w = rc["severity_weights"]
        num = sum(w[k] * v for k, v in factors.items() if np.isfinite(v))
        den = sum(w[k] for k, v in factors.items() if np.isfinite(v))
        index = float(np.clip(num / den, 0.0, 1.5)) if den > 0 else float("nan")

        sev = Severity.OK
        if np.isfinite(index):
            if index >= float(rc["severity_alarm"]):
                sev = Severity.ALARM
            elif index >= float(rc["severity_warn"]):
                sev = Severity.WARN

        # ресурс цикла катализатора по правилу Аррениуса
        half = float(econ["catalyst_deactivation_degC_per_doubling"])
        base_days = float(econ["catalyst_base_cycle_days"])
        days = min(base_days * 0.5 ** ((t_eff - ref["t_reactor_c"]) / half), 2.0 * base_days)

        # коридор допустимого изменения температуры
        head_alarm = max(0.0, (float(rc["severity_alarm"]) - index)) / max(w["wabt"], 1e-6)
        span = max(ref["temp_p95"] - ref["t_reactor_c"] + 2.0, 1e-6)
        d_temp_max = float(np.clip(head_alarm * span, 0.0, ref["temp_p95"] - t_now))
        bounds = {"d_temp_c": (float(ref["temp_p5"] - t_now), d_temp_max)}

        notes = []
        if factors.get("dp_reactor", 0) > 0.8:
            notes.append("перепад давления на Р-202 близок к верхней границе наблюдавшегося "
                         "диапазона — повышение нагрузки нежелательно")
        if factors.get("h2_to_oil", 0) > 0.8:
            notes.append("кратность циркуляции водорода низкая — риск ускоренного "
                         "коксообразования при повышении температуры")
        if sev is Severity.ALARM:
            notes.append("режим признан недопустимо тяжёлым: повышение температуры запрещено")
        return ReliabilityReport(t=state.t, severity_index=index, severity=sev,
                                 factors=factors, bounds=bounds,
                                 catalyst_days_left=float(days), notes=notes)
