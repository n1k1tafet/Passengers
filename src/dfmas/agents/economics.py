"""Агент экономики: прозрачный стоимостной прокси.

Фактических экономических данных в пакете нет, поэтому ТЗ прямо разрешает
прокси. Все цены и коэффициенты вынесены в ``config/economics.yaml`` и
помечены как допущения. Важны не абсолютные суммы, а СООТНОШЕНИЯ, задающие
компромисс.

Единственное, что задано экспертом и воспроизведено точно:
  * тонна цетаноповышающей присадки стоит как 100 тонн ДТ;
  * чем выше остаточная сера в ГО ДТ, тем дешевле его производство —
    у нас это получается само: меньшая глубина обессеривания означает более
    низкую температуру, а значит меньше топлива в печи, меньше водорода и
    медленнее дезактивацию катализатора.

Считаются ПЕРЕМЕННЫЕ затраты, ₽/ч:
  топливо печи + водород + компрессия + амортизация ресурса катализатора.
"""
from __future__ import annotations

import numpy as np

from ..models.hds import h2_consumption_nm3_per_t
from .base import Agent


class EconomicsAgent(Agent):
    name = "economics"
    role = "Агент экономики — переменные затраты, выпуск, стоимость ресурса катализатора"

    def on_evaluate(self, state, feed, action, sulfur_forecast: float,
                    t_reactor_c: float, load_t_h: float) -> dict:
        e = self.ctx.config.economics
        h = e["hydrotreater"]
        ref = self.ctx.bundle.reference

        load_new = load_t_h * (1.0 + action.d_load_pct / 100.0)
        t_new = t_reactor_c + action.d_temp_c

        fuel = float(h["fuel_cost_per_degC_per_t"]) * (t_new - ref["t_reactor_c"]) * load_new
        h2_nm3 = h2_consumption_nm3_per_t(feed.sulfur_wt, sulfur_forecast,
                                          float(h["h2_nm3_per_t_per_pct_S"])) * load_new
        h2_cost = h2_nm3 * float(h["h2_cost_per_nm3"])
        recycle = (load_new / max(ref["load"], 1e-9)) * 250.0 * load_new \
            * float(h["recycle_cost_per_nm3"]) * 0.001

        half = float(h["catalyst_deactivation_degC_per_doubling"])
        # Оценка ресурса ограничена сверху удвоенным базовым циклом: правило
        # Аррениуса корректно описывает УСКОРЕНИЕ дезактивации, но не даёт
        # права обещать бесконечный ресурс при работе ниже опорной температуры.
        days = float(h["catalyst_base_cycle_days"]) * 0.5 ** ((t_new - ref["t_reactor_c"]) / half)
        days = float(min(days, 2.0 * float(h["catalyst_base_cycle_days"])))
        catalyst = float(h["catalyst_cycle_cost"]) / max(days * 24.0, 1.0)

        cost = float(fuel + h2_cost + recycle + catalyst)
        revenue = load_new * float(e["prices"]["diesel_t"])
        return {"cost_rub_h": cost, "fuel_rub_h": float(fuel), "h2_rub_h": float(h2_cost),
                "recycle_rub_h": float(recycle), "catalyst_rub_h": float(catalyst),
                "throughput_t_h": float(load_new), "gross_revenue_rub_h": float(revenue),
                "margin_rub_h": float(revenue - cost),
                "catalyst_days_left": float(days)}
