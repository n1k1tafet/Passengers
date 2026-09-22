"""Агент блендинга: рецептура товарного ДТ.

Материалов по блендингу в пакете нет — эксперт подтвердил, что допустим явно
модельный сценарий с описанными допущениями. Здесь он и реализован.

Модель смешения (все допущения помечены [ДОП]):
  * сера           — аддитивна по массе;
  * Т95, плотность — аддитивны по объёму (стандартное инженерное приближение);
  * цетановое число — аддитивно по объёму для базовых компонентов;
  * цетаноповышающая присадка — кусочно-линейный отклик с насыщением:
    первые 0.3 % об. дают 25 ед./%, далее 5 ед./% (поведение 2-этилгексилнитрата).

Заданные экспертом условия воспроизведены точно:
  * доля присадки не более 3 %;
  * тонна присадки стоит как 100 тонн ДТ;
  * чем больше серы в гидроочищенном ДТ, тем дешевле его производство —
    себестоимость резервуаров считается ТОЙ ЖЕ моделью, что и режим
    гидроочистки (кинетика + экономика), поэтому цепочка непротиворечива.

Задача решается линейным программированием (scipy.optimize.linprog):
минимизируем себестоимость тонны смеси при жёстких ограничениях качества и
условии, что доли составляют 100 %. Если допустимого решения нет, агент
честно возвращает ``feasible=False`` с причиной.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from ..models.hds import HDSParams, required_temperature
from .base import Agent
from .contracts import BlendComponent, BlendPlan

#: Кусочно-линейный отклик цетанового числа на присадку [ДОП]
ADDITIVE_SEGMENTS = ((0.003, 25.0 / 0.01), (0.03, 5.0 / 0.01))   # (доля, ед./доля)

#: Собственные свойства цетаноповышающей присадки (2-этилгексилнитрат) [ДОП].
#: Учитываются, чтобы при дозировке в несколько процентов баланс по плотности
#: и фракционному составу оставался корректным.
ADDITIVE_PROPS = {"sulfur_mg_kg": 0.0, "t95_c": 200.0, "density": 965.0}


@dataclass
class Tank:
    """Резервуар гидроочищенного ДТ с известным качеством."""
    name: str
    sulfur_mg_kg: float
    t95_c: float
    cetane: float
    density: float
    cfpp_c: float = -6.0          # предельная температура фильтруемости
    available_share: float = 1.0
    cost_rub_t: float | None = None
    cost_premium_rub_t: float = 0.0   # надбавка к расчётной себестоимости


def default_tanks(bundle, econ) -> list[Tank]:
    """Три партии разной глубины обессеривания — опорный модельный сценарий.

    Качество взято из наблюдавшегося в ЛИМС диапазона гидроочищенного ДТ,
    себестоимость рассчитана кинетикой: более глубокое обессеривание требует
    более высокой температуры, а значит топлива, водорода и ресурса катализатора.
    """
    return [
        Tank("Р-1 глубокая очистка", 3.0, 344.0, 52.5, 834.0, cfpp_c=-12.0),
        Tank("Р-2 базовый режим", 8.5, 347.0, 51.2, 836.0, cfpp_c=-8.0),
        Tank("Р-3 мягкий режим", 16.0, 351.0, 50.0, 839.0, cfpp_c=-5.0),
        # Зимняя облегчённая фракция: узкий отбор даёт низкую ПТФ, но снижает
        # выход дизельного топлива, поэтому себестоимость выше. [ДОП]
        Tank("Р-4 зимняя облегчённая", 6.0, 338.0, 50.5, 828.0, cfpp_c=-26.0,
             cost_premium_rub_t=2600.0),
    ]


class BlendingAgent(Agent):
    name = "blending"
    role = "Агент блендинга — рецептура товарного ДТ и дозировка присадки"

    # ------------------------------------------------------------------ цена
    def tank_cost(self, tank: Tank) -> float:
        """Себестоимость тонны из резервуара, ₽/т, через требуемую температуру."""
        if tank.cost_rub_t is not None:
            return float(tank.cost_rub_t) + float(tank.cost_premium_rub_t)
        e = self.ctx.config.economics
        h = e["hydrotreater"]
        ref = self.ctx.bundle.reference
        hds = HDSParams(**self.ctx.bundle.hds)
        t_req = required_temperature(tank.sulfur_mg_kg, ref["s_feed_wt"] * 10_000.0,
                                     ref["lhsv"], ref["p_mpa"], hds)
        fuel = float(h["fuel_cost_per_degC_per_t"]) * (t_req - ref["t_reactor_c"])
        h2 = (max(0.0, ref["s_feed_wt"] - tank.sulfur_mg_kg / 10_000.0)
              * float(h["h2_nm3_per_t_per_pct_S"]) * float(h["h2_cost_per_nm3"]))
        half = float(h["catalyst_deactivation_degC_per_doubling"])
        days = float(h["catalyst_base_cycle_days"]) * 0.5 ** ((t_req - ref["t_reactor_c"]) / half)
        cat_per_t = float(h["catalyst_cycle_cost"]) / max(days * 24.0 * max(ref["load"], 1.0), 1.0)
        return float(e["prices"]["straight_run_diesel_t"] + fuel + h2 + cat_per_t
                     + tank.cost_premium_rub_t)

    # ------------------------------------------------------------------ план
    def safety_margin(self, param: str) -> float:
        """Запас до предела, закрывающий разброс измерения.

        Линейная программа по своей природе прижимает смесь ровно к пределу —
        экономически это оптимум, операционно это брак при первом же анализе.
        Поэтому предел ужесточается на 90-й процентиль расхождения
        «лаборатория — поточный анализатор» (models/conformal.py).
        Запас рассчитан по данным, а не назначен «на глаз».
        """
        cal = self.ctx.plant.calibrations.get(param + ".measurement")
        if cal is None or cal.n < 20:
            return 0.0
        return float(max(0.0, np.quantile(cal.residuals, 0.90)))

    def on_plan(self, grade: str, tanks: list[Tank] | None = None,
                target_overrides: dict | None = None) -> BlendPlan:
        cfg = self.ctx.config
        e = cfg.economics
        limits = dict(cfg.grade(grade)["limits"])
        if target_overrides:
            for k, v in target_overrides.items():
                limits.setdefault(k, {}).update(v)
        tanks = tanks or self.ctx.extras.get("tanks") or default_tanks(self.ctx.bundle, e)
        costs = [self.tank_cost(t) for t in tanks]
        add_cost = float(e["prices"]["cetane_additive_t"])

        n = len(tanks)
        # Переменные: доли резервуаров x[0..n-1] и два сегмента присадки a1, a2.
        # Все доли — от одной объёмной основы, сумма ровно 1, поэтому
        # ограничения пишутся как «свойство смеси vs предел» без нормировки
        # на Sum(x). Прежняя форма давала погрешность порядка доли присадки,
        # из-за которой смесь могла оказаться чуть вне норматива.
        seg_caps = [ADDITIVE_SEGMENTS[0][0],
                    ADDITIVE_SEGMENTS[1][0] - ADDITIVE_SEGMENTS[0][0]]
        seg_gain = [ADDITIVE_SEGMENTS[0][1], ADDITIVE_SEGMENTS[1][1]]
        ap = ADDITIVE_PROPS
        nv = n + 2

        c = np.array(costs + [add_cost, add_cost], dtype=float)
        A_eq = np.ones((1, nv)); b_eq = np.array([1.0])
        A_ub, b_ub = [], []

        def add_row(coef, rhs):
            A_ub.append([float(x) for x in coef]); b_ub.append(float(rhs))

        margin_s = self.safety_margin("Mg.Sulfur")
        if "Mg.Sulfur" in limits:                       # сера смеси <= предел с запасом
            lim = float(limits["Mg.Sulfur"]["max"]) - margin_s
            add_row([t.sulfur_mg_kg for t in tanks] + [ap["sulfur_mg_kg"]] * 2, lim)
        if "95%.T" in limits:
            add_row([t.t95_c for t in tanks] + [ap["t95_c"]] * 2,
                    float(limits["95%.T"]["max"]))
        if "CetaneNumber" in limits:                    # ЦЧ смеси >= предел
            add_row([-t.cetane for t in tanks] + [-seg_gain[0], -seg_gain[1]],
                    -float(limits["CetaneNumber"]["min"]))
        if "D15" in limits:
            d = limits["D15"]
            if "max" in d:
                add_row([t.density for t in tanks] + [ap["density"]] * 2, float(d["max"]))
            if "min" in d:
                add_row([-t.density for t in tanks] + [-ap["density"]] * 2, -float(d["min"]))
        n_hard_rows = len(A_ub)
        # Рекомендательные ограничения добавляются последними, чтобы их можно
        # было снять одним срезом, если задача окажется несовместной.
        if "CFPP" in limits:
            add_row([t.cfpp_c for t in tanks] + [0.0, 0.0], float(limits["CFPP"]["max"]))

        bounds = [(0.0, float(t.available_share)) for t in tanks] + \
                 [(0.0, seg_caps[0]), (0.0, seg_caps[1])]

        def solve(rows, rhs):
            return linprog(c, A_ub=np.array(rows) if rows else None,
                           b_ub=np.array(rhs) if rhs else None,
                           A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        relaxed = []
        res = solve(A_ub, b_ub)
        if not res.success and len(A_ub) > n_hard_rows:
            # Рекомендательные требования не выполнимы — снимаем их и честно
            # сообщаем об этом, вместо того чтобы молча выдать рецептуру.
            relaxed = ["ПТФ"]
            res = solve(A_ub[:n_hard_rows], b_ub[:n_hard_rows])
        if not res.success:
            return BlendPlan(grade=grade, feasible=False,
                             message=f"допустимой рецептуры нет (предел по сере ужесточён на "
                                     f"{margin_s:.2f} мг/кг на разброс измерения): "
                                     "имеющиеся партии не позволяют "
                                     "одновременно выполнить все жёсткие требования "
                                     f"(серa ≤ {limits.get('Mg.Sulfur',{}).get('max','-')} мг/кг, "
                                     f"ЦЧ ≥ {limits.get('CetaneNumber',{}).get('min','-')}, "
                                     f"Т95 ≤ {limits.get('95%.T',{}).get('max','-')} °C)")

        x = res.x[:n]; a = res.x[n:]
        add_share = float(a.sum())
        comps = [BlendComponent(name=t.name, share=float(xi), sulfur_mg_kg=t.sulfur_mg_kg,
                                t95_c=t.t95_c, cetane=t.cetane, density=t.density,
                                cost_rub_t=cost)
                 for t, xi, cost in zip(tanks, x, costs) if xi > 1e-9]
        blended = {
            "CFPP": float(sum(t.cfpp_c * xi for t, xi in zip(tanks, x))),
            "Mg.Sulfur": float(sum(t.sulfur_mg_kg * xi for t, xi in zip(tanks, x))
                               + ap["sulfur_mg_kg"] * add_share),
            "95%.T": float(sum(t.t95_c * xi for t, xi in zip(tanks, x))
                           + ap["t95_c"] * add_share),
            "CetaneNumber": float(sum(t.cetane * xi for t, xi in zip(tanks, x))
                                  + seg_gain[0] * a[0] + seg_gain[1] * a[1]),
            "D15": float(sum(t.density * xi for t, xi in zip(tanks, x))
                         + ap["density"] * add_share),
        }
        return BlendPlan(grade=grade, feasible=True, components=comps,
                         additive_share=add_share, blended=blended,
                         cost_rub_t=float(res.fun),
                         message=(("ВНИМАНИЕ: рекомендательные требования (" +
                                   ", ".join(relaxed) + ") выполнить не удалось, "
                                   "они сняты. " if relaxed else "") +
                                  f"расчётный предел по сере ужесточён на {margin_s:.2f} мг/кг "
                                  f"(90-й процентиль расхождения лаборатории и поточного "
                                  f"анализатора). " +
                                  ("присадка не требуется" if add_share < 1e-6 else
                                   f"требуется присадка {add_share*100:.2f} % об. "
                                   f"(стоимость присадки в 100 раз выше ДТ — доля минимизирована)")))
