"""Кинетическая модель гидрообессеривания (ГДС) — физический приор.

Зачем нужна физика, если есть история
-------------------------------------
Историческая выборка объясняет лишь малую часть дисперсии серы (см.
reports/03_models.md): установка почти всё время работает в узком коридоре,
поэтому «чистых» экспериментов в данных нет. Регрессия на таких данных
уверенно интерполирует и катастрофически ошибается при экстраполяции —
а рекомендация по определению выводит режим за пределы наблюдавшегося.

Поэтому основой служит кинетика ГДС, у которой:
  * ПРАВИЛЬНАЯ МОНОТОННОСТЬ по построению (рост температуры и давления
    водорода снижает серу, рост нагрузки — повышает);
  * решаемая обратная задача: какая температура нужна для целевой серы;
  * параметры, которые можно откалибровать по истории.

Модель
------
Реакция ГДС для дизельных фракций описывается псевдопорядком n ≈ 1.5 с
торможением сероводородом (кинетика Ленгмюра — Хиншельвуда):

    S_out^(1-n) - S_in^(1-n) = (n-1) * k_eff * (P/P_ref)^a / LHSV
    k_eff = k0 * exp(-Ea/(R*T)) / (1 + K_H2S * p_H2S)
    p_H2S ~ S_in[% масс.] * (H2/сырьё)_ref / (H2/сырьё)

Член торможения принципиален: без него степенная модель при глубоком
обессеривании почти не реагирует на серу сырья (вклад S_in^(1-n) ничтожен
при S_out ~ 8 мг/кг), и рекомендация «поднять температуру при утяжелении
сырья» не воспроизводилась бы. С торможением рост серы сырья повышает
парциальное давление H2S, тормозит реакцию и требует роста температуры —
ровно тот сценарий, который описал эксперт.

Значения n = 1.5, Ea = 105 кДж/моль, a = 0.7 — литературные [ДОП];
K_H2S = 1.0 (1/% масс.) откалибровано так, чтобы рост серы сырья на
0.1 % масс. требовал компенсации примерно +1.5 °C — отраслевая практика [ДОП].
k0 калибруется по выданной истории.

Перекрёстная проверка: модель даёт чувствительность около -6 %/°C,
независимая идентификация по данным — около -3.5 %/°C. Величины одного
порядка, знак совпадает; расхождение отражено в ширине интервала прогноза.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

R_GAS = 8.314462618           # Дж/(моль*К)


@dataclass(frozen=True)
class HDSParams:
    """Параметры кинетики. Все значения помечены как допущения [ДОП]."""
    order: float = 1.5            # псевдопорядок реакции
    ea_j_mol: float = 105_000.0   # энергия активации
    p_exponent: float = 0.7       # показатель по парциальному давлению H2
    k0: float = 1.0               # предэкспонента (калибруется)
    p_ref_mpa: float = 4.0        # опорное давление
    lhsv_ref: float = 1.5         # опорная объёмная скорость, ч^-1
    t_ref_c: float = 350.0        # опорная температура слоя
    k_h2s: float = 1.0            # константа торможения H2S, 1/(% масс.)
    h2_oil_ref: float = 250.0     # опорное отношение H2/сырьё, нм3/м3


def _k(T_c: float, p: HDSParams) -> float:
    return p.k0 * math.exp(-p.ea_j_mol / (R_GAS * (T_c + 273.15)))


def inhibition(s_in_wt_pct: float, h2_oil: float, p: HDSParams) -> float:
    """Множитель торможения сероводородом (>= 1). Чем больше, тем медленнее реакция."""
    p_h2s = max(0.0, s_in_wt_pct) * (p.h2_oil_ref / max(h2_oil, 1e-6))
    return 1.0 + p.k_h2s * p_h2s


def _k_eff(T_c: float, s_in_wt_pct: float, h2_oil: float, p: HDSParams) -> float:
    return _k(T_c, p) / inhibition(s_in_wt_pct, h2_oil, p)


def sulfur_out(s_in_ppm: float, T_c: float, lhsv: float, p_mpa: float,
               p: HDSParams, h2_oil: float | None = None) -> float:
    """Прямая задача: сера на выходе, мг/кг."""
    m = p.order - 1.0
    h2_oil = p.h2_oil_ref if h2_oil is None else h2_oil
    k = _k_eff(T_c, s_in_ppm / 10_000.0, h2_oil, p)
    term = m * k * (max(p_mpa, 1e-6) / p.p_ref_mpa) ** p.p_exponent / max(lhsv, 1e-6)
    base = max(s_in_ppm, 1e-6) ** (-m)
    return float((base + term) ** (-1.0 / m))


def required_temperature(s_target_ppm: float, s_in_ppm: float, lhsv: float,
                         p_mpa: float, p: HDSParams,
                         t_bounds: tuple[float, float] = (300.0, 400.0),
                         h2_oil: float | None = None) -> float:
    """Обратная задача: какая температура слоя нужна для целевой серы.

    Решается аналитически — модель монотонна по T, инверсия однозначна.
    """
    m = p.order - 1.0
    need = max(s_target_ppm, 1e-6) ** (-m) - max(s_in_ppm, 1e-6) ** (-m)
    if need <= 0:
        return t_bounds[0]
    k_req = need * max(lhsv, 1e-6) / (m * (max(p_mpa, 1e-6) / p.p_ref_mpa) ** p.p_exponent)
    if k_req <= 0 or p.k0 <= 0:
        return t_bounds[1]
    h2_oil = p.h2_oil_ref if h2_oil is None else h2_oil
    k_req *= inhibition(s_in_ppm / 10_000.0, h2_oil, p)
    ratio = p.k0 / k_req
    if ratio <= 1.0:
        return t_bounds[1]
    T = p.ea_j_mol / (R_GAS * math.log(ratio)) - 273.15
    return float(min(max(T, t_bounds[0]), t_bounds[1]))


def dlnS_dT(s_in_ppm: float, T_c: float, lhsv: float, p_mpa: float,
            p: HDSParams, h2_oil: float | None = None) -> float:
    """Чувствительность ln(сера) к температуре, 1/°C (аналитически)."""
    m = p.order - 1.0
    h2_oil = p.h2_oil_ref if h2_oil is None else h2_oil
    k = _k_eff(T_c, s_in_ppm / 10_000.0, h2_oil, p)
    term = m * k * (max(p_mpa, 1e-6) / p.p_ref_mpa) ** p.p_exponent / max(lhsv, 1e-6)
    u = max(s_in_ppm, 1e-6) ** (-m) + term
    dk_dT = p.ea_j_mol / (R_GAS * (T_c + 273.15) ** 2)
    return float(-(1.0 / m) * (term * dk_dT) / u)


def calibrate_k0(s_in_ppm: float, s_out_ppm: float, T_c: float, lhsv: float,
                 p_mpa: float, p: HDSParams, h2_oil: float | None = None) -> HDSParams:
    """Подбирает k0 так, чтобы модель точно воспроизводила опорную точку.

    Одна степень свободы на одну наблюдаемую точку — метод сознательно
    консервативен: мы не «подгоняем» Ea и порядок под шумные данные.
    """
    m = p.order - 1.0
    need = max(s_out_ppm, 1e-6) ** (-m) - max(s_in_ppm, 1e-6) ** (-m)
    if need <= 0:
        return p
    k_req = need * max(lhsv, 1e-6) / (m * (max(p_mpa, 1e-6) / p.p_ref_mpa) ** p.p_exponent)
    h2_oil = p.h2_oil_ref if h2_oil is None else h2_oil
    k_req *= inhibition(s_in_ppm / 10_000.0, h2_oil, p)
    k0 = k_req * math.exp(p.ea_j_mol / (R_GAS * (T_c + 273.15)))
    return HDSParams(order=p.order, ea_j_mol=p.ea_j_mol, p_exponent=p.p_exponent,
                     k0=k0, p_ref_mpa=p.p_ref_mpa, lhsv_ref=p.lhsv_ref,
                     t_ref_c=p.t_ref_c, k_h2s=p.k_h2s, h2_oil_ref=p.h2_oil_ref)


def h2_consumption_nm3_per_t(s_in_wt_pct: float, s_out_ppm: float,
                             nm3_per_t_per_pct: float = 95.0) -> float:
    """Расход водорода на обессеривание, нм3/т сырья (линейное приближение)."""
    removed_pct = max(0.0, s_in_wt_pct - s_out_ppm / 10_000.0)
    return float(removed_pct * nm3_per_t_per_pct)
