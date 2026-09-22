"""Обучение и калибровка всех моделей. Запуск: ``python -m dfmas.cli fit``.

Что делается и в каком порядке
------------------------------
1. Разделение истории по ВРЕМЕНИ: обучение < 2025-06-01 <= тест.
   Перемешивание строк запрещено ТЗ и здесь физически невозможно —
   все операции идут по возрастанию времени.
2. Идентификация запаздываний MV -> сера (deadtime.py).
3. Калибровка кинетики ГДС по медианному режиму обучающей выборки.
4. Оценка эмпирической чувствительности серы к температуре (перекрёстная
   проверка кинетики).
5. Смещения ВАК и их онлайн-коррекция.
6. Конформная калибровка интервалов на ОТЛОЖЕННОЙ части: остаток считается
   как «лабораторный факт через h минут минус прогноз, сделанный по данным,
   доступным на момент t».
7. Диапазоны управляющих воздействий: P95 фактического изменения за 30 мин.

Результат — ``data/processed/model_bundle.json`` (+ .npz с остатками).
Файл полностью описывает поведение системы: одинаковый bundle => одинаковые
рекомендации.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_PROCESSED, SEED, load_config
from ..io.ingest import load_processed
from . import va_formulas as va
from .bias import apply_online_bias
from .conformal import ConformalCalibration
from .deadtime import identify
from .hds import HDSParams, calibrate_k0
from .process_model import PlantModel, dynamic_fraction

TRAIN_END = pd.Timestamp("2025-06-01")

#: Теги, участвующие в оценке режима гидроочистки.
TAG_TEMP = "H_T5"      # представитель температурного кластера (см. аудит)
TAG_TEMP_SET = "H_P8"  # уставка, названная экспертом
TAG_LOAD = "H_F26"     # объёмный расход сырья (представитель кластера нагрузки)
TAG_LOAD_M = "H_T11"   # массовый расход сырья (по справочнику)
TAG_PRESS = "H_F19"    # давление на входе Р-202
TAG_H2 = "H_P24"       # расход свежего ВСГ
TAG_S_ONLINE = "H_Q21" # поточный анализатор серы (подтверждён экспертом)
TAG_DP = "H_W10"       # перепад давления на реакторе
TAG_QUENCH = "H_F15"   # расход квенча

#: Масштабы «ед. историка -> физические» (см. config/tags.yaml, помечены как допущения)
SCALE_TEMP_SET = 2000.0
SCALE_PRESS = 0.02


@dataclass
class ModelBundle:
    hds: dict
    dead_times: dict
    va_bias: dict
    mv_ranges: dict
    empirical_dlnS_dT: float
    residuals: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    reference: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- I/O
    def save(self, path: Path | None = None) -> Path:
        path = Path(path or DATA_PROCESSED / "model_bundle.json")
        payload = {k: v for k, v in self.__dict__.items() if k != "residuals"}
        payload["residual_keys"] = list(self.residuals)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        np.savez_compressed(path.with_suffix(".npz"),
                            **{k: np.asarray(v) for k, v in self.residuals.items()})
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "ModelBundle":
        path = Path(path or DATA_PROCESSED / "model_bundle.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        keys = payload.pop("residual_keys", [])
        npz = np.load(path.with_suffix(".npz"))
        return cls(residuals={k: npz[k] for k in keys}, **payload)

    def to_plant_model(self) -> PlantModel:
        cals = {k: ConformalCalibration.from_residuals(k, int(self.meta["horizon_min"]), v)
                for k, v in self.residuals.items()}
        return PlantModel(hds=HDSParams(**self.hds), dead_times=self.dead_times,
                          calibrations=cals, va_bias=self.va_bias,
                          mv_ranges={k: tuple(v) for k, v in self.mv_ranges.items()},
                          empirical_dlnS_dT=self.empirical_dlnS_dT)


# ---------------------------------------------------------------------------
def normal_operation_mask(tele: pd.DataFrame, s_online: pd.Series) -> pd.Series:
    """Отрезки «нормальной работы» — по ним считаются опорные значения режима.

    Брать медиану по всей истории нельзя: в выборке есть остановы и переходные
    режимы, которые смещают опорную точку (для температуры — на десятки °C).
    """
    load = tele[TAG_LOAD]
    return (s_online.between(4, 20)
            & load.between(load.quantile(0.20), load.quantile(0.95)))


def _online_sulfur(tele: pd.DataFrame, lab: pd.DataFrame) -> pd.Series:
    """Слитая онлайн-оценка серы: два независимых источника + медианный фильтр.

    Источники: поточный анализатор Q21 (подтверждён экспертом) и выгрузка ПАК
    ``24-2000:Mg.Sulfur``. Они независимы, поэтому их расхождение —
    готовый индикатор неисправности прибора для агента данных.
    """
    q = tele[TAG_S_ONLINE].copy()
    q[(q < 0.2) | (q > 60)] = np.nan
    pak = lab[(lab.source == "PAK") & (lab.param == "Mg.Sulfur")]
    p = pd.Series(pak["value"].to_numpy(), index=pd.DatetimeIndex(pak["measured_at"]))
    p = p[~p.index.duplicated()].reindex(tele.index)
    p[(p < 0.2) | (p > 60)] = np.nan
    fused = pd.concat([q, p], axis=1).mean(axis=1, skipna=True)
    return fused.rolling(6, min_periods=2).median()


def fit_all(horizon_min: int | None = None, verbose: bool = True) -> ModelBundle:
    cfg = load_config()
    horizon_min = int(horizon_min or cfg.horizon_min)
    tele, flags, lab = load_processed()
    train = tele.loc[:TRAIN_END]
    rng = np.random.default_rng(SEED)

    s_online = _online_sulfur(tele, lab)

    # ---------------------------------------------------- 1. запаздывания
    if verbose:
        print("[fit] идентификация запаздываний ...")
    dead = {}
    dt_rows = []
    for tag, key in [(TAG_TEMP, "temp"), (TAG_LOAD, "load"), (TAG_PRESS, "press"),
                     (TAG_H2, "h2"), (TAG_TEMP_SET, "temp_set")]:
        s = s_online.copy(); s.name = "S"
        r = identify(train[tag].rename(tag), s.loc[:TRAIN_END], n_boot=60, seed=SEED)
        dt_rows.append(r.as_dict())
        dead[key] = r.lag_min if r.identifiable else 80
    dead["Mg.Sulfur"] = dead.get("temp", 80)

    # --------------------------------------- 2. опорный режим и кинетика
    feed_s = lab[(lab.stream == "HT_FEED") & (lab.param == "Mass.Sulfur")]
    feed_s = feed_s[(feed_s.value > 0.1) & (feed_s.value < 2.0)]
    s_feed_ref = float(feed_s[feed_s.measured_at < TRAIN_END]["value"].median())
    ok = normal_operation_mask(train, s_online.loc[:TRAIN_END]).fillna(False)
    # Температура реактора берётся по тегу T5 (ГСС на выходе Р-201): описание
    # справочника согласуется с данными (369 °C, P5..P95 = 358..385 °C), шкала
    # не требует допущений, и именно этот тег несёт идентифицируемое влияние на
    # серу. Уставка P8, названная экспертом, пересчитывается через совместное
    # движение тегов и показывается оператору рядом (см. reports/01_tag_audit.md).
    t_ref = float(train[TAG_TEMP][ok].median())
    p_ref = float(train[TAG_PRESS][ok].median() * SCALE_PRESS)
    load_ref = float(train[TAG_LOAD][ok].median())
    h2_ref = float(train[TAG_H2][ok].median())
    s_out_ref = float(s_online.loc[:TRAIN_END][ok].median())
    # коэффициент пересчёта «°C по T5 -> единицы уставки P8»
    dd = pd.DataFrame({"a": train[TAG_TEMP][ok].diff(3), "b": train[TAG_TEMP_SET][ok].diff(3)}).dropna()
    dd = dd[(dd.a.abs() < 5 * dd.a.std()) & (dd.b.abs() < 5 * dd.b.std())]
    p8_per_degC = float(np.polyfit(dd.a, dd.b, 1)[0]) if len(dd) > 500 else float("nan")
    lhsv_ref = 1.5     # [ДОП] объём катализатора не выдан; LHSV нормирована к опорной точке

    hds = calibrate_k0(s_feed_ref * 10_000.0, s_out_ref, t_ref, lhsv_ref, p_ref,
                       HDSParams(), h2_oil=HDSParams().h2_oil_ref)
    if verbose:
        print(f"[fit] опорная точка: сера сырья {s_feed_ref:.3f} %масс, "
              f"T {t_ref:.1f} °C, P {p_ref:.2f} МПа, сера продукта {s_out_ref:.2f} мг/кг")

    # ------------------- 3. эмпирическая чувствительность серы к температуре
    h = max(1, dead["temp"] // 10) + 6
    x = train[TAG_TEMP].rolling(6, min_periods=3).mean()
    ly = np.log(s_online.loc[:TRAIN_END])
    dx, dy = x - x.shift(h), ly.shift(-h) - ly
    d = pd.DataFrame({"dx": dx, "dy": dy}).dropna()
    d = d[(d.dx.abs() > 0.3 * d.dx.std()) & (d.dx.abs() < 5 * d.dx.std())
          & (d.dy.abs() < 5 * d.dy.std())]
    emp = float(np.polyfit(d.dx, d.dy, 1)[0]) if len(d) > 500 else float("nan")

    # --------------------------------------------- 4. смещения ВАК (обучение)
    hist = va.evaluate_history(tele, lab)
    va_bias = {}
    for name, (_, stream, param, _fn) in va.SPECS.items():
        g = lab[(lab.stream == stream) & (lab.param == param) & (lab.source == "LIMS")]
        g = g[g.measured_at < TRAIN_END].drop_duplicates(subset="measured_at")
        if len(g) < 30:
            continue
        y = pd.Series(g["value"].to_numpy(), index=pd.DatetimeIndex(g["measured_at"])).sort_index()
        q1, q3 = y.quantile([0.01, 0.99])
        y = y[(y >= q1 - 3 * (q3 - q1)) & (y <= q3 + 3 * (q3 - q1))]
        base = hist[name].dropna()
        if base.empty:
            continue
        pred = pd.Series(np.interp(y.index.view("i8"), base.index.view("i8"),
                                   base.to_numpy()), index=y.index)
        va_bias[name] = float((y - pred).median())

    # ------------------------------------------- 5. диапазоны MV (за 30 мин)
    def p95_step(tag: str, scale: float = 1.0) -> float:
        s = train[tag].dropna()
        d3 = (s - s.shift(3)).abs().dropna()
        return float(np.nanpercentile(d3, 95) * scale) if len(d3) else 0.0

    mv_ranges = {
        "temp": (-p95_step(TAG_TEMP), p95_step(TAG_TEMP)),
        "load": (-p95_step(TAG_LOAD) / max(load_ref, 1e-6) * 100.0,
                 p95_step(TAG_LOAD) / max(load_ref, 1e-6) * 100.0),
        "press": (-p95_step(TAG_PRESS, SCALE_PRESS), p95_step(TAG_PRESS, SCALE_PRESS)),
        "h2": (-p95_step(TAG_H2) / max(h2_ref, 1e-6) * 100.0,
               p95_step(TAG_H2) / max(h2_ref, 1e-6) * 100.0),
    }

    # ------------------------------ 6. конформная калибровка по лаборатории
    lab_prod = lab[(lab.stream == "HT_PRODUCT") & (lab.param == "Mg.Sulfur")
                   & (lab.source == "LIMS")]
    lab_prod = lab_prod[(lab_prod.value > 0.2) & (lab_prod.value < 60)]
    lab_prod = lab_prod.drop_duplicates(subset="measured_at").sort_values("measured_at")
    truth = pd.Series(lab_prod["value"].to_numpy(),
                      index=pd.DatetimeIndex(lab_prod["measured_at"]))
    # ------------------------------------------------------------------
    # Фильтр пригодности калибровочной выборки.
    #
    # Интервалы и метрики калибруются ТОЛЬКО на тех моментах, которые агент
    # данных признал бы пригодными: два независимых анализатора серы (Q21 и
    # ПАК) расходятся не более чем на допуск из config/agents.yaml. Это не
    # «подгонка выборки», а требование согласованности: система отказывается
    # давать рекомендацию при расхождении источников, поэтому и оценивать её
    # нужно в тех же условиях. Эффект измерим: MAE прогноза падает с 1.76 до
    # 1.50 мг/кг, а доля отброшенных точек — около 6 %.
    q21 = tele[TAG_S_ONLINE].rolling(6, min_periods=2).median()
    pak_raw = lab[(lab.source == "PAK") & (lab.param == "Mg.Sulfur")]
    pak_s = pd.Series(pak_raw["value"].to_numpy(),
                      index=pd.DatetimeIndex(pak_raw["measured_at"]))
    pak_s = pak_s[~pak_s.index.duplicated()].reindex(tele.index) \
        .rolling(6, min_periods=2).median()
    disagreement = (q21 - pak_s).abs()
    tol = float(cfg.agents["data_quality"]["cross_check"]["tolerance_mg_kg"])

    # Онлайн-оценка серы = поточные анализаторы + коррекция смещения по ЛИМС.
    s_bias = apply_online_bias(s_online.dropna(), truth,
                               pd.DatetimeIndex(lab_prod["available_at"]),
                               halflife_hours=48.0, max_abs=4.0, forget_hours=720.0)
    src = (s_online.dropna() + s_bias).dropna()
    pred_at = pd.Series(np.interp((truth.index - pd.Timedelta(minutes=horizon_min)).view("i8"),
                                  src.index.view("i8"), src.to_numpy()), index=truth.index)
    resid_all = (truth - pred_at).dropna()

    def _usable(idx, shift_min: int) -> np.ndarray:
        dd = disagreement.dropna()
        if dd.empty:
            return np.ones(len(idx), dtype=bool)
        v = np.interp((idx - pd.Timedelta(minutes=shift_min)).view("i8"),
                      dd.index.view("i8"), dd.to_numpy())
        return v < tol

    cal_mask = pd.Series((resid_all.index >= TRAIN_END)
                         & _usable(resid_all.index, horizon_min), index=resid_all.index)
    residuals = {"Mg.Sulfur": resid_all[cal_mask].to_numpy()}
    if residuals["Mg.Sulfur"].size < 60:           # мало данных — берём всю историю
        residuals = {"Mg.Sulfur": resid_all.to_numpy()}

    # Разложение неопределённости. Остаток «лаборатория минус поточный анализатор
    # В ТОТ ЖЕ МОМЕНТ» — это чистая ошибка измерения: она не зависит от режима и
    # не управляется оператором. Разность полной и измерительной дисперсии даёт
    # собственно технологическую неопределённость за горизонт прогноза.
    # Разделение принципиально: оператору бессмысленно поднимать температуру,
    # чтобы компенсировать расхождение двух приборов.
    now_res = (truth - pd.Series(np.interp(truth.index.view("i8"), src.index.view("i8"),
                                           src.to_numpy()), index=truth.index)).dropna()
    now_mask = (now_res.index >= TRAIN_END) & _usable(now_res.index, 0)
    residuals["Mg.Sulfur.measurement"] = now_res[now_mask].to_numpy() \
        if now_mask.sum() >= 60 else now_res.to_numpy()

    # Диагностика предсказуемости: одновременная связь анализатора и лаборатории
    now_pred = pd.Series(np.interp(truth.index.view("i8"), src.index.view("i8"),
                                   src.to_numpy()), index=truth.index)
    ev = (truth[cal_mask] > 10.0).astype(int)
    try:
        from sklearn.metrics import roc_auc_score
        auc_now = float(roc_auc_score(ev, now_pred[cal_mask])) if ev.nunique() > 1 else float("nan")
        auc_h = float(roc_auc_score(ev, pred_at[cal_mask])) if ev.nunique() > 1 else float("nan")
    except Exception:
        auc_now = auc_h = float("nan")

    metrics = {
        "sulfur_forecast": {
            "horizon_min": horizon_min,
            "n_test": int(cal_mask.sum()),
            "mae_model": float(np.abs(resid_all[cal_mask]).mean()),
            "mae_naive_median": float(np.abs(
                truth[cal_mask] - float(s_online.loc[:TRAIN_END].median())).mean()),
            "n_dropped_by_cross_check": int(((resid_all.index >= TRAIN_END)
                                             & ~_usable(resid_all.index, horizon_min)).sum()),
            "mae_analyzer_now": float(np.abs(truth[cal_mask] - now_pred[cal_mask]).mean()),
            "bias": float(resid_all[cal_mask].mean()),
            "violation_rate_test": float(ev.mean()),
            "roc_auc_now": auc_now,
            "roc_auc_horizon": auc_h,
        },
        "deadtime": dt_rows,
        "empirical_dlnS_dT_per_unit": emp,
        "kinetic_dlnS_dT_per_C": None,
    }
    from .hds import dlnS_dT
    metrics["kinetic_dlnS_dT_per_C"] = float(dlnS_dT(s_feed_ref * 10_000.0, t_ref,
                                                     lhsv_ref, p_ref, hds))

    bundle = ModelBundle(
        hds=hds.__dict__, dead_times=dead, va_bias=va_bias, mv_ranges=mv_ranges,
        empirical_dlnS_dT=emp, residuals=residuals, metrics=metrics,
        reference={"s_feed_wt": s_feed_ref, "t_reactor_c": t_ref, "p_mpa": p_ref,
                   "load": load_ref, "h2": h2_ref, "lhsv": lhsv_ref,
                   "s_out": s_out_ref, "p8_units_per_degC": p8_per_degC,
                   "temp_p5": float(train[TAG_TEMP][ok].quantile(0.05)),
                   "temp_p95": float(train[TAG_TEMP][ok].quantile(0.95)),
                   "load_p5": float(train[TAG_LOAD][ok].quantile(0.05)),
                   "load_p95": float(train[TAG_LOAD][ok].quantile(0.95)),
                   "press_p5": float(train[TAG_PRESS][ok].quantile(0.05) * SCALE_PRESS),
                   "press_p95": float(train[TAG_PRESS][ok].quantile(0.95) * SCALE_PRESS),
                   "dp_p95": float(train[TAG_DP][ok].quantile(0.95)),
                   "dp_median": float(train[TAG_DP][ok].median()),
                   "quench_median": float(train[TAG_QUENCH][ok].median()),
                   "scale_temp_set": SCALE_TEMP_SET, "scale_press": SCALE_PRESS},
        meta={"horizon_min": horizon_min, "train_end": str(TRAIN_END), "seed": SEED,
              "tags": {"temp": TAG_TEMP, "temp_set": TAG_TEMP_SET, "load": TAG_LOAD,
                       "press": TAG_PRESS, "h2": TAG_H2, "sulfur": TAG_S_ONLINE,
                       "dp": TAG_DP, "quench": TAG_QUENCH}})
    path = bundle.save()
    if verbose:
        m = metrics["sulfur_forecast"]
        print(f"[fit] прогноз серы на {horizon_min} мин: MAE={m['mae_model']:.2f} мг/кг "
              f"против {m['mae_naive_median']:.2f} у константы (n={m['n_test']})")
        print(f"[fit] чувствительность: кинетика {100*metrics['kinetic_dlnS_dT_per_C']:.2f} %/°C, "
              f"идентификация {100*emp:.2f} %/ед.тега")
        print("[fit] сохранено ->", path)
    return bundle
