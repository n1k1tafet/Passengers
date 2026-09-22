"""Бэктест на отложенной по времени истории.

Чем этот бэктест отличается от «прогнали модель на тесте»
---------------------------------------------------------
Проверяется не модель, а СИСТЕМА ЦЕЛИКОМ: на каждом шаге запускается полный
мультиагентный цикл ровно тем же кодом, что и в онлайне, и через ту же
витрину «на момент времени». Поэтому:

* временная утечка невозможна конструктивно — агенты физически не видят
  данных с ``available_at > t``;
* проверяется в том числе поведение при плохих данных: отказы попадают в
  статистику наравне с рекомендациями.

Что измеряется
--------------
1. **Распределение решений** — сколько раз система молчала, вмешивалась,
   отказывалась. Показывает, не «дёргает» ли она установку.
2. **Калибровка риска** — главный проверяемый показатель. Если система
   заявила риск 10 %, нарушения должны происходить примерно в 10 % случаев.
   Это единственная метрика, которую можно честно проверить на истории:
   она не требует знать, что было бы при другом управлении.
3. **Качество удержания в спецификации при решении «не вмешиваться»** —
   доля случаев, когда система сказала «всё в порядке», а факт оказался
   нарушением. Это её ошибка первого рода.

Чего бэктест НЕ доказывает (и это честно сказано в отчёте): эффект от
рекомендаций, которых в истории не было. Как подтвердил эксперт, оценка
альтернативных действий выполняется модельно и отдельно.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .io.ingest import load_processed
from .models.fit import TRAIN_END, _online_sulfur
from .system import DarkFactorySystem

SULFUR_LIMIT = 10.0


@dataclass
class BacktestResult:
    rows: pd.DataFrame
    decisions: dict
    calibration: pd.DataFrame
    summary: dict = field(default_factory=dict)


def run_backtest(system: DarkFactorySystem, start: str = "2025-06-01",
                 end: str = "2026-08-01", step_hours: int = 12,
                 grade: str = "DT_SUMMER", verbose: bool = True) -> BacktestResult:
    tele, flags, lab = load_processed()
    s_online = _online_sulfur(tele, lab)
    lab_prod = lab[(lab.stream == "HT_PRODUCT") & (lab.param == "Mg.Sulfur")
                   & (lab.source == "LIMS")]
    lab_prod = lab_prod[(lab_prod.value > 0.2) & (lab_prod.value < 60)]
    truth = pd.Series(lab_prod["value"].to_numpy(),
                      index=pd.DatetimeIndex(lab_prod["measured_at"])).sort_index()
    truth = truth[~truth.index.duplicated()]

    horizon = pd.Timedelta(minutes=system.config.horizon_min)
    times = pd.date_range(start, end, freq=f"{step_hours}h")
    rows = []
    for i, t in enumerate(times):
        if t not in tele.index and t > tele.index.max():
            break
        res = system.run_at(t, grade=grade, plan_blend=False)
        rec = res.recommendation
        item = None
        if rec.option is not None:
            q = rec.option.predictions["quality"]
            item = next((x for x in q.items if x.param == "Mg.Sulfur"), None)
        # факт через горизонт: лабораторный результат, если он есть в окне
        w0, w1 = t + horizon - pd.Timedelta(hours=3), t + horizon + pd.Timedelta(hours=3)
        lab_win = truth.loc[w0:w1]
        fact_lab = float(lab_win.iloc[0]) if len(lab_win) else np.nan
        fact_online = float(s_online.asof(t + horizon)) if len(s_online) else np.nan
        rows.append({
            "t": t, "decision": rec.decision, "action": rec.action_label,
            "confidence": rec.confidence,
            "forecast": item.forecast if item else np.nan,
            "risk": item.exceed_prob if item else np.nan,
            "risk_process": item.risk_process if item else np.nan,
            "fact_lab": fact_lab, "fact_online": fact_online,
            "violation_lab": (fact_lab > SULFUR_LIMIT) if np.isfinite(fact_lab) else np.nan,
            "violation_online": (fact_online > SULFUR_LIMIT) if np.isfinite(fact_online) else np.nan,
        })
        if verbose and (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(times)}")
    df = pd.DataFrame(rows)

    decisions = df["decision"].value_counts().to_dict()
    calib = _calibration(df)
    summary = {
        "n_cycles": int(len(df)),
        "period": [str(df["t"].min()), str(df["t"].max())],
        "decisions": decisions,
        "share_hold": float((df["decision"].str.startswith("hold")).mean()),
        "share_act": float((df["decision"].str.startswith("act")).mean()),
        "share_refuse": float((df["decision"] == "refuse").mean()),
        "n_with_lab_fact": int(df["violation_lab"].notna().sum()),
    }
    sub = df[df["violation_lab"].notna()]
    if len(sub) >= 20:
        summary["mean_declared_risk"] = float(sub["risk"].mean())
        summary["observed_violation_rate"] = float(sub["violation_lab"].mean())
        try:
            from sklearn.metrics import roc_auc_score
            y = sub["violation_lab"].astype(bool).astype(int).to_numpy()
            p = sub["risk"].astype(float).to_numpy()
            ok = np.isfinite(p)
            if len(set(y[ok].tolist())) > 1:
                summary["roc_auc_risk"] = float(roc_auc_score(y[ok], p[ok]))
        except Exception as exc:
            summary["roc_auc_risk_error"] = str(exc)
    hold = sub[sub["decision"].str.startswith("hold")]
    if len(hold) >= 10:
        summary["violation_rate_when_hold"] = float(hold["violation_lab"].mean())
    rec_mode = sub[sub["decision"] == "act_recovery"]
    if len(rec_mode) >= 10:
        summary["violation_rate_when_recovery"] = float(rec_mode["violation_lab"].mean())
    act = sub[sub["decision"].str.startswith("act")]
    if len(act) >= 10:
        summary["violation_rate_when_act"] = float(act["violation_lab"].mean())
    return BacktestResult(rows=df, decisions=decisions, calibration=calib, summary=summary)


def _calibration(df: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    sub = df[df["violation_lab"].notna() & df["risk"].notna()]
    if len(sub) < 30:
        return pd.DataFrame(columns=["корзина риска", "n", "заявленный риск",
                                     "наблюдаемая доля нарушений"])
    try:
        q = pd.qcut(sub["risk"], bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=["корзина риска", "n", "заявленный риск",
                                     "наблюдаемая доля нарушений"])
    g = sub.groupby(q, observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "заявленный риск": g["risk"].mean(),
        "наблюдаемая доля нарушений": g["violation_lab"].mean(),
    }).reset_index()
    out.columns = ["корзина риска", "n", "заявленный риск", "наблюдаемая доля нарушений"]
    out["корзина риска"] = out["корзина риска"].astype(str)
    return out
