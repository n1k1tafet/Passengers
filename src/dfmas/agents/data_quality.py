"""Агент данных: полнота, актуальность, согласованность.

Второй шаг цикла принятия решения из ТЗ. Агент не «чистит» данные — чистка
уже выполнена на этапе ETL. Здесь оценивается, МОЖНО ЛИ ВООБЩЕ принимать
решение по этому срезу, и формулируется причина отказа, понятная оператору.

Проверки
--------
1. Полнота: доступны ли ключевые теги режима и хотя бы один источник серы.
2. Свежесть: возраст телеметрии и анализов против порогов из config/specs.yaml.
3. Согласованность: два НЕЗАВИСИМЫХ источника серы (поточный анализатор Q21 и
   выгрузка ПАК) должны совпадать. Расхождение больше допуска означает
   неисправность прибора — и это единственный способ её заметить без выезда
   на объект.
4. Аномальность режима: расстояние Махаланобиса текущего вектора режима до
   облака нормальной работы. Ловит сочетания параметров, каждый из которых
   по отдельности в норме.
5. Залипание: доля значений с флагом flatline в окне.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.fit import (TAG_DP, TAG_H2, TAG_LOAD, TAG_PRESS, TAG_S_ONLINE,
                          TAG_TEMP, TAG_TEMP_SET)
from .base import Agent
from .contracts import DataIssue, DataQualityReport, Severity

KEY_TAGS = [TAG_TEMP, TAG_LOAD, TAG_PRESS, TAG_H2, TAG_S_ONLINE, TAG_DP]


class DataQualityAgent(Agent):
    name = "data"
    role = "Агент данных — полнота, актуальность и согласованность источников"

    def on_assess(self, state) -> DataQualityReport:
        cfg = self.ctx.config
        dq = cfg.agents["data_quality"]
        stale = cfg.specs["staleness_hours"]
        issues: list[DataIssue] = []
        penalties = 0.0

        # ---------------------------------------------------------- полнота
        missing = [t for t in KEY_TAGS if not np.isfinite(state.get(t))]
        if missing:
            issues.append(DataIssue("MISSING_TAGS",
                                    "нет валидных значений ключевых тегов режима",
                                    Severity.ALARM if TAG_S_ONLINE in missing else Severity.WARN,
                                    missing))
            penalties += 0.45 if TAG_S_ONLINE in missing else 0.15

        bad_flags = [t for t in KEY_TAGS
                     if state.tag_flag.get(t) in ("sentinel", "digital_state", "out_of_range")]
        if bad_flags:
            issues.append(DataIssue("BAD_FLAG",
                                    "последний отсчёт помечен историком как непригодный",
                                    Severity.WARN, bad_flags))
            penalties += 0.10 * len(bad_flags)

        flat = [t for t in KEY_TAGS if state.tag_flag.get(t) == "flatline"]
        if flat:
            issues.append(DataIssue("FLATLINE", "значение не меняется — подозрение на залипание прибора",
                                    Severity.WARN, flat))
            penalties += 0.10 * len(flat)

        # --------------------------------------------------------- свежесть
        freshness = {}
        tele_age = max((state.tag_age_min.get(t, np.inf) for t in KEY_TAGS
                        if np.isfinite(state.get(t))), default=np.inf)
        freshness["TELEMETRY"] = tele_age / 60.0
        if tele_age / 60.0 > stale["TELEMETRY"]:
            issues.append(DataIssue("STALE_TELEMETRY",
                                    f"телеметрия старше {stale['TELEMETRY']} ч "
                                    f"(возраст {tele_age/60:.1f} ч)", Severity.ALARM, []))
            penalties += 0.40

        pak = state.lab_value("HT_PRODUCT", "Mg.Sulfur", source="PAK")
        if pak is not None:
            freshness["PAK"] = pak.age_hours
            if pak.age_hours > stale["PAK"]:
                issues.append(DataIssue("STALE_PAK",
                                        f"поточный анализ серы устарел ({pak.age_hours:.1f} ч)",
                                        Severity.WARN, []))
                penalties += 0.10
        feed = state.lab_value("HT_FEED", "Mass.Sulfur", source="LIMS")
        if feed is None:
            issues.append(DataIssue("NO_FEED_SULFUR",
                                    "нет лабораторного анализа серы сырья гидроочистки",
                                    Severity.WARN, []))
            penalties += 0.15
        else:
            freshness["LIMS_feed_sulfur"] = feed.age_hours
            if feed.age_hours > stale["LIMS"]:
                issues.append(DataIssue("STALE_FEED_SULFUR",
                                        f"анализ серы сырья устарел: {feed.age_hours:.0f} ч "
                                        f"при допустимых {stale['LIMS']:.0f} ч",
                                        Severity.WARN, []))
                penalties += 0.20
        prod = state.lab_value("HT_PRODUCT", "Mg.Sulfur", source="LIMS")
        if prod is not None:
            freshness["LIMS_product_sulfur"] = prod.age_hours

        # ---------------------------------------------------- согласованность
        cross = {}
        q21 = state.mean(TAG_S_ONLINE, 60)
        pak_v = pak.value if pak is not None else np.nan
        if np.isfinite(q21) and np.isfinite(pak_v):
            diff = abs(q21 - pak_v)
            cross = {"q21": q21, "pak": pak_v, "abs_diff": diff}
            tol = float(dq["cross_check"]["tolerance_mg_kg"])
            if diff > tol:
                issues.append(DataIssue(
                    "SOURCE_MISMATCH",
                    f"два независимых анализатора серы расходятся на {diff:.1f} мг/кг "
                    f"(допуск {tol}): Q21={q21:.1f}, ПАК={pak_v:.1f}",
                    Severity.ALARM, [TAG_S_ONLINE]))
                penalties += 0.30

        # --------------------------------------------- аномальность режима
        maha = self._mahalanobis(state)
        if np.isfinite(maha):
            cross["regime_mahalanobis"] = maha
            if maha > 4.0:
                issues.append(DataIssue(
                    "REGIME_ANOMALY",
                    f"сочетание параметров режима нетипично (расстояние {maha:.1f} σ) — "
                    f"модели работают вне области обучения",
                    Severity.WARN, []))
                penalties += 0.15

        # ---------------------------------------------- доля валидных точек
        win = state.window[[c for c in KEY_TAGS if c in state.window.columns]]
        valid_share = float(win.notna().mean().mean()) if not win.empty else 0.0
        if valid_share < float(dq["min_valid_share"]):
            issues.append(DataIssue("SPARSE_WINDOW",
                                    f"в окне только {valid_share*100:.0f} % валидных значений",
                                    Severity.WARN, []))
            penalties += 0.15

        score = float(np.clip(1.0 - penalties, 0.0, 1.0))
        usable = score >= 0.45 and not any(
            i.severity is Severity.ALARM and i.code in
            ("MISSING_TAGS", "STALE_TELEMETRY", "SOURCE_MISMATCH") for i in issues)
        return DataQualityReport(t=state.t, usable=usable, score=score, issues=issues,
                                 freshness=freshness, cross_check=cross)

    # ------------------------------------------------------------------ util
    def _mahalanobis(self, state) -> float:
        ref = self.ctx.bundle.reference
        stats = self.ctx.extras.get("regime_stats")
        if stats is None:
            return float("nan")
        mu, sd = stats["mu"], stats["sd"]
        x = np.array([state.mean(t, 60) for t in stats["tags"]], dtype=float)
        if not np.all(np.isfinite(x)):
            return float("nan")
        z = (x - mu) / np.where(sd > 0, sd, 1.0)
        return float(np.sqrt(float(z @ np.linalg.solve(stats["corr"], z)) / len(z)))
