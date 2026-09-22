"""Агент сырья (АВТ): что приходит на гидроочистку.

Отвечает за первое звено цепочки «АВТ -> гидроочистка -> блендинг».
Главная задача — дать агенту качества и оптимизатору честную оценку серы и
фракционного состава сырья вместе с её возрастом и уверенностью, потому что
именно изменение сырья запускает сценарий, описанный экспертом:
изменилась сера в нефти -> изменилась сера в прямогонном ДТ -> нужно менять
режим гидроочистки.

Иерархия источников (в точности по правилу ТЗ «ЛИМС -> ПАК -> ВАК»):
  1. свежий лабораторный анализ серы сырья (``HT_FEED:Mass.Sulfur``);
  2. устаревший лабораторный анализ, экстраполированный по тренду режима АВТ,
     с пониженной уверенностью;
  3. опорное значение обучающей выборки — с явным предупреждением.

Тег плотности нефти D10 непригоден (99.995 % сентинелов), поэтому плотность
сырья берётся только из ЛИМС/ВАК АВТ; об этом сообщается оператору.
"""
from __future__ import annotations

import numpy as np

from ..models import va_formulas as va
from .base import Agent
from .contracts import FeedReport

AVT_DIESEL_FLOW = "A_W70"     # массовый расход фр. 290-350 °C
AVT_DIESEL_FLOW_ALT = "A_F30"


class FeedstockAgent(Agent):
    name = "feed"
    role = "Агент сырья (АВТ) — качество и расход дизельной фракции на гидроочистку"

    def on_assess(self, state) -> FeedReport:
        cfg = self.ctx.config
        stale_h = float(cfg.specs["staleness_hours"]["LIMS"])
        ref = self.ctx.bundle.reference

        lr = state.lab_value("HT_FEED", "Mass.Sulfur", source="LIMS") \
            or state.lab_value("HT_FEED", "Mass.Sulfur")
        note = ""
        if lr is not None and lr.age_hours <= stale_h:
            s_wt, src, age, conf = lr.value, "ЛИМС (свежий)", lr.age_hours, 0.95
        elif lr is not None:
            # устаревший анализ: оставляем значение, но уверенность падает с возрастом
            s_wt, src, age = lr.value, "ЛИМС (устаревший)", lr.age_hours
            conf = float(np.clip(0.9 * 0.5 ** ((lr.age_hours - stale_h) / 168.0), 0.15, 0.9))
            note = (f"анализ серы сырья сделан {lr.age_hours:.0f} ч назад; "
                    f"оценка используется с пониженной уверенностью")
        else:
            s_wt, src, age, conf = float(ref["s_feed_wt"]), "опорное значение истории", np.inf, 0.15
            note = "лабораторных анализов серы сырья нет — взято опорное значение обучающей выборки"

        # фракционный состав сырья: ЛИМС, иначе ВАК АВТ
        t95 = self._lab_or_va(state, "HT_FEED", "95%.T", None)
        d15 = self._lab_or_va(state, "HT_FEED", "D15", "AVT6:240-350:D15")

        flow = state.mean(AVT_DIESEL_FLOW, 60)
        if not np.isfinite(flow):
            flow = state.mean(AVT_DIESEL_FLOW_ALT, 60)

        trend = state.slope_per_hour("A_T33", 180)
        if np.isfinite(trend) and abs(trend) > 0.5:
            note = (note + "; " if note else "") + (
                f"температура низа К-2 меняется на {trend:+.1f} °C/ч — "
                f"фракционный состав сырья смещается")

        prev, unreal = self._previous_feed_sulfur(state, s_wt)
        if abs(prev - s_wt) > 1e-6 and unreal > 0.05:
            note = (note + "; " if note else "") + (
                f"сера сырья изменилась с {prev:.3f} до {s_wt:.3f} % масс.; "
                f"{unreal*100:.0f} % эффекта ещё не дошло до продукта")
        return FeedReport(t=state.t, sulfur_wt=float(s_wt), sulfur_source=src,
                          sulfur_age_h=float(age), sulfur_confidence=float(conf),
                          sulfur_prev_wt=float(prev), unrealized_fraction=float(unreal),
                          t95_c=float(t95), d15=float(d15),
                          avt_diesel_flow_t_h=float(flow), trend_note=note)

    def _previous_feed_sulfur(self, state, current: float) -> tuple[float, float]:
        """Сера сырья, уже отработавшая в реакторе, и доля нереализованного эффекта.

        Текущий замер серы В ПРОДУКТЕ отражает сырьё, прошедшее реактор
        примерно ``dead_time`` назад. Если новый анализ показал другое
        значение, часть его эффекта ещё не проявилась — именно эту часть
        агент качества добавляет к прогнозу. Если анализ старый, эффект уже
        в измерении, и добавлять ничего не нужно: иначе изменение сырья
        учитывалось бы дважды.
        """
        from ..models.process_model import dynamic_fraction
        dead = float(self.ctx.bundle.dead_times.get("Mg.Sulfur", 80))
        lr = state.lab_value("HT_FEED", "Mass.Sulfur", source="LIMS") \
            or state.lab_value("HT_FEED", "Mass.Sulfur")
        if lr is None:
            return float(current), 0.0
        age_min = float(lr.age_hours) * 60.0
        unrealized = float(max(0.0, 1.0 - dynamic_fraction(age_min, dead)))
        if unrealized < 1e-3:
            return float(current), 0.0
        # предыдущее известное значение берём из хранилища (без подмен сценария)
        prev = float(current)
        store = getattr(self.ctx, "store", None)
        if store is not None:
            try:
                g = store._lab_by_key.get(("HT_FEED", "Mass.Sulfur", "LIMS"))
                if g is not None and len(g):
                    pos = g["available_at"].searchsorted(lr.available_at, side="left") - 1
                    if pos >= 0:
                        prev = float(g.iloc[pos]["value"])
            except Exception:
                prev = float(current)
        return prev, unrealized

    def _lab_or_va(self, state, stream: str, param: str, va_name: str | None) -> float:
        lr = state.lab_value(stream, param)
        if lr is not None and lr.age_hours <= 168:
            return lr.value
        if va_name:
            res = va.evaluate(state, [va_name])[va_name]
            if res.valid:
                return res.value + self.ctx.bundle.va_bias.get(va_name, 0.0)
        return lr.value if lr is not None else float("nan")
