"""Оркестратор: один цикл принятия решения.

Реализует ровно последовательность из раздела 1 ТЗ:

  1. получить состояние процесса          -> AsOfStore.snapshot(t)
  2. проверить полноту/актуальность       -> агент данных
  3. оценить текущее и прогнозное качество -> агенты сырья и качества
  4. оценить тяжесть режима                -> агент надёжности
  5. сформировать варианты                 -> агент оптимизации
  6. отбросить недопустимые                -> он же (жёсткие ограничения)
  7. сравнить оставшиеся                   -> фронт Парето + свёртка
  8. выбрать и объяснить, либо отказаться  -> оркестратор + агент объяснения

Разрешение конфликта целей
--------------------------
Конфликты возникают постоянно: агент качества требует поднять температуру,
агент надёжности это запрещает, агент экономики хочет снизить затраты.
Правило разрешения — ЛЕКСИКОГРАФИЧЕСКОЕ и зафиксировано в ТЗ: «качество и
жёсткие технологические ограничения имеют приоритет над экономическим
эффектом». Поэтому:
  * недопустимые по качеству и надёжности варианты отсеиваются ДО экономики;
  * если допустимых вариантов не осталось — система ОТКАЗЫВАЕТСЯ давать
    рекомендацию и называет причину; это штатный, а не аварийный исход.
"""
from __future__ import annotations

import numpy as np

from ..models.process_model import Action
from .base import Agent, state_hash
from .contracts import Recommendation, Severity


def _fmt(v, nd=1, dash="—"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    return f"{float(v):.{nd}f}"


class OrchestratorAgent(Agent):
    name = "orchestrator"
    role = "Оркестратор — сбор оценок, разрешение конфликта целей, итоговое решение"

    def run_cycle(self, state, grade: str = "DT_SUMMER",
                  horizon_min: int | None = None, plan_blend: bool = True) -> Recommendation:
        bus = self.bus
        cfg = self.ctx.config
        horizon_min = int(horizon_min or cfg.horizon_min)
        quality_agent = self.ctx.extras["agents"]["quality"]
        quality_agent.reset_cache()

        # --------------------------------------------------------- шаг 2
        dq = bus.request(self.name, "data", "assess", state=state)
        if not dq.usable:
            reasons = [i.detail for i in dq.issues if i.severity is not Severity.OK]
            rec = Recommendation(
                t=state.t, decision="refuse",
                headline="Надёжной рекомендации нет: входные данные непригодны",
                action_label="вмешательство не предлагается",
                option=None, confidence=float(dq.score),
                reasons=reasons or ["интегральная пригодность данных ниже порога"],
                state_hash=state_hash(state, {"grade": grade, "horizon": horizon_min}))
            bus.note(self.name, "decision.refuse", {"reason": "данные непригодны",
                                                    "score": round(dq.score, 3)})
            rec.expected_effect = {"действие": "режим не менять, устранить проблему с данными"}
            self._attach_explanation(rec, dq, None, None, None, None, [])
            return rec

        # ------------------------------------------------------- шаги 3-4
        feed = bus.request(self.name, "feed", "assess", state=state)
        baseline = bus.request(self.name, "quality", "assess", state=state, grade=grade,
                               feed=feed, action=Action(), horizon_min=horizon_min)
        reliability = bus.request(self.name, "reliability", "assess", state=state,
                                  action=Action())

        # --------------------------------------------------------- шаг 5-7
        options = bus.request(
            self.name, "optimizer", "optimize", state=state, grade=grade, feed=feed,
            reliability=reliability, horizon_min=horizon_min,
            quality_fn=lambda st, g, f, a, h: quality_agent.on_assess(st, g, f, a, h),
            economics_fn=lambda st, f, a, s, t, l: self.ctx.extras["agents"]["economics"]
            .on_evaluate(st, f, a, s, t, l))
        feasible = [o for o in options if o.metrics.feasible]
        compliant = [o for o in feasible if o.metrics.compliant]
        hold = next((o for o in options if o.action.is_hold()), None)

        if not feasible:
            # Спецификация УЖЕ нарушена (или будет нарушена при любом действии
            # в допустимом коридоре). Отказываться в этой ситуации нельзя:
            # установка работает, продукт идёт, и оператору нужен план вывода
            # в норму. Поэтому система переходит в режим ВОССТАНОВЛЕНИЯ:
            # выбирает вариант, максимально снижающий нарушенный показатель,
            # и прямо сообщает, что норматив сейчас не выполняется.
            # Отказ остаётся там, где ему место: непригодные данные и полный
            # запрет со стороны надёжности.
            corrective = [o for o in options
                          if not (reliability.severity is Severity.ALARM
                                  and o.action.d_temp_c > 0)]
            if not corrective:
                reasons = ["агент надёжности запретил все корректирующие действия: "
                           f"индекс тяжести режима {reliability.severity_index:.2f}"]
                rec = Recommendation(
                    t=state.t, decision="refuse",
                    headline="Допустимого решения нет — требуется решение технолога",
                    action_label="вмешательство в автоматическом режиме не предлагается",
                    option=None, confidence=float(dq.score * baseline.confidence),
                    reasons=reasons,
                    state_hash=state_hash(state, {"grade": grade, "horizon": horizon_min}))
                rec.constraint_table = self._constraints(baseline)
                rec.expected_effect = {"действие": "передать ситуацию технологу; "
                                                   "ограничить нагрузку до выяснения"}
                bus.note(self.name, "decision.refuse", {"reason": "нет допустимых действий"})
                self._attach_explanation(rec, dq, feed, baseline, reliability, None, [])
                return rec

            def _worst_forecast(o):
                q = o.predictions["quality"]
                worst = 0.0
                for it in q.items:
                    if not it.hard or it.margin is None or not np.isfinite(it.margin):
                        continue
                    worst = min(worst, float(it.margin))
                return worst

            best = max(corrective, key=lambda o: (round(_worst_forecast(o), 4),
                                                  o.metrics.margin_rub_h))
            chosen_quality = best.predictions["quality"]
            chosen_econ = best.predictions["economics"]
            chosen_rel = best.predictions["reliability"]
            violated_now = [it.ru_name for it in baseline.items
                            if it.hard and it.margin is not None
                            and np.isfinite(it.margin) and it.margin < 0]
            rec = Recommendation(
                t=state.t, decision="act_recovery",
                headline=("Спецификация нарушена: "
                          + (", ".join(violated_now) if violated_now else "прогноз вне предела")
                          + " — предложен режим вывода в норму"),
                action_label=best.action.label(), option=best,
                confidence=float(dq.score * chosen_quality.confidence),
                state_hash=state_hash(state, {"grade": grade, "horizon": horizon_min}))
            rec.constraint_table = self._constraints(chosen_quality)
            rec.reasons = [
                "ни один вариант в допустимом коридоре не возвращает показатели в норму "
                f"за {horizon_min} мин — выбран вариант максимального приближения к нормативу",
                "продукт этого периода подлежит отдельному решению по назначению "
                "(некондиция/смешение), система об этом не решает",
            ] + self._reasons(best, hold, corrective, reliability, chosen_econ)
            rec.expected_effect = self._effect(baseline, chosen_quality, hold, best,
                                               chosen_econ, chosen_rel)
            t_now_c = state.mean(self.ctx.bundle.meta["tags"]["temp"], 60)
            if not np.isfinite(t_now_c):
                t_now_c = self.ctx.bundle.reference["t_reactor_c"]
            rec.setpoint_plan = build_setpoint_plan(
                self.ctx, state, baseline, float(t_now_c),
                float(self.ctx.bundle.mv_ranges["temp"][1]), feed_report=feed)
            bus.note(self.name, "decision.recovery",
                     {"нарушено": violated_now, "действие": best.action.label()})
            if plan_blend:
                rec.blend = bus.request(self.name, "blending", "plan", grade=grade)
            self._attach_explanation(rec, dq, feed, chosen_quality, chosen_rel, best, [],
                                     baseline=baseline)
            return rec

        # Выбор уровня: сначала полностью соответствующие спецификации,
        # иначе — вариант минимального риска с честным предупреждением.
        best_effort = False
        if compliant:
            pool = compliant
            best = pool[0]
        else:
            # Режим минимального риска. Чистый минимум риска вырождается в угол
            # сетки (максимум температуры, минимум нагрузки), потому что любое
            # снижение нагрузки чуть-чуть снижает серу. Поэтому среди вариантов,
            # практически равных по риску (в пределах 1 п.п.), выбирается лучший
            # по марже: снижать выпуск ради десятых долей процента риска
            # экономически бессмысленно и технологически вредно.
            pool = feasible
            r_min = min(o.metrics.worst_exceed_prob for o in pool)
            near = [o for o in pool if o.metrics.worst_exceed_prob <= r_min + 0.01]
            best = max(near, key=lambda o: (o.metrics.margin_rub_h
                                            if np.isfinite(o.metrics.margin_rub_h) else -1e18,
                                            -o.metrics.severity_index))
            best_effort = True

        # ---------------------------------------- защита от лишних действий
        min_benefit = float(cfg.agents["orchestrator"]["min_action_benefit_rub_h"])
        decision = "act"
        extra_reason = []
        if hold is not None and hold.metrics.feasible and \
                (not best_effort or hold.metrics.compliant == best.metrics.compliant):
            gain = hold.metrics.cost_rub_h - best.metrics.cost_rub_h
            risk_gain = hold.metrics.worst_exceed_prob - best.metrics.worst_exceed_prob
            if not best.action.is_hold() and gain < min_benefit and risk_gain < 0.02:
                best = hold
                decision = "hold"
                extra_reason.append(
                    f"выигрыш лучшего вмешательства ({gain:,.0f} ₽/ч) ниже порога "
                    f"{min_benefit:,.0f} ₽/ч, а риск по качеству почти не снижается — "
                    f"лишнее управляющее действие не оправдано".replace(",", " "))
        if best.action.is_hold():
            decision = "hold"
        if best_effort:
            decision = "act_best_effort" if not best.action.is_hold() else "hold_best_effort"

        chosen_quality = best.predictions["quality"]
        chosen_econ = best.predictions["economics"]
        chosen_rel = best.predictions["reliability"]

        bus.note(self.name, "conflict.resolution", {
            "правило": "качество и жёсткие ограничения > надёжность > экономика > выпуск",
            "допустимых вариантов": len(feasible),
            "на фронте Парето": sum(1 for o in feasible if o.pareto),
            "запрет надёжности": reliability.severity.value,
        })

        alternatives = [o for o in pool if o is not best and o.pareto][:3]
        rec = Recommendation(
            t=state.t, decision=decision,
            headline=self._headline(decision, baseline, chosen_quality),
            action_label=best.action.label(), option=best, alternatives=alternatives,
            confidence=float(dq.score * chosen_quality.confidence),
            state_hash=state_hash(state, {"grade": grade, "horizon": horizon_min}))
        rec.constraint_table = self._constraints(chosen_quality)
        rec.reasons = self._reasons(best, hold, feasible, reliability, chosen_econ) + extra_reason
        if best_effort:
            worst = max((i for i in chosen_quality.items
                         if i.hard and i.exceed_prob is not None), key=lambda i: i.exceed_prob)
            rec.reasons.insert(0,
                f"ни один вариант не обеспечивает риск нарушения ниже 10 %: "
                f"выбран режим МИНИМАЛЬНОГО РИСКА, остаточный риск по показателю "
                f"«{worst.ru_name}» {worst.exceed_prob*100:.0f} %. Основная причина — "
                f"разброс между лабораторией и поточным анализатором, а не режим установки")
        rec.expected_effect = self._effect(baseline, chosen_quality, hold, best, chosen_econ,
                                           chosen_rel)
        t_now_c = state.mean(self.ctx.bundle.meta["tags"]["temp"], 60)
        if not np.isfinite(t_now_c):
            t_now_c = self.ctx.bundle.reference["t_reactor_c"]
        rec.setpoint_plan = build_setpoint_plan(
            self.ctx, state, baseline, float(t_now_c),
            float(self.ctx.bundle.mv_ranges["temp"][1]), feed_report=feed)
        if rec.setpoint_plan is not None:
            bus.note(self.name, "setpoint.plan", rec.setpoint_plan.as_dict())
        if plan_blend:
            rec.blend = bus.request(self.name, "blending", "plan", grade=grade)
        self._attach_explanation(rec, dq, feed, chosen_quality, chosen_rel, best, alternatives,
                                 baseline=baseline)
        return rec

    # ----------------------------------------------------------------- utils
    def _headline(self, decision: str, baseline, chosen) -> str:
        s0 = next((i for i in baseline.items if i.param == "Mg.Sulfur"), None)
        s1 = next((i for i in chosen.items if i.param == "Mg.Sulfur"), None)
        if decision.endswith("best_effort"):
            return (f"Спецификация не гарантирована: выбран режим минимального риска, "
                    f"прогноз серы {_fmt(s1.forecast, 2)} мг/кг "
                    f"(риск {s1.exceed_prob*100:.0f} % против {s0.exceed_prob*100:.0f} % "
                    f"при бездействии)")
        if decision == "hold":
            return (f"Режим устойчив: прогноз серы {_fmt(s1.forecast, 2)} мг/кг при пределе "
                    f"{_fmt(s1.limit, 0)}, риск {s1.exceed_prob*100:.0f} % — вмешательство "
                    f"не требуется")
        return (f"Риск по сере {s0.exceed_prob*100:.0f} % — предлагается коррекция режима, "
                f"прогноз снижается до {_fmt(s1.forecast, 2)} мг/кг "
                f"(риск {s1.exceed_prob*100:.0f} %)")

    def _constraints(self, quality) -> list[dict]:
        rows = []
        for it in quality.items:
            if it.limit_kind == "min":
                verdict = "выполнено" if (it.forecast is not None and np.isfinite(it.forecast)
                                          and it.forecast >= it.limit) else "НЕ ВЫПОЛНЕНО"
                lim = f"≥ {_fmt(it.limit, 1)}"
            elif it.limit_kind == "range":
                verdict = "выполнено" if (it.margin is not None and np.isfinite(it.margin)
                                          and it.margin >= 0) else "НЕ ВЫПОЛНЕНО"
                lim = f"диапазон, верх {_fmt(it.limit, 1)}"
            else:
                verdict = "выполнено" if (it.forecast is not None and np.isfinite(it.forecast)
                                          and it.forecast <= it.limit) else "НЕ ВЫПОЛНЕНО"
                lim = f"≤ {_fmt(it.limit, 1)}"
            if it.forecast is None or not np.isfinite(it.forecast):
                verdict = "нет данных"
            if it.exceed_prob is not None and np.isfinite(it.exceed_prob) and it.hard \
                    and it.exceed_prob > 0.10 and verdict == "выполнено":
                verdict = "риск выше допустимого"
            rows.append({"name": it.ru_name + (" (жёсткое)" if it.hard else " (рекоменд.)"),
                         "forecast": _fmt(it.forecast, 2), "limit": lim,
                         "margin": _fmt(it.margin, 2),
                         "risk": "—" if it.exceed_prob is None or not np.isfinite(it.exceed_prob)
                                 else f"{it.exceed_prob*100:.0f} %",
                         "verdict": verdict})
        return rows

    def _reasons(self, best, hold, feasible, reliability, econ) -> list[str]:
        out = []
        n_par = sum(1 for o in feasible if o.pareto)
        out.append(f"из {len(feasible)} допустимых вариантов {n_par} находятся на фронте "
                   f"Парето; выбран лучший по свёртке с приоритетом качества")
        if hold is not None and not best.action.is_hold():
            d_risk = hold.metrics.worst_exceed_prob - best.metrics.worst_exceed_prob
            d_cost = best.metrics.cost_rub_h - hold.metrics.cost_rub_h
            verb = "снижается" if d_risk >= 0 else "повышается"
            price = (f"при снижении переменных затрат на {abs(d_cost):,.0f} ₽/ч"
                     if d_cost < 0 else f"ценой {d_cost:,.0f} ₽/ч переменных затрат")
            out.append(f"относительно бездействия риск нарушения {verb} на "
                       f"{abs(d_risk)*100:.0f} п.п. {price}".replace(",", " "))
        if reliability.severity is not Severity.OK:
            out.append(f"коридор изменения температуры сужен агентом надёжности "
                       f"(индекс тяжести {reliability.severity_index:.2f})")
        out.append(f"ресурс цикла катализатора при выбранном режиме — "
                   f"{econ['catalyst_days_left']:.0f} сут")
        return out

    def _effect(self, baseline, chosen, hold, best, econ, rel) -> dict:
        s0 = next((i for i in baseline.items if i.param == "Mg.Sulfur"), None)
        s1 = next((i for i in chosen.items if i.param == "Mg.Sulfur"), None)
        det = s1.risk_detail or {}
        eff = {
            "сера через горизонт":
                f"{_fmt(s0.forecast, 2)} → {_fmt(s1.forecast, 2)} мг/кг "
                f"(интервал 90 % [{_fmt(s1.lo, 2)}; {_fmt(s1.hi, 2)}])",
            "риск нарушения по сере (лаб.)":
                f"{s0.exceed_prob*100:.0f} % → {s1.exceed_prob*100:.0f} %"
                if s0.exceed_prob is not None and s1.exceed_prob is not None else "—",
            "в том числе технологический риск":
                f"{s1.risk_process*100:.0f} % (остальное — расхождение лаборатории и "
                f"поточного анализатора, σ={_fmt(det.get('sigma_measurement'), 2)} мг/кг)"
                if s1.risk_process is not None and np.isfinite(s1.risk_process) else "—",
            "выпуск": f"{econ['throughput_t_h']:.1f} т/ч",
            "переменные затраты": f"{econ['cost_rub_h']:,.0f} ₽/ч".replace(",", " "),
            "тяжесть режима": f"индекс {rel.severity_index:.2f} ({rel.severity.value})",
            "ресурс катализатора": f"{econ['catalyst_days_left']:.0f} сут",
        }
        if hold is not None and not best.action.is_hold():
            d_cost = best.metrics.cost_rub_h - hold.metrics.cost_rub_h
            d_risk = (hold.metrics.worst_exceed_prob - best.metrics.worst_exceed_prob) * 100
            d_thr = best.metrics.throughput_t_h - hold.metrics.throughput_t_h
            eff["эффект против бездействия"] = (
                f"риск {'снижается' if d_risk > 0 else 'растёт'} на {abs(d_risk):.0f} п.п.; "
                f"переменные затраты {'ниже' if d_cost < 0 else 'выше'} на "
                f"{abs(d_cost):,.0f} ₽/ч; выпуск {d_thr:+.1f} т/ч".replace(",", " "))
        return eff

    def _attach_explanation(self, rec, dq, feed, quality, reliability, chosen, alternatives,
                            baseline=None):
        if quality is None or feed is None or reliability is None:
            rec.expected_effect.setdefault("пояснение", rec.headline)
            return
        res = self.bus.request(self.name, "explainer", "explain", rec=rec, dq=dq, feed=feed,
                               quality=quality, reliability=reliability, chosen=chosen,
                               alternatives=alternatives, baseline=baseline)
        rec.expected_effect["_markdown"] = res["markdown"]


# ---------------------------------------------------------------------------
def build_setpoint_plan(ctx, state, quality, current_temp_c: float,
                        step_limit_c: float, feed_report=None) -> "SetpointPlan | None":
    """Сколько градусов и за сколько шагов нужно, чтобы уйти от риска.

    Отвечает на вопрос, который оператор задаёт первым: «а что нужно,
    чтобы было спокойно?». Целевой уровень серы берётся из конформной
    калибровки — это тот уровень, при котором вероятность лабораторного
    нарушения опускается до 10 %. Требуемая температура считается обратной
    задачей кинетики ГДС. Если одного шага рекомендации не хватает,
    выдаётся многошаговый план — как это и делает технолог.
    """
    from .contracts import SetpointPlan
    from ..models.hds import HDSParams, required_temperature

    s_item = next((i for i in quality.items if i.param == "Mg.Sulfur"), None)
    if s_item is None or not s_item.risk_detail:
        return None
    target = s_item.risk_detail.get("target_for_10pct")
    if target is None or not np.isfinite(target) or target <= 0:
        return None
    ref = ctx.bundle.reference
    hds = HDSParams(**ctx.bundle.hds)
    feed_lr = state.lab_value("HT_FEED", "Mass.Sulfur", source="LIMS") \
        or state.lab_value("HT_FEED", "Mass.Sulfur")
    s_feed = feed_lr.value if feed_lr is not None else ref["s_feed_wt"]
    s_feed_prev = float(getattr(feed_report, "sulfur_prev_wt", s_feed)) \
        if feed_report is not None else s_feed
    load = state.mean(ctx.bundle.meta["tags"]["load"], 60)
    if not np.isfinite(load) or load <= 0:
        load = ref["load"]
    lhsv = ref["lhsv"] * load / max(ref["load"], 1e-9)
    press = state.mean(ctx.bundle.meta["tags"]["press"], 60) * ref["scale_press"]
    if not np.isfinite(press) or press <= 0:
        press = ref["p_mpa"]

    # температура, эквивалентная ТЕКУЩЕМУ уровню серы по кинетике, и целевая
    # Опорная температура считается по сырью, которое УЖЕ отработало в
    # реакторе: тогда модельное смещение сокращается, а разница сырья
    # превращается в требуемую компенсацию по температуре.
    t_equiv = required_temperature(max(s_item.value, 0.3), s_feed_prev * 10_000.0,
                                   lhsv, press, hds)
    t_target = required_temperature(max(target, 0.3), s_feed * 10_000.0, lhsv, press, hds)
    delta = float(t_target - t_equiv)
    n_steps = int(np.ceil(abs(delta) / max(step_limit_c, 1e-6))) if step_limit_c > 0 else 0
    step_min = int(ctx.config.step_min)
    steps = []
    acc = 0.0
    for i in range(min(n_steps, 8)):
        d = float(np.sign(delta) * min(step_limit_c, abs(delta) - abs(acc)))
        acc += d
        steps.append({"шаг": i + 1, "через, мин": step_min * i,
                      "Δt, °C": round(d, 2),
                      "накопленное Δt, °C": round(acc, 2)})
    reach = abs(current_temp_c + delta) <= ref["temp_p95"] + 2.0
    note = ""
    if delta <= 0.05:
        note = "текущий режим уже обеспечивает целевой уровень — повышение не требуется"
    elif n_steps > 1:
        note = (f"требуется {n_steps} шага(ов) по {step_min} мин: одношаговое изменение "
                f"ограничено {step_limit_c:.1f} °C (P95 фактической скорости изменения "
                f"за 30 мин по истории)")
    if not reach:
        note += ("; целевая температура выходит за верхнюю границу наблюдавшегося "
                 "режима — требуется решение технолога или изменение нагрузки")
    return SetpointPlan(target_sulfur=float(target), required_temp_c=float(current_temp_c + delta),
                        current_temp_c=float(current_temp_c), delta_total_c=delta,
                        steps=steps, reachable=bool(reach), note=note)
