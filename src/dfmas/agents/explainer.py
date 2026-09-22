"""Агент объяснения: текст рекомендации для оператора.

Текст собирается ДЕТЕРМИНИРОВАННО из шаблонов по числовым результатам других
агентов. Ни одного генеративного элемента — это прямой ответ на требование
воспроизводимости: «при одинаковом состоянии системы должны воспроизводиться
значимые результаты... главное одно и то же решение и одинаковые численные
результаты». Здесь совпадает не только решение и числа, но и сам текст.

Структура ответа повторяет таблицу из раздела 5 ТЗ:
время и состояние -> проблема -> действие -> ожидаемый эффект ->
проверка ограничений -> уверенность -> объяснение.
"""
from __future__ import annotations

import numpy as np

from .base import Agent


def _fmt(v, nd=1, dash="—"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    return f"{float(v):.{nd}f}"


class ExplainerAgent(Agent):
    name = "explainer"
    role = "Агент объяснения — формулировка рекомендации на языке оператора"

    def on_explain(self, rec, dq, feed, quality, reliability, chosen, alternatives,
                   baseline=None) -> dict:
        """``quality`` — оценка ВЫБРАННОГО варианта, ``baseline`` — бездействия."""
        baseline = baseline if baseline is not None else quality
        lines: list[str] = []
        t = rec.t.strftime("%Y-%m-%d %H:%M")

        lines.append(f"### Время и состояние")
        lines.append(f"Срез **{t}**. Пригодность данных **{dq.score:.2f}**"
                     f"{' — данных достаточно' if dq.usable else ' — данных недостаточно'}.")
        fresh = ", ".join(f"{k}: {_fmt(v, 1)} ч" for k, v in sorted(dq.freshness.items()))
        if fresh:
            lines.append(f"Свежесть источников — {fresh}.")
        lines.append(f"Сырьё гидроочистки: сера **{_fmt(feed.sulfur_wt, 3)} % масс.** "
                     f"({feed.sulfur_source}, возраст {_fmt(feed.sulfur_age_h, 0)} ч), "
                     f"Т95 {_fmt(feed.t95_c)} °C, расход {_fmt(feed.avt_diesel_flow_t_h)} т/ч.")

        lines.append("")
        lines.append("### Проблема / риск")
        s = next((i for i in baseline.items if i.param == "Mg.Sulfur"), None)
        if s is not None:
            lines.append(f"Сера продукта сейчас **{_fmt(s.value, 2)} мг/кг** ({s.source}); "
                         f"прогноз через {baseline.horizon_min} мин **без вмешательства** — "
                         f"**{_fmt(s.forecast, 2)}** мг/кг, интервал 90 % "
                         f"[{_fmt(s.lo, 2)}; {_fmt(s.hi, 2)}].")
            if s.exceed_prob is not None and np.isfinite(s.exceed_prob):
                lines.append(f"Вероятность выхода за предел {_fmt(s.limit, 0)} мг/кг "
                             f"при бездействии — **{s.exceed_prob*100:.0f} %**.")
            det = s.risk_detail or {}
            tgt = det.get("target_for_10pct")
            if tgt is not None and np.isfinite(tgt):
                lines.append(f"Чтобы риск опустился до 10 %, уровень серы должен быть "
                             f"не выше **{tgt:.2f} мг/кг** "
                             f"(предел 10 минус 90-й процентиль разброса измерения).")
        if reliability.severity.value != "ok":
            notes = "; ".join(n.rstrip(".") for n in reliability.notes)
            lines.append(f"Тяжесть режима — индекс {reliability.severity_index:.2f} "
                         f"({reliability.severity.value})."
                         + (f" {notes}." if notes else ""))
        for issue in dq.issues:
            lines.append(f"- Данные: {issue.detail}.")

        lines.append("")
        lines.append("### Предлагаемое действие")
        lines.append(f"**{rec.action_label}**")
        if chosen is not None and rec.decision == "act":
            for tag_line in self._tag_lines(chosen):
                lines.append(f"- {tag_line}")

        lines.append("")
        lines.append("### Ожидаемый эффект")
        for key, val in rec.expected_effect.items():
            lines.append(f"- {key}: {val}")

        lines.append("")
        lines.append("### Проверка ограничений")
        lines.append("| Показатель | Прогноз | Предел | Запас | Риск | Вердикт |")
        lines.append("|---|---|---|---|---|---|")
        for row in rec.constraint_table:
            lines.append(f"| {row['name']} | {row['forecast']} | {row['limit']} | "
                         f"{row['margin']} | {row['risk']} | {row['verdict']} |")

        lines.append("")
        lines.append("### Уверенность")
        lines.append(f"Итоговая уверенность **{rec.confidence:.2f}** "
                     f"(качество данных {dq.score:.2f} x модельная уверенность "
                     f"{quality.confidence:.2f}).")
        for n in quality.notes:
            lines.append(f"- {n}")

        lines.append("")
        lines.append("### Почему выбран этот вариант")
        for r in rec.reasons:
            lines.append(f"- {r}")
        if alternatives:
            lines.append("")
            lines.append("Допустимые альтернативы:")
            for alt in alternatives[:3]:
                m = alt.metrics
                lines.append(f"- {alt.action.label()} — риск по сере "
                             f"{m.worst_exceed_prob*100:.0f} %, затраты "
                             f"{m.cost_rub_h:,.0f} ₽/ч, выпуск {m.throughput_t_h:.1f} т/ч, "
                             f"тяжесть {m.severity_index:.2f}"
                             .replace(",", " "))
        return {"markdown": "\n".join(lines)}

    def _tag_lines(self, option) -> list[str]:
        """Перевод действия в конкретные теги и уставки."""
        ref = self.ctx.bundle.reference
        tags = self.ctx.bundle.meta["tags"]
        a = option.action
        out = []
        if abs(a.d_temp_c) >= 0.05:
            p8 = ref.get("p8_units_per_degC")
            extra = ""
            if p8 is not None and np.isfinite(p8):
                extra = (f"; эквивалент по уставке {tags['temp_set']} "
                         f"(Р-202, температура ГСС на входе): {a.d_temp_c * p8:+.4f} ед.")
            out.append(f"температура реактора ({tags['temp']} — ГСС на выходе Р-201): "
                       f"{a.d_temp_c:+.1f} °C{extra}")
        if abs(a.d_load_pct) >= 0.05:
            out.append(f"загрузка по сырью ({tags['load']}): {a.d_load_pct:+.1f} %")
        if abs(a.d_press_mpa) >= 0.005:
            out.append(f"давление на входе Р-202 ({tags['press']}): {a.d_press_mpa:+.2f} МПа")
        if abs(a.d_h2_pct) >= 0.5:
            out.append(f"расход свежего ВСГ ({tags['h2']}): {a.d_h2_pct:+.1f} %")
        return out
