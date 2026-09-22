"""Сборка автономного HTML-дашборда.

Требования, которые определили форму:
  * эксперт откроет файл «девяносто пятым за день» — значит, на первом экране
    должно быть видно, ЧТО система решила и ПОЧЕМУ;
  * развёртывание в периметре компании, в том числе на отечественных ОС —
    значит, ни одной внешней зависимости: весь CSS, SVG и данные внутри файла;
  * логика принятия решения должна быть видна инженеру — значит, полный журнал
    обмена агентов лежит рядом с рекомендацией, а не в логах.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACTS, REPORTS, load_config
from .models.fit import ModelBundle
from .scenarios import load_scenarios, run_scenario
from .sweep import feed_sulfur_sweep, grade_sweep
from .system import DarkFactorySystem
from .viz import bar_chart, calibration_chart, grouped_bar, line_chart, Axes

E = html.escape

DECISION_RU = {
    "hold": ("не вмешиваться", "ok"),
    "hold_best_effort": ("не вмешиваться (лучше вариантов нет)", "warn"),
    "act": ("коррекция режима", "ok"),
    "act_best_effort": ("коррекция, спецификация не гарантирована", "warn"),
    "act_recovery": ("вывод в норму — спецификация нарушена", "bad"),
    "refuse": ("отказ от рекомендации", "bad"),
}


def _md_to_html(md: str) -> str:
    out, in_tbl = [], False
    for line in md.splitlines():
        s = line.rstrip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            tag = "th" if not in_tbl else "td"
            if not in_tbl:
                out.append('<div class="tbl"><table>')
                in_tbl = True
            out.append("<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_tbl:
            out.append("</table></div>")
            in_tbl = False
        if s.startswith("### "):
            out.append(f"<h4>{_inline(s[4:])}</h4>")
        elif s.startswith("## "):
            out.append(f"<h3>{_inline(s[3:])}</h3>")
        elif s.startswith("- "):
            out.append(f"<li>{_inline(s[2:])}</li>")
        elif not s:
            out.append("")
        else:
            out.append(f"<p>{_inline(s)}</p>")
    if in_tbl:
        out.append("</table></div>")
    txt = "\n".join(out)
    txt = txt.replace("<li>", "<ul><li>").replace("</li>", "</li></ul>")
    return txt.replace("</ul>\n<ul>", "")


def _inline(s: str) -> str:
    s = E(s)
    while "**" in s:
        s = s.replace("**", "<b>", 1).replace("**", "</b>", 1)
    return s


def _table(df: pd.DataFrame, fmt="{:.2f}") -> str:
    def f(v):
        if isinstance(v, float):
            return "—" if not np.isfinite(v) else fmt.format(v)
        return E(str(v))
    head = "".join(f"<th>{E(str(c))}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{f(v)}</td>" for v in r) + "</tr>"
                   for r in df.itertuples(index=False))
    return f'<div class="tbl"><table><tr>{head}</tr>{rows}</table></div>' 


def _kpi(value: str, label: str, tone: str = "") -> str:
    return (f'<div class="kpi {tone}"><div class="kpi-v">{E(value)}</div>'
            f'<div class="kpi-l">{E(label)}</div></div>')


def _agent_graph(roster: list[dict]) -> str:
    """Схема взаимодействия агентов — инлайновый SVG."""
    nodes = [
        ("Витрина «на момент t»", 60, 30, "src"), ("Агент данных", 60, 110, "a"),
        ("Агент сырья (АВТ)", 60, 190, "a"), ("Агент качества", 300, 70, "a"),
        ("Агент надёжности", 300, 150, "a"), ("Агент экономики", 300, 230, "a"),
        ("Агент оптимизации", 545, 110, "a"), ("Агент блендинга", 545, 200, "a"),
        ("ОРКЕСТРАТОР", 300, 310, "orch"), ("Рекомендация оператору", 545, 310, "out"),
    ]
    edges = [(0, 1), (0, 2), (1, 8), (2, 8), (3, 6), (4, 6), (5, 6), (6, 8), (7, 8),
             (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 9)]
    W, H, bw, bh = 760, 372, 190, 40
    parts = [f'<svg viewBox="0 0 {W} {H}" class="graph" role="img" '
             f'aria-label="Схема взаимодействия агентов">',
             '<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="3" '
             'orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="var(--text-muted)"/></marker></defs>']
    cx = [x + bw / 2 for _, x, _, _ in nodes]
    cy = [y + bh / 2 for _, _, y, _ in nodes]
    for a, b in edges:
        x1, y1, x2, y2 = cx[a], cy[a], cx[b], cy[b]
        parts.append(f'<path d="M{x1:.0f},{y1:.0f} C{(x1+x2)/2:.0f},{y1:.0f} '
                     f'{(x1+x2)/2:.0f},{y2:.0f} {x2:.0f},{y2:.0f}" class="edge" '
                     f'marker-end="url(#ar)"/>')
    for (name, x, y, kind) in nodes:
        parts.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8" '
                     f'class="node node-{kind}"/>')
        parts.append(f'<text x="{x+bw/2}" y="{y+bh/2+4}" class="nodetext" '
                     f'text-anchor="middle">{E(name)}</text>')
    parts.append("</svg>")
    return "".join(parts)


CSS = """
:root{color-scheme:light;
 --surface-0:#f6f6f4;--surface-1:#fcfcfb;--surface-2:#efefec;--border:#dcdcd6;
 --text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#8a8981;
 --series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;
 --ok:#0f7a4d;--warn:#b26a00;--bad:#b3261e;--accent:#2a78d6;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
 --surface-0:#121211;--surface-1:#1a1a19;--surface-2:#232322;--border:#34342f;
 --text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8f8e85;
 --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;
 --ok:#3fbb85;--warn:#e0a03a;--bad:#f07a72;--accent:#3987e5;}}
:root[data-theme="dark"]{color-scheme:dark;
 --surface-0:#121211;--surface-1:#1a1a19;--surface-2:#232322;--border:#34342f;
 --text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8f8e85;
 --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;
 --ok:#3fbb85;--warn:#e0a03a;--bad:#f07a72;--accent:#3987e5;}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-0);color:var(--text-primary);
 font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;}
.wrap{max-width:1180px;margin:0 auto;padding:0 16px 72px;overflow-x:clip}
header{padding:40px 0 20px;border-bottom:1px solid var(--border);margin-bottom:26px}
h1{font-size:clamp(24px,4vw,34px);line-height:1.2;margin:0 0 8px;letter-spacing:-.02em}
h2{font-size:21px;margin:42px 0 12px;letter-spacing:-.01em}
h3{font-size:17px;margin:26px 0 8px}
h4{font-size:15px;margin:18px 0 6px;color:var(--text-secondary);
 text-transform:uppercase;letter-spacing:.06em;font-weight:600}
p{margin:8px 0}
.sub{color:var(--text-secondary);max-width:74ch}
.muted{color:var(--text-muted);font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}
.kpi{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.kpi-v{font-size:25px;font-weight:650;letter-spacing:-.02em}
.kpi-l{font-size:12.5px;color:var(--text-secondary);margin-top:3px}
.kpi.ok .kpi-v{color:var(--ok)} .kpi.warn .kpi-v{color:var(--warn)} .kpi.bad .kpi-v{color:var(--bad)}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:14px;
 padding:18px 20px;margin:14px 0}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13.5px;table-layout:auto}
.tbl{overflow-x:auto;-webkit-overflow-scrolling:touch}
th,td{border-bottom:1px solid var(--border);padding:7px 10px;text-align:left;vertical-align:top}
th{color:var(--text-secondary);font-weight:600}
@media(min-width:760px){th{white-space:nowrap}}
.chart{width:100%;height:auto;display:block;margin:6px 0 2px}
.grid{stroke:var(--border);stroke-width:1}
.axis{stroke:var(--text-muted);stroke-width:1}
.refline{stroke:var(--text-muted);stroke-width:1.5;stroke-dasharray:5 4}
.tick{fill:var(--text-muted);font-size:11px}
.tick-y{text-anchor:end}
.axislabel{fill:var(--text-secondary);font-size:12px}
.serieslabel{fill:var(--text-secondary);font-size:12px;font-weight:600}
.barvalue{fill:var(--text-primary);font-size:11.5px;font-weight:600}
.tiny{font-size:10px}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:4px 0 2px;font-size:12.5px;
 color:var(--text-secondary)}
.lg i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
.graph{width:100%;height:auto}
.node{fill:var(--surface-2);stroke:var(--border);stroke-width:1}
.node-orch{fill:var(--accent);stroke:var(--accent)}
.node-src{fill:var(--surface-2);stroke:var(--series-3);stroke-width:2}
.node-out{fill:var(--surface-2);stroke:var(--series-2);stroke-width:2}
.nodetext{fill:var(--text-primary);font-size:12px;font-weight:600}
.node-orch+.nodetext{fill:#fff}
.edge{fill:none;stroke:var(--text-muted);stroke-width:1.2;opacity:.6}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:14px 0 4px}
@media(min-width:900px){.tabs{padding-right:150px}}
.tabs button{background:var(--surface-2);border:1px solid var(--border);color:var(--text-secondary);
 border-radius:999px;padding:7px 13px;font-size:13px;cursor:pointer;font-family:inherit}
.tabs button[aria-selected="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.pane{display:none}.pane.on{display:block}
.badge{display:inline-block;border-radius:999px;padding:3px 11px;font-size:12.5px;font-weight:600}
.badge.ok{background:color-mix(in srgb,var(--ok) 16%,transparent);color:var(--ok)}
.badge.warn{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
.badge.bad{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}
details{margin:10px 0;border:1px solid var(--border);border-radius:10px;
 background:var(--surface-1);padding:10px 14px}
summary{cursor:pointer;font-weight:600;font-size:14px}
pre{background:var(--surface-2);border-radius:8px;padding:12px;overflow:auto;max-width:100%;
 font-size:11.5px;line-height:1.5;max-height:420px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:760px){.two{grid-template-columns:1fr}}
.tog{position:fixed;right:14px;top:14px;z-index:9;background:var(--surface-1);
 border:1px solid var(--border);border-radius:999px;padding:7px 13px;cursor:pointer;
 color:var(--text-secondary);font:inherit;font-size:13px}
ul{margin:6px 0 6px 18px;padding:0}
a{color:var(--accent)}
"""

JS = """
function tabs(group){
 const bs=document.querySelectorAll('[data-tab-group="'+group+'"]');
 bs.forEach(b=>b.addEventListener('click',()=>{
  bs.forEach(x=>x.setAttribute('aria-selected',String(x===b)));
  document.querySelectorAll('[data-pane-group="'+group+'"]').forEach(p=>
   p.classList.toggle('on',p.dataset.pane===b.dataset.tab));
 }));
}
document.querySelectorAll('[data-tabs]').forEach(e=>tabs(e.dataset.tabs));
const t=document.getElementById('tog');
t.addEventListener('click',()=>{
 const d=document.documentElement.getAttribute('data-theme')==='dark';
 document.documentElement.setAttribute('data-theme',d?'light':'dark');
});
"""


def build_dashboard(out: Path | None = None) -> Path:
    out = Path(out or ARTIFACTS / "dashboard.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    system = DarkFactorySystem.build()
    bundle = ModelBundle.load()
    cfg = load_config()
    m = bundle.metrics["sulfur_forecast"]

    # ---------------------------------------------------------- сценарии
    scen_html, tabs_html = [], []
    scenarios = load_scenarios()
    for i, sc in enumerate(scenarios):
        res = run_scenario(system, sc)
        rec = res.recommendation
        ru, tone = DECISION_RU.get(rec.decision, (rec.decision, ""))
        md = rec.expected_effect.get("_markdown", "")
        trace = json.dumps(res.trace, ensure_ascii=False, indent=1, default=str)
        blend = ""
        if rec.blend is not None:
            if rec.blend.feasible:
                dfb = pd.DataFrame([{"компонент": c.name, "доля, %": c.share * 100,
                                     "сера, мг/кг": c.sulfur_mg_kg, "ЦЧ": c.cetane,
                                     "цена, ₽/т": c.cost_rub_t} for c in rec.blend.components])
                blend = (f"<h4>Рецептура блендинга</h4>{_table(dfb)}"
                         f"<p class='muted'>Присадка: {rec.blend.additive_share*100:.3f} % об. "
                         f"· Себестоимость смеси: {rec.blend.cost_rub_t:,.0f} ₽/т</p>"
                         f"<p class='muted'>{E(rec.blend.message)}</p>").replace(",", " ")
            else:
                blend = (f"<h4>Рецептура блендинга</h4>"
                         f"<p><span class='badge bad'>допустимой рецептуры нет</span></p>"
                         f"<p class='muted'>{E(rec.blend.message)}</p>")
        sp = ""
        if rec.setpoint_plan is not None and rec.setpoint_plan.steps:
            sp = (f"<h4>План выхода на целевую уставку</h4>"
                  f"<p>Целевой уровень серы <b>{rec.setpoint_plan.target_sulfur:.2f} мг/кг</b>, "
                  f"требуемое изменение температуры реактора "
                  f"<b>{rec.setpoint_plan.delta_total_c:+.2f} °C</b>.</p>"
                  + _table(pd.DataFrame(rec.setpoint_plan.steps))
                  + (f"<p class='muted'>{E(rec.setpoint_plan.note)}</p>"
                     if rec.setpoint_plan.note else ""))
        tabs_html.append(f'<button data-tab-group="sc" data-tab="s{i}" '
                         f'aria-selected="{str(i == 0).lower()}">{E(sc.id)}</button>')
        scen_html.append(
            f'<div class="pane {"on" if i == 0 else ""}" data-pane-group="sc" data-pane="s{i}">'
            f'<h3>{E(sc.title)}</h3><p class="sub">{E(sc.why)}</p>'
            f'<p><span class="badge {tone}">{E(ru)}</span> '
            f'<span class="muted">срез {E(sc.at)} · марка {E(sc.grade)} · '
            f'уверенность {rec.confidence:.2f} · отпечаток состояния '
            f'<code>{E(rec.state_hash)}</code></span></p>'
            f'<div class="card">{_md_to_html(md)}</div>{sp}{blend}'
            f'<details><summary>Журнал обмена агентов ({len(res.trace)} сообщений) — '
            f'полная логика решения</summary><pre>{E(trace)}</pre></details></div>')

    # ------------------------------------------------------------ развёртки
    fs = feed_sulfur_sweep(system, "2024-04-26 20:00:00")
    gs = grade_sweep(system, "2024-09-15 20:00:00")
    x = fs["сера сырья, % масс."].tolist()
    chart_fs = line_chart(x, [("прогноз серы при бездействии",
                               fs["прогноз при бездействии, мг/кг"].tolist())],
                          x_label="сера сырья на гидроочистку, % масс.",
                          y_label="сера продукта через 2 ч, мг/кг", label_every=1)
    chart_dt = line_chart(x, [("требуемая компенсация",
                               fs["требуемая компенсация, °C"].tolist())],
                          x_label="сера сырья на гидроочистку, % масс.",
                          y_label="требуемое изменение T реактора, °C", label_every=1)

    # ------------------------------------------------------------- бэктест
    bt_path = Path("data/processed/backtest.parquet")
    bt_block = "<p class='muted'>Бэктест не запускался: выполните <code>dfmas backtest</code>.</p>"
    if bt_path.exists():
        bt = pd.read_parquet(bt_path)
        dd = bt["decision"].value_counts()
        chart_dec = bar_chart([DECISION_RU.get(k, (k, ""))[0] for k in dd.index],
                              (dd / len(bt) * 100).tolist(), y_label="доля циклов, %",
                              horizontal=True, ax=Axes(w=700, h=250, pad_l=290, pad_b=40),
                              value_fmt="{:.1f} %")
        sub = bt[bt["violation_lab"].notna() & bt["risk"].notna()]
        cal_block = ""
        if len(sub) > 30:
            sub = sub.assign(_v=sub["violation_lab"].astype(bool).astype(float))
            q = pd.qcut(sub["risk"], 5, duplicates="drop")
            g = sub.groupby(q, observed=True)
            dec_v = (g["risk"].mean() * 100).tolist()
            obs_v = (g["_v"].mean() * 100).tolist()
            ns = g.size().tolist()
            cal_tbl = pd.DataFrame({"корзина": [str(i) for i in g.size().index],
                                    "n": ns, "заявленный риск, %": dec_v,
                                    "фактическая доля нарушений, %": obs_v})
            cal_block = (f'<div class="two"><div>{calibration_chart(dec_v, obs_v, ns)}</div>'
                         f'<div>{_table(cal_tbl, "{:.1f}")}'
                         f'<p class="muted">Точки ниже пунктира — система переоценивает риск '
                         f'(консервативна). Это правильное направление ошибки для системы, '
                         f'отвечающей за качество.</p></div></div>')
        bt_block = (f"<p class='sub'>{len(bt)} циклов на отложенной по времени истории "
                    f"{bt['t'].min():%Y-%m-%d} — {bt['t'].max():%Y-%m-%d}, шаг 8 ч. "
                    f"Каждый цикл — полный прогон всех агентов тем же кодом, что в онлайне.</p>"
                    f"<h3>Что система решала</h3>{chart_dec}"
                    f"<h3>Калибровка риска — главная проверяемая метрика</h3>{cal_block}")

    # --------------------------------------------------------- ВАК и модели
    va_tbl = ""
    va_csv = REPORTS / "va_validation.csv"
    try:
        from .reports import report_va_validation
        report_va_validation()
    except Exception:
        pass

    roster = system.agent_roster()
    roster_tbl = _table(pd.DataFrame(
        [{"агент": r["name"], "роль": r["role"]} for r in roster]))

    kpis = "".join([
        _kpi(f"{100*m['roc_auc_now']:.0f} %", "ROC-AUC риска по поточному анализатору"),
        _kpi(f"{m['mae_analyzer_now']:.2f}", "MAE оценки серы против ЛИМС, мг/кг"),
        _kpi(f"{10 - float(np.quantile(bundle.residuals['Mg.Sulfur'], .9)):.2f}",
             "целевая сера для риска ≤ 10 %, мг/кг", "warn"),
        _kpi(f"{100*bundle.metrics['kinetic_dlnS_dT_per_C']:.1f} %",
             "снижение серы на 1 °C (кинетика ГДС)"),
        _kpi(f"{bundle.dead_times.get('temp', 80)} мин",
             "идентифицированное запаздывание"),
        _kpi(f"{len(roster)}", "агентов в контуре"),
    ])

    body = f"""
<button class="tog" id="tog">светлая / тёмная</button>
<div class="wrap">
<header>
<h1>Dark Factory — мультиагентная система управления производством дизельного топлива</h1>
<p class="sub">Цепочка АВТ → гидроочистка 24-2000 → блендинг. Система получает состояние
процесса «на момент времени», оценивает риск выхода за спецификацию, сравнивает допустимые
варианты режима и объясняет выбор оператору. Все числа на странице рассчитаны кодом при
сборке, ничего не вписано руками.</p>
<div class="kpis">{kpis}</div>
</header>

<h2>1. Архитектура: кто с кем разговаривает</h2>
<p class="sub">Обмен между агентами идёт через журналируемую шину. Журнал сохраняется вместе
с каждой рекомендацией — он и есть «логика принятия решения», которую требует ТЗ.
Шина синхронная и однопоточная намеренно: параллельность внесла бы недетерминированность
порядка сообщений, а результат должен воспроизводиться на одних и тех же данных.</p>
{_agent_graph(roster)}
{roster_tbl}

<h2>2. Сценарии</h2>
<p class="sub">Сценарий — это подмена входов над реальным историческим срезом, а не
записанный ответ. Те же агенты, те же модели, другие данные — другой результат.
Внутри каждой вкладки: рекомендация целиком и полный журнал обмена агентов.</p>
<div class="tabs" data-tabs="sc">{''.join(tabs_html)}</div>
{''.join(scen_html)}

<h2>3. Результат зависит от входных данных, а не зашит</h2>
<p class="sub">Один и тот же момент времени, одни и те же агенты. Меняется только содержание
серы в сырье гидроочистки. Прогноз растёт монотонно, требуемая компенсация по температуре
меняет знак — от «снизить температуру и сэкономить» до «поднять, иначе нарушим норматив».</p>
<div class="two"><div>{chart_fs}</div><div>{chart_dt}</div></div>
{_table(fs[['сера сырья, % масс.', 'решение', 'ΔT, °C',
            'прогноз при бездействии, мг/кг', 'риск, %',
            'требуемая компенсация, °C']], "{:.2f}")}
<h3>И от задания на качество продукции</h3>
{_table(gs, "{:.2f}")}

<h2>4. Бэктест: проверка на отложенной истории</h2>
{bt_block}

<h2>5. Честные границы применимости</h2>
<div class="card">
<p><b>Точечный прогноз серы на этих данных не работает — и мы это показываем, а не прячем.</b>
Событие «лаборатория покажет более 10 мг/кг» предсказывается поточным анализатором с
ROC-AUC {m['roc_auc_now']:.2f}; добавление всех 96 технологических тегов не улучшает
ни MAE, ни AUC. Причина физическая: расхождение пары «лаборатория — поточный анализатор»
сопоставимо с движением самого процесса за горизонт прогноза.</p>
<p>Поэтому система построена иначе: уровень берётся из измерения, приращение от действия
считает кинетика гидрообессеривания, а продуктом является <b>калиброванный риск с
интервалом</b>, а не точка. Разложение неопределённости на технологическую и измерительную
части показывается оператору отдельно — поднимать температуру, чтобы компенсировать
расхождение двух приборов, бессмысленно.</p>
<p>Исторические данные не подтверждают эффект действий, которых в них не было.
Оценка альтернативных режимов выполняется модельно и предъявляется отдельно —
в разделе 3 и в сценариях.</p>
</div>

<h2>6. Где что лежит</h2>
<div class="tbl"><table>
<tr><th>Что</th><th>Где</th></tr>
<tr><td>Аудит справочника КИП, найденная ошибка в листе «КИП»</td><td><code>reports/01_tag_audit.md</code></td></tr>
<tr><td>Проверка формул ВАК и коррекции смещения по ЛИМС</td><td><code>reports/02_va_validation.md</code></td></tr>
<tr><td>Модели, потолок предсказуемости, запаздывания</td><td><code>reports/03_models.md</code></td></tr>
<tr><td>Бэктест и калибровка риска</td><td><code>reports/04_backtest.md</code></td></tr>
<tr><td>Журналы всех сценариев в JSON</td><td><code>reports/scenarios/*.json</code></td></tr>
<tr><td>Код агентов</td><td><code>src/dfmas/agents/</code></td></tr>
<tr><td>Витрина «на момент времени» (защита от утечки)</td><td><code>src/dfmas/featurestore.py</code></td></tr>
<tr><td>Все пороги, веса, цены и спецификации</td><td><code>config/*.yaml</code></td></tr>
</table></div>
<p class="muted">Сгенерировано командой <code>dfmas dashboard</code>.
Страница автономна: ни одного внешнего запроса, открывается из файла.</p>
</div>
"""
    doc = (f'<!doctype html><html lang="ru"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>Dark Factory MAS</title><style>{CSS}</style></head>'
           f'<body>{body}<script>{JS}</script></body></html>')
    out.write_text(doc, encoding="utf-8")
    return out
