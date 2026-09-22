"""Командный интерфейс: единственная точка запуска всего проекта.

    python -m dfmas.cli ingest                 # сырые файлы -> витрина parquet
    python -m dfmas.cli fit                    # обучение и калибровка моделей
    python -m dfmas.cli run --at "2025-11-20 14:00"   # один цикл решения
    python -m dfmas.cli scenario --all         # все тестовые сценарии
    python -m dfmas.cli sweep --at ...         # развёртка по сере сырья и марке
    python -m dfmas.cli backtest               # бэктест на отложенной истории
    python -m dfmas.cli report                 # сгенерировать reports/*.md
    python -m dfmas.cli dashboard              # собрать HTML-дашборд
    python -m dfmas.cli all                    # полный конвейер с нуля
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from .config import ARTIFACTS, REPORTS


def _system():
    from .system import DarkFactorySystem
    return DarkFactorySystem.build()


def cmd_ingest(args) -> int:
    from .io.ingest import run_ingest
    run_ingest()
    return 0


def cmd_fit(args) -> int:
    from .models.fit import fit_all
    fit_all(horizon_min=args.horizon)
    return 0


def cmd_run(args) -> int:
    sysm = _system()
    res = sysm.run_at(args.at, grade=args.grade, horizon_min=args.horizon)
    rec = res.recommendation
    print(rec.expected_effect.get("_markdown", rec.headline))
    if args.json:
        path = Path(args.json)
        res.save(path)
        print(f"\n[сохранено] {path}")
    return 0


def cmd_scenario(args) -> int:
    from .scenarios import load_scenarios, run_all, run_scenario
    sysm = _system()
    if args.all:
        run_all(sysm)
        print(f"\nПолные журналы: {REPORTS / 'scenarios'}")
        return 0
    scs = {s.id: s for s in load_scenarios()}
    if args.id not in scs:
        print("Доступные сценарии:", ", ".join(scs))
        return 2
    sc = scs[args.id]
    print(f"# {sc.title}\n\n{sc.why}\n")
    res = run_scenario(sysm, sc)
    print(res.recommendation.expected_effect.get("_markdown", ""))
    return 0


def cmd_sweep(args) -> int:
    from .sweep import feed_sulfur_sweep, grade_sweep
    sysm = _system()
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("## Зависимость решения от СЕРЫ СЫРЬЯ (одно и то же время, одни агенты)\n")
    fs = feed_sulfur_sweep(sysm, args.at, grade=args.grade)
    print(fs.to_string(index=False))
    print("\n## Зависимость от ЗАДАНИЯ НА КАЧЕСТВО (марка товарного ДТ)\n")
    gs = grade_sweep(sysm, args.at)
    print(gs.to_string(index=False))
    REPORTS.mkdir(parents=True, exist_ok=True)
    fs.to_csv(REPORTS / "sweep_feed_sulfur.csv", index=False)
    gs.to_csv(REPORTS / "sweep_grade.csv", index=False)
    print(f"\n[сохранено] {REPORTS/'sweep_feed_sulfur.csv'}, {REPORTS/'sweep_grade.csv'}")
    return 0


def cmd_backtest(args) -> int:
    from .backtest import run_backtest
    from .reports import report_backtest
    sysm = _system()
    bt = run_backtest(sysm, start=args.start, end=args.end, step_hours=args.step,
                      grade=args.grade)
    print(json.dumps(bt.summary, ensure_ascii=False, indent=1))
    print()
    print(bt.calibration.to_string(index=False))
    bt.rows.to_parquet("data/processed/backtest.parquet")
    print("\n[сохранено]", report_backtest(bt))
    return 0


def cmd_report(args) -> int:
    from .reports import report_models, report_tag_audit, report_va_validation
    for fn in (report_tag_audit, report_va_validation, report_models):
        print("[отчёт]", fn())
    return 0


def cmd_dashboard(args) -> int:
    from .dashboard import build_dashboard
    print("[дашборд]", build_dashboard())
    return 0


def cmd_all(args) -> int:
    cmd_ingest(args)
    cmd_fit(args)
    cmd_report(args)
    from .scenarios import run_all
    print("\n[сценарии]")
    run_all(_system())
    cmd_dashboard(args)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dfmas",
                                 description="Мультиагентная система управления "
                                             "производством дизельного топлива")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="сырые файлы -> витрина parquet")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("fit", help="обучение и калибровка моделей")
    p.add_argument("--horizon", type=int, default=None, help="горизонт прогноза, мин")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("run", help="один цикл принятия решения")
    p.add_argument("--at", required=True, help="момент времени, например 2025-11-20 14:00")
    p.add_argument("--grade", default="DT_SUMMER", help="марка: HT_PRODUCT|DT_SUMMER|DT_WINTER")
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--json", default=None, help="сохранить полный журнал агентов в файл")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("scenario", help="тестовые сценарии")
    p.add_argument("id", nargs="?", default=None)
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_scenario)

    p = sub.add_parser("sweep", help="развёртка решения по входным условиям")
    p.add_argument("--at", default="2024-04-26 20:00:00")
    p.add_argument("--grade", default="DT_SUMMER")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("backtest", help="бэктест на отложенной истории")
    p.add_argument("--start", default="2025-06-01")
    p.add_argument("--end", default="2026-08-01")
    p.add_argument("--step", type=int, default=8, help="шаг в часах")
    p.add_argument("--grade", default="DT_SUMMER")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("report", help="сгенерировать отчёты reports/*.md")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("dashboard", help="собрать HTML-дашборд")
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("all", help="полный конвейер: ingest -> fit -> report -> сценарии -> дашборд")
    p.add_argument("--horizon", type=int, default=None)
    p.set_defaults(func=cmd_all)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
