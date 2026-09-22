"""Запуск тестовых сценариев.

Сценарий — это подмена входов над реальным историческим срезом, а не
записанный ответ. Тот же код агентов, те же модели, другие данные — и другой
результат. Именно этого требовал эксперт: сценарии не должны быть «жёсткими».
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .config import CONFIG_DIR, REPORTS
from .agents.blending import Tank
from .system import CycleResult, DarkFactorySystem


@dataclass
class Scenario:
    id: str
    title: str
    why: str
    at: str
    grade: str = "DT_SUMMER"
    overrides: dict = field(default_factory=dict)
    lab_overrides: dict = field(default_factory=dict)
    tanks: list | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        tanks = None
        if d.get("tanks"):
            tanks = [Tank(**t) for t in d["tanks"]]
        lab = {}
        for k, v in (d.get("lab_overrides") or {}).items():
            stream, param = k.split(":", 1)
            lab[(stream, param)] = float(v)
        ovr = {k: (float("nan") if v is None or (isinstance(v, float) and np.isnan(v))
                   else float(v)) for k, v in (d.get("overrides") or {}).items()}
        return cls(id=d["id"], title=d["title"], why=d.get("why", "").strip(),
                   at=str(d["at"]), grade=d.get("grade", "DT_SUMMER"),
                   overrides=ovr, lab_overrides=lab, tanks=tanks)


def load_scenarios(path: Path | None = None) -> list[Scenario]:
    path = Path(path or CONFIG_DIR / "scenarios.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Scenario.from_dict(d) for d in data["scenarios"]]


def run_scenario(system: DarkFactorySystem, sc: Scenario) -> CycleResult:
    return system.run_at(sc.at, grade=sc.grade, overrides=sc.overrides or None,
                         lab_overrides=sc.lab_overrides or None, tanks=sc.tanks)


def run_all(system: DarkFactorySystem, out_dir: Path | None = None,
            verbose: bool = True) -> dict[str, CycleResult]:
    out_dir = Path(out_dir or REPORTS / "scenarios")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for sc in load_scenarios():
        res = run_scenario(system, sc)
        results[sc.id] = res
        payload = json.loads(res.to_json())
        payload["scenario"] = {"id": sc.id, "title": sc.title, "why": sc.why,
                               "at": sc.at, "grade": sc.grade,
                               "overrides": {k: (None if not np.isfinite(v) else v)
                                             for k, v in sc.overrides.items()},
                               "lab_overrides": {f"{k[0]}:{k[1]}": v
                                                 for k, v in sc.lab_overrides.items()},
                               "tanks": [t.__dict__ for t in sc.tanks] if sc.tanks else None}
        (out_dir / f"{sc.id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        if verbose:
            rec = res.recommendation
            print(f"  [{sc.id:18s}] {rec.decision:18s} | {rec.action_label[:52]:52s} "
                  f"| conf {rec.confidence:.2f}")
    return results
