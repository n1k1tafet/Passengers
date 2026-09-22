"""Загрузка конфигурации и путей проекта.

Все пороги, веса, цены и спецификации вынесены в ``config/*.yaml``.
В коде нет ни одной «магической константы», влияющей на рекомендацию, —
это условие воспроизводимости из ТЗ.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("DFMAS_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = ROOT / "config"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"

# Единственный источник случайности в проекте. Любая модель, использующая
# генератор, получает именно это зерно -> повторный запуск даёт тот же ответ.
SEED = 20260915


def _load(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Config:
    tags: dict = field(default_factory=dict)
    specs: dict = field(default_factory=dict)
    economics: dict = field(default_factory=dict)
    agents: dict = field(default_factory=dict)

    @property
    def horizon_min(self) -> int:
        return int(self.agents["orchestrator"]["prediction_horizon_minutes"])

    @property
    def step_min(self) -> int:
        return int(self.agents["orchestrator"]["recommendation_step_minutes"])

    def grade(self, name: str) -> dict:
        return self.specs["grades"][name]


@lru_cache(maxsize=1)
def load_config() -> Config:
    return Config(tags=_load("tags.yaml"), specs=_load("specs.yaml"),
                  economics=_load("economics.yaml"), agents=_load("agents.yaml"))


def mv_tags(unit: str = "hydrotreater", include_secondary: bool = False) -> list[str]:
    cfg = load_config()
    out = []
    for item in cfg.tags[unit]["manipulated"]:
        if item["role"] == "MV" or (include_secondary and item["role"].startswith("MV")):
            out.append(item["tag"])
    return out


def tag_meta(unit: str, tag: str) -> dict:
    cfg = load_config()
    for group in ("manipulated", "controlled", "disturbance"):
        for item in cfg.tags.get(unit, {}).get(group, []) or []:
            if item["tag"] == tag:
                return item
    return {}
