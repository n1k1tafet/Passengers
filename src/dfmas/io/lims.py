"""Парсер ЛИМС/ПАК выгрузок в длинный формат с явной семантикой доступности.

Ключевое соглашение проекта
---------------------------
Каждое измерение качества несёт ДВЕ метки времени:
  * ``measured_at``  — момент отбора пробы (то, что лежит в файле);
  * ``available_at`` — момент, когда результат физически мог попасть
    к оператору/в систему.

Для ЛИМС ``available_at = measured_at + LIMS_PUBLISH_DELAY`` (4 часа,
подтверждено экспертом), для ПАК — +1 шаг опроса (10 минут).
Любая выборка "на момент t" фильтруется по ``available_at <= t``,
поэтому утечка из будущего невозможна конструктивно, а не "по договорённости".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LIMS_PUBLISH_DELAY = pd.Timedelta(hours=4)
PAK_PUBLISH_DELAY = pd.Timedelta(minutes=10)
TELEMETRY_PUBLISH_DELAY = pd.Timedelta(0)

# Единицы измерения в строке 3 файла ЛИМС перепутаны (подтверждено экспертом:
# "смотрите на показатели качества в названиях"). Восстанавливаем по имени показателя.
CANONICAL_UNITS = {
    "CFPP": "degC", "90%.T": "degC", "50%.T": "degC", "95%.T": "degC",
    "EBP.T": "degC", "IBP.T": "degC", "PourPoint": "degC", "CloudPoint": "degC",
    "CloudPoint_1": "degC", "FilterabilityLimit.T": "degC", "FlashPoint": "degC",
    "D15": "kg/m3", "I350": "vol%", "I250": "vol%",
    "Mass.Sulfur": "wt%", "Mg.Sulfur": "mg/kg", "CetaneNumber": "cetane",
}

# Человекочитаемые русские названия показателей (для дашборда и объяснений)
RU_NAMES = {
    "CFPP": "Предельная температура фильтруемости (ПТФ)",
    "90%.T": "Температура выкипания 90 % об.",
    "95%.T": "Температура выкипания 95 % об.",
    "50%.T": "Температура выкипания 50 % об.",
    "EBP.T": "Конец кипения", "IBP.T": "Начало кипения",
    "PourPoint": "Температура застывания", "CloudPoint": "Температура помутнения",
    "CloudPoint_1": "Температура помутнения (дубль)",
    "FilterabilityLimit.T": "Предел фильтруемости",
    "FlashPoint": "Температура вспышки", "D15": "Плотность при 15 °C",
    "I350": "Отгон до 350 °C", "I250": "Отгон до 250 °C",
    "Mass.Sulfur": "Массовая доля серы", "Mg.Sulfur": "Массовая концентрация серы",
    "CetaneNumber": "Цетановое число",
}

# Нормализация "Установка 'X'. Точка отбора 'N'. Продукт 'P'" -> короткий ключ
_STREAM_RE = re.compile(
    r"Установка\s*'(?P<unit>[^']+)'\.*\s*Точка отбора\s*'(?P<point>[^']+)'\.*\s*Продукт\s*'(?P<product>[^']+)'")

# Короткие коды потоков, которыми оперирует вся система
STREAM_CODES = {
    ("АВТ", "1", "ФРАКЦ_ДИЗ"): "AVT_DIESEL_P1",
    ("АВТ", "1", "Дизельное топливо"): "AVT_DIESEL_P1",
    ("АВТ", "2", "Дизельное топливо"): "AVT_DIESEL_P2",
    ("АВТ", "2.1", "Дизельное топливо"): "AVT_DIESEL_P21",
    ("АВТ", "3", "Дизельное топливо"): "AVT_DIESEL_P3",
    ("Гидроочистка", "1", "ФРАКЦ_ДИЗ"): "HT_FEED",     # сырьё гидроочистки
    ("Гидроочистка", "2", "Дизельное топливо"): "HT_PRODUCT",  # гидроочищенное ДТ
}

STREAM_RU = {
    "AVT_DIESEL_P1": "АВТ, точка 1 — дизельная фракция",
    "AVT_DIESEL_P2": "АВТ, точка 2 — дизельное топливо",
    "AVT_DIESEL_P21": "АВТ, точка 2.1 — дизельное топливо",
    "AVT_DIESEL_P3": "АВТ, точка 3 — дизельное топливо",
    "HT_FEED": "Гидроочистка, точка 1 — сырьё (прямогонная дизельная фракция)",
    "HT_PRODUCT": "Гидроочистка, точка 2 — гидроочищенное ДТ (продукт)",
}


def _stream_code(header: str) -> str:
    m = _STREAM_RE.search(str(header).replace("''", "'"))
    if not m:
        return "UNKNOWN"
    key = (m.group("unit").strip(" ."), m.group("point").strip(" ."),
           m.group("product").strip(" ."))
    return STREAM_CODES.get(key, "UNKNOWN:" + "|".join(key))


def read_lims(path: str | Path) -> pd.DataFrame:
    """Читает выгрузку ЛИМС (пары колонок «метка времени / значение»)."""
    raw = pd.read_excel(path, header=None, dtype=object)
    groups, params = raw.iloc[0].tolist(), raw.iloc[1].tolist()
    records = []
    current = None
    for col in range(0, raw.shape[1], 2):
        if col < len(groups) and pd.notna(groups[col]):
            current = _stream_code(groups[col])
        param = params[col] if col < len(params) else None
        if pd.isna(param):
            continue
        block = raw.iloc[3:, [col, col + 1]].copy()
        block.columns = ["measured_at", "value"]
        block["measured_at"] = pd.to_datetime(block["measured_at"], errors="coerce")
        block["value"] = pd.to_numeric(block["value"], errors="coerce")
        block = block.dropna()
        if block.empty:
            continue
        block["stream"] = current
        block["param"] = str(param).strip()
        records.append(block)
    out = pd.concat(records, ignore_index=True)
    out["unit"] = out["param"].map(CANONICAL_UNITS).fillna("?")
    out["source"] = "LIMS"
    out["available_at"] = out["measured_at"] + LIMS_PUBLISH_DELAY
    return out.sort_values("measured_at").reset_index(drop=True)


def read_pak(path: str | Path) -> pd.DataFrame:
    """Читает выгрузку ПАК (поточные анализаторы 24-2000)."""
    raw = pd.read_excel(path, header=None, dtype=object)
    tags, units = raw.iloc[0].tolist(), raw.iloc[1].tolist()
    records = []
    for col in range(0, raw.shape[1], 2):
        tag = tags[col] if col < len(tags) else None
        if pd.isna(tag):
            continue
        block = raw.iloc[2:, [col, col + 1]].copy()
        block.columns = ["measured_at", "value"]
        block["measured_at"] = pd.to_datetime(block["measured_at"], errors="coerce")
        block["value"] = pd.to_numeric(block["value"], errors="coerce")
        block = block.dropna()
        if block.empty:
            continue
        block["stream"] = "HT_PRODUCT"
        block["param"] = {"24-2000:Mg.Sulfur": "Mg.Sulfur", "24-2000:D15": "D15"}.get(
            str(tag).strip(), str(tag).strip())
        block["unit"] = CANONICAL_UNITS.get(block["param"].iloc[0], str(units[col]))
        records.append(block)
    out = pd.concat(records, ignore_index=True)
    out["source"] = "PAK"
    out["available_at"] = out["measured_at"] + PAK_PUBLISH_DELAY
    return out.sort_values("measured_at").reset_index(drop=True)
