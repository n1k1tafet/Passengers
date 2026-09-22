"""ETL: сырые выгрузки -> витрина parquet с флагами качества.

Запуск:  ``python -m dfmas.cli ingest``   (или ``make data``)

Результат в ``data/processed``:
  * ``telemetry.parquet``    — очищенная 10-минутная телеметрия обеих установок
  * ``telemetry_flags.parquet`` — коды качества каждого значения
  * ``lab.parquet``          — ЛИМС + ПАК в длинном формате с ``available_at``
  * ``tag_profiles.json``    — паспорта тегов (границы, цифровые состояния)

Принципы
--------
* Ничего не «чинится» молча: каждое отброшенное значение имеет код причины.
* Префиксы ``A_`` / ``H_`` разводят одноимённые теги двух установок
  (в обоих файлах есть, например, F9 и T6 — это РАЗНЫЕ приборы).
* Паспорта тегов строятся ТОЛЬКО по обучающему отрезку истории, чтобы
  границы не подсматривали будущее.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_PROCESSED, DATA_RAW, load_config
from ..quality.sentinels import build_profile, clean_frame
from .lims import read_lims, read_pak

AVT_PREFIX, HT_PREFIX = "A_", "H_"
#: Граница обучающей части истории. Всё, что позже, не участвует ни в
#: построении паспортов тегов, ни в обучении моделей.
TRAIN_END = pd.Timestamp("2025-06-01")


def _read_telemetry(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.columns = [prefix + c for c in df.columns]
    return df.astype("float32")


def run_ingest(raw_dir: Path | None = None, out_dir: Path | None = None,
               verbose: bool = True) -> dict[str, Path]:
    raw = Path(raw_dir or DATA_RAW)
    out = Path(out_dir or DATA_PROCESSED)
    out.mkdir(parents=True, exist_ok=True)

    avt = _read_telemetry(raw / "avt_tags.csv", AVT_PREFIX)
    ht = _read_telemetry(raw / "242000_tags.csv", HT_PREFIX)
    tele = avt.join(ht, how="outer").sort_index()
    if verbose:
        print(f"[ingest] телеметрия: {tele.shape[0]} точек x {tele.shape[1]} тегов, "
              f"{tele.index.min()} .. {tele.index.max()}")

    cfg = load_config()
    flat = int(cfg.agents["data_quality"]["flatline_samples"])
    train = tele.loc[:TRAIN_END]
    profiles = {c: build_profile(train[c], c) for c in tele.columns}
    values, flags = clean_frame(tele, profiles, flatline_samples=flat)
    bad_share = (flags.to_numpy() != 0).mean() * 100
    if verbose:
        print(f"[ingest] помечено непригодными {bad_share:.2f} % значений")

    lab = pd.concat([read_lims(raw / "LIMS.xlsx"), read_pak(raw / "PAK.xlsx")],
                    ignore_index=True)
    lab = lab.sort_values("available_at").reset_index(drop=True)
    if verbose:
        print(f"[ingest] анализы: {len(lab)} записей, "
              f"{lab.source.value_counts().to_dict()}")

    paths = {}
    paths["telemetry"] = out / "telemetry.parquet"
    values.to_parquet(paths["telemetry"])
    paths["flags"] = out / "telemetry_flags.parquet"
    flags.to_parquet(paths["flags"])
    paths["lab"] = out / "lab.parquet"
    lab.to_parquet(paths["lab"])
    paths["profiles"] = out / "tag_profiles.json"
    with open(paths["profiles"], "w", encoding="utf-8") as fh:
        json.dump({k: v.to_dict() for k, v in profiles.items()}, fh,
                  ensure_ascii=False, indent=1)
    if verbose:
        print("[ingest] готово ->", out)
    return paths


def load_processed(out_dir: Path | None = None):
    out = Path(out_dir or DATA_PROCESSED)
    tele = pd.read_parquet(out / "telemetry.parquet")
    flags = pd.read_parquet(out / "telemetry_flags.parquet")
    lab = pd.read_parquet(out / "lab.parquet")
    return tele, flags, lab
