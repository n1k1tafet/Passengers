"""Обнаружение непригодных данных и корректный отказ."""
import numpy as np
import pandas as pd


def test_sentinel_307_is_detected():
    from dfmas.quality.sentinels import (SENTINEL, build_profile, flag_series)
    s = pd.Series([1.0, 2.0, 307.0, 3.0, 307.0] * 40, name="x")
    prof = build_profile(s, "x")
    flags = flag_series(s, prof)
    assert (flags == SENTINEL).sum() == 80


def test_digital_states_found_without_hardcoded_list():
    from dfmas.quality.sentinels import DIGITAL_STATE, build_profile, flag_series
    rng = np.random.default_rng(0)
    v = np.r_[rng.normal(50, 2, 5000), np.full(120, 251.0)]
    s = pd.Series(v, name="x")
    prof = build_profile(s, "x")
    assert 251.0 in prof.digital_states
    assert (flag_series(s, prof) == DIGITAL_STATE).sum() == 120


def test_flatline_detected():
    from dfmas.quality.sentinels import FLATLINE, build_profile, flag_series
    rng = np.random.default_rng(1)
    v = np.r_[rng.normal(10, 1, 500), np.full(40, 10.0), rng.normal(10, 1, 500)]
    s = pd.Series(v, name="x")
    prof = build_profile(s, "x")
    assert (flag_series(s, prof, flatline_samples=18) == FLATLINE).sum() >= 30


def test_system_refuses_on_conflicting_sulfur_analyzers(system):
    rec = system.run_at("2024-09-15 20:00:00", overrides={"H_Q21": 3.0}).recommendation
    assert rec.decision == "refuse"
    assert any("расход" in r or "расхожд" in r for r in rec.reasons)


def test_system_refuses_when_key_tag_missing(system):
    rec = system.run_at("2024-09-15 20:00:00",
                        overrides={"H_Q21": float("nan"), "H_T5": float("nan")}).recommendation
    assert rec.decision == "refuse"


def test_stale_feed_analysis_lowers_confidence(system):
    fresh = system.run_at("2024-09-15 20:00:00").recommendation
    assert fresh.confidence > 0.5
    a = system.run_at("2025-11-20 14:00:00").recommendation
    assert a.confidence <= fresh.confidence


def test_refusal_always_has_a_reason(system):
    for ts in ["2023-03-04 03:00:00", "2024-09-15 20:00:00"]:
        rec = system.run_at(ts, overrides={"H_Q21": 3.0}).recommendation
        if rec.decision == "refuse":
            assert rec.reasons, "отказ без объяснения недопустим"
