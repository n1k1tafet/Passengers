"""Свойства моделей, на которые опирается безопасность рекомендаций."""
import numpy as np
import pytest

from dfmas.models.hds import (HDSParams, calibrate_k0, dlnS_dT, required_temperature,
                              sulfur_out)


@pytest.fixture(scope="module")
def hds():
    return calibrate_k0(9500, 8.5, 350.0, 1.5, 4.0, HDSParams())


def test_sulfur_decreases_with_temperature(hds):
    vals = [sulfur_out(9500, T, 1.5, 4.0, hds) for T in (344, 347, 350, 353, 356)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_sulfur_increases_with_throughput(hds):
    vals = [sulfur_out(9500, 350, l, 4.0, hds) for l in (1.2, 1.4, 1.6, 1.8)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_sulfur_decreases_with_pressure(hds):
    vals = [sulfur_out(9500, 350, 1.5, p, hds) for p in (3.4, 3.8, 4.2, 4.6)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_heavier_feed_requires_higher_temperature(hds):
    """Сценарий эксперта: выросла сера сырья -> нужна более высокая температура."""
    temps = [required_temperature(8.0, s * 10_000, 1.5, 4.0, hds)
             for s in (0.80, 0.95, 1.10, 1.25)]
    assert all(a < b for a, b in zip(temps, temps[1:]))
    assert temps[-1] - temps[0] > 3.0


def test_inverse_problem_is_consistent(hds):
    T = required_temperature(6.0, 9500, 1.5, 4.0, hds)
    assert abs(sulfur_out(9500, T, 1.5, 4.0, hds) - 6.0) < 0.05


def test_temperature_sensitivity_in_industrial_range(hds):
    s = 100 * dlnS_dT(9500, 350, 1.5, 4.0, hds)
    assert -12.0 < s < -2.0, f"чувствительность {s:.2f} %/°C вне разумного диапазона"


def test_dynamic_fraction_respects_dead_time():
    from dfmas.models.process_model import dynamic_fraction
    assert dynamic_fraction(30, 80) == 0.0
    assert 0.0 < dynamic_fraction(120, 80) < 1.0
    assert dynamic_fraction(600, 80) > 0.99


def test_conformal_interval_covers_calibration_level():
    from dfmas.models.conformal import ConformalCalibration
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1.5, 4000)
    cal = ConformalCalibration.from_residuals("x", 120, r)
    lo, hi = cal.interval(10.0, 0.90)
    cover = ((10.0 + r >= lo) & (10.0 + r <= hi)).mean()
    assert 0.87 < cover < 0.93


def test_online_bias_only_uses_published_results():
    import pandas as pd
    from dfmas.models.bias import apply_online_bias
    idx = pd.date_range("2024-01-01", periods=200, freq="10min")
    base = pd.Series(np.zeros(len(idx)), index=idx)
    lab = pd.Series([5.0], index=[idx[10]])
    avail = pd.DatetimeIndex([idx[100]])          # опубликован сильно позже
    b = apply_online_bias(base, lab, avail, 24, 10.0, 720)
    assert abs(b.iloc[50]) < 1e-9, "коррекция применена до публикации результата"
    assert b.iloc[150] > 1.0, "коррекция не применена после публикации"
