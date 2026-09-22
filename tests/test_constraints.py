"""Жёсткие ограничения не нарушаются никогда — это условие безопасности."""
import numpy as np
import pytest

TIMES = ["2024-09-15 20:00:00", "2025-11-20 14:00:00", "2024-04-26 20:00:00",
         "2026-02-10 06:00:00"]


@pytest.mark.parametrize("ts", TIMES)
def test_recommended_option_never_violates_point_limits(system, ts):
    rec = system.run_at(ts).recommendation
    if rec.option is None or rec.decision == "act_recovery":
        return                      # режим вывода в норму — спецификация уже нарушена
    for item in rec.option.predictions["quality"].items:
        if not item.hard or item.margin is None or not np.isfinite(item.margin):
            continue
        assert item.margin >= -1e-9, (
            f"{ts}: рекомендация нарушает жёсткое ограничение {item.param}")


@pytest.mark.parametrize("ts", TIMES)
def test_economics_cannot_override_quality(system, ts):
    """Среди выбранных не должно быть варианта с большей маржой и нарушением."""
    res = system.run_at(ts)
    rec = res.recommendation
    if rec.option is None:
        return
    assert rec.option.metrics.feasible or rec.decision == "act_recovery"


def test_blend_shares_sum_to_one(system):
    rec = system.run_at("2024-09-15 20:00:00", grade="DT_SUMMER").recommendation
    plan = rec.blend
    assert plan is not None and plan.feasible
    total = sum(c.share for c in plan.components) + plan.additive_share
    assert abs(total - 1.0) < 1e-6


def test_additive_share_within_three_percent(system):
    from dfmas.agents.blending import Tank
    poor = [Tank("Р-1", 3.0, 344.0, 45.0, 834.0), Tank("Р-2", 8.0, 347.0, 44.0, 836.0)]
    rec = system.run_at("2024-09-15 20:00:00", grade="DT_SUMMER", tanks=poor).recommendation
    assert rec.blend.additive_share <= 0.03 + 1e-9


def test_blend_respects_sulfur_limit_with_margin(system):
    rec = system.run_at("2024-09-15 20:00:00", grade="DT_SUMMER").recommendation
    assert rec.blend.blended["Mg.Sulfur"] <= 10.0


def test_blend_infeasible_is_reported_not_faked(system):
    from dfmas.agents.blending import Tank
    dirty = [Tank("Р-1", 12.0, 349.0, 52.0, 836.0), Tank("Р-2", 14.0, 351.0, 51.5, 838.0)]
    rec = system.run_at("2024-09-15 20:00:00", grade="DT_SUMMER", tanks=dirty).recommendation
    assert rec.blend.feasible is False
    assert "нет" in rec.blend.message.lower()


def test_reliability_alarm_blocks_temperature_increase(system):
    rec = system.run_at("2025-11-20 14:00:00",
                        overrides={"H_W10": 6.0, "H_P24": 0.40}).recommendation
    if rec.option is not None:
        rel = rec.option.predictions["reliability"]
        if rel.severity.value == "alarm":
            assert rec.option.action.d_temp_c <= 1e-9
