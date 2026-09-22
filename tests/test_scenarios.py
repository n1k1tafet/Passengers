"""Сценарии «мягкие»: результат меняется вслед за входными данными."""
import numpy as np


def test_scenarios_all_run(system):
    from dfmas.scenarios import load_scenarios, run_scenario
    for sc in load_scenarios():
        res = run_scenario(system, sc)
        assert res.recommendation.decision in {
            "hold", "act", "refuse", "act_best_effort", "hold_best_effort", "act_recovery"}
        assert res.trace, f"{sc.id}: пустой журнал обмена агентов"


def test_decision_responds_to_feed_sulfur(system):
    """Ключевое требование эксперта: результат зависит от качества сырья."""
    from dfmas.sweep import feed_sulfur_sweep
    df = feed_sulfur_sweep(system, "2024-04-26 20:00:00",
                           values=[0.70, 1.00, 1.30, 1.60, 1.80])
    assert df["прогноз при бездействии, мг/кг"].is_monotonic_increasing, (
        "прогноз без вмешательства должен расти вместе с серой сырья")
    comp = df["требуемая компенсация, °C"].to_numpy()
    assert np.all(np.diff(comp) > 0), "требуемая температура не растёт с серой сырья"
    assert df["решение"].nunique() > 1, "решение не меняется — сценарий «жёсткий»"


def test_decision_responds_to_grade(system):
    """И от задания на качество продукции."""
    from dfmas.sweep import grade_sweep
    df = grade_sweep(system, "2024-09-15 20:00:00")
    prices = df["цена смеси, ₽/т"].dropna().unique()
    assert len(prices) > 1, "рецептура не зависит от марки топлива"


def test_winter_grade_uses_low_cfpp_component(system):
    rec = system.run_at("2024-09-15 20:00:00", grade="DT_WINTER").recommendation
    assert rec.blend.feasible
    assert rec.blend.blended["CFPP"] <= -20.0 + 1e-6
