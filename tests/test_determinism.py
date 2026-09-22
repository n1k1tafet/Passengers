"""Воспроизводимость.

Эксперт: «При одинаковом состоянии системы должны воспроизводиться значимые
результаты... главное — одно и то же решение и одинаковые численные
результаты». У нас совпадает ещё и текст: он собирается из шаблонов.
"""
import json


def test_same_state_same_recommendation(system):
    a = system.run_at("2025-11-20 14:00:00")
    b = system.run_at("2025-11-20 14:00:00")
    ra, rb = a.recommendation, b.recommendation
    assert ra.state_hash == rb.state_hash
    assert ra.decision == rb.decision
    assert ra.action_label == rb.action_label
    assert ra.headline == rb.headline
    assert abs(ra.confidence - rb.confidence) < 1e-12
    assert json.dumps(ra.as_dict(), sort_keys=True, default=str) == \
           json.dumps(rb.as_dict(), sort_keys=True, default=str)


def test_numeric_results_identical_across_runs(system):
    vals = []
    for _ in range(3):
        rec = system.run_at("2024-09-15 20:00:00").recommendation
        item = next(i for i in rec.option.predictions["quality"].items
                    if i.param == "Mg.Sulfur")
        vals.append((round(item.forecast, 9), round(item.exceed_prob, 9),
                     round(rec.option.action.d_temp_c, 9)))
    assert len(set(vals)) == 1


def test_state_hash_changes_with_input(system):
    base = system.run_at("2024-09-15 20:00:00").recommendation
    moved = system.run_at("2024-09-15 20:00:00",
                          lab_overrides={("HT_FEED", "Mass.Sulfur"): 1.6}).recommendation
    assert base.state_hash != moved.state_hash


def test_trace_is_complete_and_ordered(system):
    res = system.run_at("2025-11-20 14:00:00")
    seqs = [m["seq"] for m in res.trace]
    assert seqs == sorted(seqs)
    senders = {m["sender"] for m in res.trace}
    assert {"orchestrator", "data", "quality", "reliability", "optimizer"} <= senders
