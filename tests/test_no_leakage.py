"""Отсутствие временной утечки — требование ТЗ, проверяемое машинно.

Утечка невозможна конструктивно: единственный способ получить данные —
``AsOfStore.snapshot(t)``, который фильтрует по ``available_at <= t``.
Эти тесты закрепляют свойство, чтобы его нельзя было сломать правкой.
"""
import pandas as pd
import pytest


def test_snapshot_contains_no_future_lab(store):
    for ts in ["2024-03-15 08:00", "2025-07-01 00:00", "2026-01-10 12:00"]:
        t = pd.Timestamp(ts)
        st = store.snapshot(t)
        for key, reading in st.lab.items():
            assert reading.available_at <= t, (
                f"{key}: анализ с available_at={reading.available_at} попал в срез {t}")
            assert reading.measured_at <= t


def test_snapshot_contains_no_future_telemetry(store):
    t = pd.Timestamp("2025-03-01 06:00")
    st = store.snapshot(t)
    assert st.window.index.max() <= t
    assert all(age >= 0 for age in st.tag_age_min.values() if age != float("inf"))


def test_lims_publication_delay_is_applied(store):
    """ЛИМС становится доступным на 4 часа позже отбора пробы (ответ эксперта)."""
    lims = store.lab[store.lab.source == "LIMS"]
    delay = (lims["available_at"] - lims["measured_at"]).dropna().unique()
    assert len(delay) == 1
    assert pd.Timedelta(delay[0]) == pd.Timedelta(hours=4)


def test_lab_value_just_before_publication_is_invisible(store):
    """За минуту до публикации результата его ещё не видно, через минуту — видно."""
    lims = store.lab[(store.lab.source == "LIMS")
                     & (store.lab.stream == "HT_PRODUCT")
                     & (store.lab.param == "Mg.Sulfur")].sort_values("available_at")
    row = lims.iloc[len(lims) // 2]
    before = store.snapshot(row["available_at"] - pd.Timedelta(minutes=1))
    after = store.snapshot(row["available_at"] + pd.Timedelta(minutes=1))
    r_before = before.lab_value("HT_PRODUCT", "Mg.Sulfur", source="LIMS")
    r_after = after.lab_value("HT_PRODUCT", "Mg.Sulfur", source="LIMS")
    assert r_after is not None and r_after.measured_at == row["measured_at"]
    if r_before is not None:
        assert r_before.measured_at < row["measured_at"]


def test_lims_is_not_shadowed_by_pak(store):
    """Частый ПАК не должен затирать редкий, но контрольный результат ЛИМС."""
    st = store.snapshot("2025-03-01 06:00")
    lims = st.lab_value("HT_PRODUCT", "Mg.Sulfur", source="LIMS")
    pak = st.lab_value("HT_PRODUCT", "Mg.Sulfur", source="PAK")
    assert lims is not None and pak is not None
    assert lims.source == "LIMS" and pak.source == "PAK"


def test_future_access_requires_explicit_method(store):
    """Обращение к будущему возможно только через метод с говорящим именем."""
    assert hasattr(store, "future_value")
    v = store.future_value("H_Q21", pd.Timestamp("2024-05-01 00:00"), 120)
    assert v == v  # не NaN — метод работает, но вызывается только офлайн
