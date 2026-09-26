import pytest

from tasks.agentic_autoresearch.temporal_store import (TemporalObject, TemporalStore,
                                                       TemporalViolation, normalize_ts)


def test_normalize_ts_forms():
    assert normalize_ts("2024-11-04") == "2024-11-04 00:00:00"
    assert normalize_ts("2024-11-04 07:08:12") == "2024-11-04 07:08:12"
    assert normalize_ts("2024-11-04T07:08:12Z") == "2024-11-04 07:08:12"
    assert normalize_ts("2024-11-04 07:08") == "2024-11-04 07:08:00"
    for bad in [None, "", "nan", "not a date", "11/04/2024"]:
        assert normalize_ts(bad) is None


def test_ordering_is_lexicographic_after_normalization():
    assert normalize_ts("2024-09-30") < normalize_ts("2024-10-01")
    assert normalize_ts("2024-11-04 07:08:12") < normalize_ts("2024-11-04 17:00:00")


@pytest.fixture
def store():
    s = TemporalStore()
    s.add_many([("old_paper", "paper", "2020-01-01"),
                ("same_day", "paper", "2024-11-04 07:08:12"),
                ("future_paper", "paper", "2025-06-01"),
                ("no_ts", "paper", None)])
    return s


def test_visibility(store):
    as_of = "2024-11-04 12:00:00"
    assert store.is_visible("old_paper", as_of)
    assert store.is_visible("same_day", as_of)
    assert not store.is_visible("future_paper", as_of)
    assert not store.is_visible("unknown_id", as_of)


def test_missing_timestamp_is_never_silently_clean(store):
    assert not store.is_visible("no_ts", "2030-01-01")
    assert store.is_unverifiable("no_ts")


def test_boundary_is_inclusive():
    s = TemporalStore()
    s.add(TemporalObject("x", "paper", "2024-11-04 07:08:12"))
    assert s.is_visible("x", "2024-11-04 07:08:12")
    assert not s.is_visible("x", "2024-11-04 07:08:11")


def test_filter_and_assert(store):
    vis, hid = store.filter_ids(["old_paper", "future_paper", "no_ts"], "2024-11-04")
    assert vis == ["old_paper"] and set(hid) == {"future_paper", "no_ts"}
    with pytest.raises(TemporalViolation):
        store.assert_visible("future_paper", "2024-11-04")


def test_audit_counts_violations(store):
    a = store.audit(["old_paper", "future_paper", "nope"], "2024-11-04")
    assert a["future_leakage_violations"] == 1
    assert a["violating_ids"] == ["future_paper"]
    assert a["n_unknown"] == 1


def test_bad_as_of_raises(store):
    with pytest.raises(TemporalViolation):
        store.is_visible("old_paper", "yesterday")
