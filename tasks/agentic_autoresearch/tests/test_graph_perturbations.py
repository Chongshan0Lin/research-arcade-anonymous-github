import random

from tasks.agentic_autoresearch.build_graph_variants import (PERTURB, degree_profile, graph_stats,
                                                             rewire, type_shuffle, untyped)

EDGES = [["a", "b", "cites"], ["a", "c", "cites"], ["b", "c", "cites"], ["d", "e", "cites"],
         ["e", "f", "cites"], ["a", "p1", "has_paragraph"], ["b", "p2", "has_paragraph"],
         ["c", "p3", "has_paragraph"], ["p1", "x", "mentions"], ["p2", "x", "mentions"]]


def test_flat_removes_all_edges():
    assert PERTURB["flat"](EDGES, random.Random(0)) == []


def test_untyped_keeps_endpoints_single_type():
    e = untyped(EDGES, random.Random(0))
    assert sorted((a, b) for a, b, _ in e) == sorted((a, b) for a, b, _ in EDGES)
    assert {r for _, _, r in e} == {"connected"}


def test_rewire_preserves_per_type_degrees_and_count():
    r = rewire(EDGES, random.Random(1))
    assert len(r) == len(EDGES)
    assert degree_profile(r) == degree_profile(EDGES)
    s = graph_stats(r)
    assert s["self_loops"] == 0 and s["duplicates"] == 0
    assert s["relation_histogram"] == graph_stats(EDGES)["relation_histogram"]


def test_rewire_actually_changes_edges():
    r = rewire(EDGES, random.Random(2))
    assert {tuple(x) for x in r} != {tuple(x) for x in EDGES}


def test_type_shuffle_preserves_endpoints_and_histogram():
    t = type_shuffle(EDGES, random.Random(3))
    assert sorted((a, b) for a, b, _ in t) == sorted((a, b) for a, b, _ in EDGES)
    assert graph_stats(t)["relation_histogram"] == graph_stats(EDGES)["relation_histogram"]


def test_rewire_is_deterministic_under_seed():
    assert rewire(EDGES, random.Random(7)) == rewire(EDGES, random.Random(7))


def test_rewire_does_not_invent_self_loops_when_none_existed():
    edges = [["n%d" % i, "n%d" % (i + 1), "r"] for i in range(30)]
    r = rewire(edges, random.Random(5))
    assert all(a != b for a, b, _ in r)
