import math

from tasks.agentic_autoresearch.evaluate import (claim_decision, hit_at_k, holm, ndcg_at_k,
                                                 oracle_normalized_recall, paired_bootstrap_ci,
                                                 paired_permutation_p, recall_at_k, rr,
                                                 tolerant_recall_at_k, top1)


def test_known_fixture_values():
    ranked = ["p3", "p7", "p1", "p9", "p2"]
    gold = ["p7", "p2", "p5"]
    assert top1(ranked, gold) == 0.0
    assert rr(ranked, gold) == 0.5
    assert hit_at_k(ranked, gold, 5) == 1.0
    assert recall_at_k(ranked, gold, 5) == 2 / 3
    # gold has 3 items but only 2 can fit in top-2 -> oracle-normalized recall corrects for the cap
    assert recall_at_k(ranked, gold, 2) == 1 / 3
    assert oracle_normalized_recall(ranked, gold, 2) == 0.5
    expected = (1 / math.log2(3)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert abs(ndcg_at_k(ranked, gold, 2) - expected) < 1e-12


def test_tolerant_recall_counts_adjacent_paragraphs():
    assert tolerant_recall_at_k(["p6"], ["p7"], 5) == 1.0
    assert tolerant_recall_at_k(["p6"], ["p9"], 5) == 0.0
    assert recall_at_k(["p6"], ["p7"], 5) == 0.0


def test_empty_gold_and_empty_ranking():
    assert recall_at_k([], [], 5) == 0.0
    assert ndcg_at_k([], ["a"], 5) == 0.0
    assert top1([], ["a"]) == 0.0


def test_paired_bootstrap_ci_on_constant_difference():
    a = [1.0] * 40
    b = [0.0] * 40
    ci = paired_bootstrap_ci(a, b, n_boot=500, seed=1)
    assert ci["diff"] == 1.0 and ci["lo"] == 1.0 and ci["hi"] == 1.0


def test_permutation_p_detects_no_difference():
    a = [0.5] * 30
    b = [0.5] * 30
    assert paired_permutation_p(a, b, n_perm=200, seed=1) == 1.0


def test_holm_orders_and_adjusts():
    out = holm({"a": 0.001, "b": 0.04, "c": 0.5})
    assert out["a"]["p_adj"] <= out["b"]["p_adj"] <= out["c"]["p_adj"]
    assert out["a"]["reject_0_05"] is True
    assert out["c"]["reject_0_05"] is False


def _pi(vals):
    return {f"i{j}": {"ndcg@10": v} for j, v in enumerate(vals)}


def test_claim_decision_supported_and_not_supported():
    typed = _pi([1.0] * 40)
    weak = _pi([0.0] * 40)
    d = claim_decision({"typed_graph_agent": typed, "flat_agent": weak,
                        "rewired_graph_agent": weak, "type_shuffled_agent": weak}, "rtloc")
    assert d["decision"] == "Supported"
    d2 = claim_decision({"typed_graph_agent": weak, "flat_agent": typed,
                         "rewired_graph_agent": typed, "type_shuffled_agent": typed}, "rtloc")
    assert d2["decision"] == "Not supported"


def test_claim_decision_mixed_when_only_some_pass():
    typed = _pi([1.0] * 40)
    weak = _pi([0.0] * 40)
    same = _pi([1.0] * 40)
    d = claim_decision({"typed_graph_agent": typed, "flat_agent": weak,
                        "rewired_graph_agent": same, "type_shuffled_agent": same}, "rtloc")
    assert d["decision"] == "Mixed"
