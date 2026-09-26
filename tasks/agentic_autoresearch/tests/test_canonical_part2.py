"""Tests for the canonical Part II experiment: parsing, budgets, replay pools, decisions, gate."""
import hashlib

from tasks.agentic_autoresearch import canonical_agent as ca
from tasks.agentic_autoresearch.canonical_aggregate import decide
from tasks.agentic_autoresearch.final_report import FORBIDDEN, integrity_gate


def _ep():
    inst = dict(instance_id="i1", as_of="2020-01-01", source_paper_id="s1",
                paragraph_global_id=1, masked_text="t", gold_target_id="g")
    return ca.Ep(inst, "flat_search_agent", "m", "run", "hash")


def test_parse_accepts_tool_and_answer():
    assert ca.parse('{"tool":"search","query":"x"}', set())[0] == "search"
    tool, ids, prob = ca.parse('{"ranked_paper_ids":["a","b","c","d","e"]}', set("abcde"))
    assert tool == "submit" and ids == list("abcde") and prob is None


def test_parse_rejects_unseen_ids_and_dupes():
    tool, ids, prob = ca.parse('{"ranked_paper_ids":["a","a","zz"]}', {"a"})
    assert tool == "submit" and ids == ["a"] and prob == "short_answer"
    assert ca.parse("not json", {"a"}) == (None, None, "parser_failure")
    assert ca.parse('{"ranked_paper_ids":["zz"]}', {"a"})[2] == "no_valid_ids"


def test_answer_is_truncated_to_exact_length():
    _t, ids, _p = ca.parse('{"ranked_paper_ids":%s}' % str(list("abcdefgh")).replace("'", '"'),
                           set("abcdefgh"))
    assert len(ids) == ca.ANSWER_K


def test_observation_budget_is_enforced():
    ep = _ep()
    assert ep.add_obs(["a"], "x" * 100) is True
    assert ep.add_obs(["b"], "y" * (ca.BUDGET["max_observation_tokens"] * 4 + 10)) is False
    assert ep.observed == ["a"] and ep.truncations == 1


def test_observed_candidates_are_ordered_and_unique():
    ep = _ep()
    ep.add_obs(["a", "b"], "x")
    ep.add_obs(["b", "c"], "x")
    assert ep.observed == ["a", "b", "c"]


def test_matched_pool_hash_matches_replay_slice():
    ids = [f"p{i}" for i in range(50)]
    ep = _ep()
    ep.add_obs(ids, "x")
    row = ep.row(ids[:5], None, ["g"], True, ids[:5])
    want = hashlib.sha256("|".join(row["observed_candidates"]).encode()).hexdigest()[:16]
    assert row["matched_pool_hash"] == want


def test_row_reports_zero_metrics_for_invalid_answers():
    ep = _ep()
    row = ep.row([], "parser_failure", ["g"], None, [])
    assert row["valid"] is False and row["metrics"]["top1"] == 0.0 and row["metrics"]["valid"] == 0.0


def _cmp(hi, lo, diff, lo_ci, p, metric="top1"):
    return dict(model="m", higher=hi, lower=lo, metric=metric, n=200, diff=diff,
                ci_lo=lo_ci, ci_hi=diff + 0.05, p_adj=p, p_raw=p)


def test_decision_requires_both_pool_and_replay_wins():
    strong = [_cmp("correct_hybrid_one_shot", "bm25_one_shot", .05, .01, .001),
              _cmp("compact_hybrid_agent", "correct_hybrid_one_shot", .05, .01, .001),
              _cmp("compact_hybrid_agent", "matched_pool_replay::compact_hybrid_agent", .04, .01, .01)]
    funnel = {("m", "correct_connectivity_agent"): dict(n=200, relation_succeeded=100)}
    d = decide(strong, funnel, "m")
    assert d["candidate_utility"] == "Supported"
    assert d["agentic_interaction"] == "Supported"

    no_replay = strong[:2] + [_cmp("compact_hybrid_agent",
                                   "matched_pool_replay::compact_hybrid_agent", .00, -.03, .9)]
    assert decide(no_replay, funnel, "m")["agentic_interaction"] == "Candidate-pool gain only"


def test_low_uptake_is_inconclusive_not_negative():
    funnel = {("m", "correct_connectivity_agent"): dict(n=200, relation_succeeded=3)}
    d = decide([_cmp("correct_connectivity_agent", "rewired_connectivity_agent", .0, -.02, .8)],
               funnel, "m")
    assert d["connectivity_dependence_under_interaction"].startswith("Inconclusive")


def test_integrity_gate_catches_placeholders_and_unsupported_claims(tmp_path):
    fails = integrity_gate("a report with --limit 600 in it", {}, {}, "nope", [])
    assert any("--limit 600" in f for f in fails)
    fails = integrity_gate("clean", {"connectivity_retrieval_value": "supported"},
                           {"connectivity_retrieval_value": dict(lo=-0.1, p_two_sided=0.4)},
                           "nope", [])
    assert any("without a positive CI" in f for f in fails)
    assert "--limit 600" in FORBIDDEN


def test_short_answers_are_padded_to_exact_length_not_discarded():
    ep = _ep()
    ep.add_obs(["a", "b", "c", "d", "e", "f"], "x")
    ranked, n_pad = ep.pad(["b", "a"], ep.observed)
    assert len(ranked) == ca.ANSWER_K and ranked[:2] == ["b", "a"] and n_pad == 3


def test_padding_never_duplicates_or_reorders_the_model_prefix():
    ep = _ep()
    ep.add_obs(["x", "y", "z"], "obs")
    ranked, n_pad = ep.pad(["y", "y", "x"], ["x", "y", "z", "w", "v"])
    assert ranked[:2] == ["y", "x"] and len(ranked) == len(set(ranked)) == ca.ANSWER_K
