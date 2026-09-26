import pytest

from tasks.agentic_autoresearch.environment import DEFAULT_LIMITS, Episode
from tasks.agentic_autoresearch.policies import MockPolicy


def run(env, inst, condition, limits=None):
    ep = Episode(env, inst, condition, limits=limits)
    pol = MockPolicy()
    pol.reset(ep, "sys", "user")
    while not ep.finished and not ep.out_of_budget():
        tool, args = pol.act(ep)
        obs = ep.call(tool, args)
        pol.observe(tool, args, obs)
    return ep


# ---------------- temporal isolation ----------------

def test_future_notes_are_never_observed(rtloc_env, rtloc_instances):
    inst = rtloc_instances[0]
    ep = Episode(rtloc_env, inst, "typed_graph_agent")
    obs = ep.call("get_thread", {})
    assert obs["status"] == "OK"
    assert "nfuture" not in obs["observation_ids"]
    assert all("POST-CUTOFF" not in n["text"] for n in obs["notes"])
    assert ep.stats["filtered_future_notes"] == 1


def test_future_papers_are_filtered_from_search(lit_env, lit_instances):
    ep = Episode(lit_env, lit_instances[0], "typed_graph_agent")
    obs = ep.call("search_papers", dict(query="graph convolutional networks attention", k=10))
    assert "paper:2501.99999" not in obs["observation_ids"]
    assert ep.stats["filtered_future_papers"] >= 1


def test_get_paper_refuses_future_paper(lit_env, lit_instances):
    ep = Episode(lit_env, lit_instances[0], "typed_graph_agent")
    assert ep.call("get_paper", dict(paper_id="2501.99999"))["status"] == "NOT_AVAILABLE_AT_CUTOFF"


def test_citation_edges_to_future_papers_are_filtered(lit_env, lit_instances):
    ep = Episode(lit_env, lit_instances[0], "typed_graph_agent")
    obs = ep.call("get_citations", dict(paper_id="1609.02907", direction="in"))
    assert "paper:2501.99999" not in [n["node_id"] for n in obs["neighbors"]]


def test_audit_reports_zero_violations_for_a_full_episode(rtloc_env, rtloc_instances):
    ep = run(rtloc_env, rtloc_instances[0], "typed_graph_agent")
    audit = rtloc_env.store.audit(ep.observed_ids, ep.instance["as_of"])
    assert audit["future_leakage_violations"] == 0


# ---------------- condition behaviour ----------------

def test_flat_agent_gets_relations_unavailable_not_bm25_fallback(rtloc_env, rtloc_instances):
    ep = Episode(rtloc_env, rtloc_instances[0], "flat_agent")
    obs = ep.call("get_neighbors", dict(node_id="p3"))
    assert obs["status"] == "RELATIONS_UNAVAILABLE"
    assert obs.get("observation_ids", []) == []
    assert ep.stats["attempted_relation_calls"] == 1
    assert ep.call("search_papers", dict(query="table 1", k=5))["status"] == "OK"


def test_variant_changes_relations_but_not_text_or_gold(rtloc_env, rtloc_instances):
    inst = rtloc_instances[0]
    full = rtloc_env.graphs["full"][inst["instance_id"]]
    for v in ("rewire", "type_shuffle", "untyped", "flat"):
        g = rtloc_env.graphs[v][inst["instance_id"]]
        assert {k: n["text"] for k, n in g["nodes"].items()} == {k: n["text"] for k, n in full["nodes"].items()}
        assert g["gold"] == full["gold"]
    assert rtloc_env.graphs["flat"][inst["instance_id"]]["edges"] == []


def test_typed_neighbors_respect_relation_filter(rtloc_env, rtloc_instances):
    ep = Episode(rtloc_env, rtloc_instances[0], "typed_graph_agent")
    obs = ep.call("get_neighbors", dict(node_id="p3", relation_types=["para_mentions"]))
    assert [n["relation"] for n in obs["neighbors"]] == ["para_mentions"]


# ---------------- budgets, validation, trajectory ----------------

def test_budget_is_enforced(rtloc_env, rtloc_instances):
    ep = Episode(rtloc_env, rtloc_instances[0], "typed_graph_agent",
                 limits=dict(max_tool_calls=2, max_observed_tokens=16000))
    ep.call("get_thread", {})
    ep.call("get_thread", {})
    assert ep.out_of_budget()


def test_token_budget_is_enforced(rtloc_env, rtloc_instances):
    ep = Episode(rtloc_env, rtloc_instances[0], "typed_graph_agent",
                 limits=dict(max_tool_calls=10, max_observed_tokens=10))
    ep.call("get_thread", {})
    assert ep.out_of_budget()


def test_bad_tool_and_bad_arguments_become_error_observations(rtloc_env, rtloc_instances):
    ep = Episode(rtloc_env, rtloc_instances[0], "typed_graph_agent")
    assert ep.call("nonexistent_tool", {})["status"] == "ERROR"
    assert ep.call("search_papers", dict(k=5))["status"] == "ERROR"
    assert ep.stats["tool_errors"] == 2
    assert all(s["error"] for s in ep.trajectory)


def test_invalid_answer_is_rejected_and_episode_continues(rtloc_env, rtloc_instances):
    ep = Episode(rtloc_env, rtloc_instances[0], "typed_graph_agent")
    obs = ep.call("submit_answer", dict(payload=dict(ranked_paragraph_ids=[])))
    assert obs["status"] == "ERROR" and not ep.finished


def test_answer_is_normalized(rtloc_env, rtloc_instances):
    ep = Episode(rtloc_env, rtloc_instances[0], "typed_graph_agent")
    ids = [3] + [f"p{i}" for i in range(4, 13)]        # exactly 10 distinct ids
    ep.call("submit_answer", dict(payload=dict(ranked_paragraph_ids=ids, confidence=5)))
    assert ep.finished
    assert ep.answer["ranked_paragraph_ids"][:2] == ["p3", "p4"]
    assert len(ep.answer["ranked_paragraph_ids"]) == 10
    assert ep.answer["confidence"] == 1.0


def test_short_answer_is_rejected_then_accepted_after_retries(rtloc_env, rtloc_instances):
    ep = Episode(rtloc_env, rtloc_instances[0], "typed_graph_agent")
    short = dict(payload=dict(ranked_paragraph_ids=["p3", "p4"]))
    assert ep.call("submit_answer", short)["status"] == "ERROR" and not ep.finished
    assert ep.call("submit_answer", short)["status"] == "ERROR" and not ep.finished
    assert ep.call("submit_answer", short)["status"] == "ACCEPTED"   # lenient after retries
    assert ep.finished and ep.answer["raw_len"] == 2


def test_literature_answer_must_use_observed_ids(lit_env, lit_instances):
    ep = Episode(lit_env, lit_instances[0], "typed_graph_agent")
    ep.call("search_papers", dict(query="graph convolutional networks", k=10))
    made_up = dict(payload=dict(ranked_paper_ids=[f"9999.{i:05d}" for i in range(5)]))
    assert ep.call("submit_answer", made_up)["status"] == "ERROR"


def test_run_episode_pads_to_required_length(rtloc_env, rtloc_instances):
    from tasks.agentic_autoresearch.run_agent import run_episode
    from tasks.agentic_autoresearch.policies import MockPolicy
    pred, traj = run_episode(rtloc_env, rtloc_env.store, rtloc_instances[0], "typed_graph_agent",
                             MockPolicy)
    assert len(pred["ranked"]) == 10
    assert pred["unique_predictions"] == 10


def test_trajectory_rows_have_required_fields(rtloc_env, rtloc_instances):
    from tasks.agentic_autoresearch.schemas import validate_step
    ep = run(rtloc_env, rtloc_instances[0], "typed_graph_agent")
    assert ep.trajectory and all(validate_step(s) for s in ep.trajectory)
    assert ep.summary()["answered"] is True


def test_mock_policy_completes_every_condition(rtloc_env, rtloc_instances, lit_env, lit_instances):
    for cond in ("flat_agent", "untyped_graph_agent", "typed_graph_agent", "rewired_graph_agent",
                 "type_shuffled_agent"):
        ep = run(rtloc_env, rtloc_instances[0], cond)
        assert ep.answer is not None and ep.calls_used <= DEFAULT_LIMITS["rtloc"]["max_tool_calls"]
        ep2 = run(lit_env, lit_instances[0], cond)
        assert ep2.answer is not None


def test_find_paths_returns_real_path(lit_env, lit_instances):
    # instance L2 (gold = 1706.03762 for source 2401.00002); the 2401.00001 -> 1609.02907 ->
    # 1706.03762 route is not this instance's answer edge, so it must remain traversable
    ep = Episode(lit_env, lit_instances[1], "typed_graph_agent")
    obs = ep.call("find_paths", dict(source_ids=["paper:2401.00001"], target_id="paper:1706.03762",
                                     max_hops=3))
    assert obs["status"] == "OK"
    assert any("paper:1609.02907" in p["nodes"] for p in obs["paths"])


def test_gold_citation_edge_is_never_observable(lit_env, lit_instances):
    """The removed citation is itself an edge; it must be masked in both directions."""
    inst = dict(lit_instances[0])          # 2401.00001 --cites--> 1609.02907 is the gold edge
    ep = Episode(lit_env, inst, "typed_graph_agent")
    out = ep.call("get_citations", dict(paper_id=inst["source_paper_id"], direction="out"))
    assert f"paper:{inst['gold_target_id']}" not in [n["node_id"] for n in out["neighbors"]]
    back = ep.call("get_citations", dict(paper_id=inst["gold_target_id"], direction="in"))
    assert f"paper:{inst['source_paper_id']}" not in [n["node_id"] for n in back["neighbors"]]
    nb = ep.call("get_neighbors", dict(node_id=f"paper:{inst['source_paper_id']}"))
    assert f"paper:{inst['gold_target_id']}" not in [n["node_id"] for n in nb["neighbors"]]
    assert ep.stats["masked_gold_edges_hidden"] >= 3


def test_find_paths_cannot_traverse_the_gold_edge(lit_env, lit_instances):
    inst = lit_instances[0]
    ep = Episode(lit_env, inst, "typed_graph_agent")
    obs = ep.call("find_paths", dict(source_ids=[f"paper:{inst['source_paper_id']}"],
                                     target_id=f"paper:{inst['gold_target_id']}", max_hops=1))
    assert obs["status"] == "NO_PATH_FOUND"


def test_find_paths_cannot_traverse_future_nodes(lit_env, lit_instances):
    """A path must not route through (or return) a paper that did not exist at the cutoff."""
    ep = Episode(lit_env, lit_instances[1], "typed_graph_agent")
    obs = ep.call("find_paths", dict(source_ids=["paper:2501.99999"], target_id="paper:1609.02907",
                                     max_hops=3))
    assert "paper:2501.99999" not in obs["observation_ids"]
    audit = lit_env.store.audit(ep.observed_ids, ep.instance["as_of"])
    assert audit["future_leakage_violations"] == 0


def test_paragraph_nodes_inherit_owner_paper_legality(lit_env, lit_instances):
    lit_env.para_owner = {"para:1": "2501.99999", "para:2": "1609.02907"}
    as_of = lit_instances[0]["as_of"]
    assert lit_env.node_legal("para:2", as_of) is True
    assert lit_env.node_legal("para:1", as_of) is False


def test_materialized_paragraphs_are_registered_with_their_paper_date(lit_env, lit_instances):
    """get_paragraphs must not create 'known but dateless' nodes that the auditor calls violations."""
    lit_env.paragraph_lookup = lambda pid, rng: [dict(paragraph_id=777, section="S", text="t")]
    ep = Episode(lit_env, lit_instances[0], "typed_graph_agent")
    obs = ep.call("get_paragraphs", dict(paper_id="1609.02907"))
    assert obs["observation_ids"] == ["para:777"]
    assert lit_env.store.available_at("para:777") == "2016-09-09 00:00:00"
    assert lit_env.store.audit(ep.observed_ids, ep.instance["as_of"])["future_leakage_violations"] == 0
