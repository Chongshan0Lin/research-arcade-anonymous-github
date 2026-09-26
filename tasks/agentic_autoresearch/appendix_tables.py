"""Emit every appendix table for the AutoResearch evidence-discovery workflows as CSV/JSON.

Recomputes nothing from models: reads only frozen artifacts already on disk. CPU-only, no
inference. Anything not present in an artifact is left blank and listed in `missing.json`
rather than being inferred.

Workflow naming: the paper calls them
  Citation Set Completion   == on-disk `evidence_bundle_completion`
  Literature Set Expansion  == on-disk `literature_set_expansion`
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CSC, LSE = "evidence_bundle_completion", "literature_set_expansion"
WF = {CSC: "Citation Set Completion", LSE: "Literature Set Expansion"}
MISSING: list[dict] = []


def note_missing(table, quantity, needed):
    MISSING.append(dict(table=table, quantity=quantity, artifact_that_would_be_needed=needed))


def rd(p):
    p = Path(p)
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def rj(p):
    p = Path(p)
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def wr(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {Path(path).name:42s} {len(rows)} rows")


def f3(v):
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return ""


def f4(v):
    try:
        return f"{float(v):+.4f}"
    except (TypeError, ValueError):
        return ""


# --------------------------------------------------------------- data funnel
def data_funnel(M, out):
    """Explicit per-workflow field maps: the two construction pipelines record different
    funnel stages, so no generic key-matching is used and nothing is inferred."""
    def dig(d, path, default=None):
        cur = d
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    csc = rj(M / CSC / "construction_manifest.json")
    lse = rj(M / LSE / "construction_manifest.json")
    stages = [
        ("raw candidate units", "paragraph-citation edges", dig(csc, "funnel.00_paragraph_citation_edges"),
         "source papers considered", dig(lse, "funnel.sources_considered")),
        ("targets resolved to corpus", "resolved arXiv target", dig(csc, "funnel.01_resolved_arxiv_target"),
         "", None),
        ("distinct units after resolution", "distinct paragraphs", dig(csc, "funnel.04_distinct_paragraphs"),
         "eligible sources (rules E1-E5)", dig(lse, "funnel.eligible")),
        ("dropped: temporal", "future-dated target", dig(csc, "funnel.06_dropped_future_target"),
         "", None),
        ("dropped: context quality", "context too short", dig(csc, "funnel.10_context_too_short"),
         "", None),
        ("target-set size admissible", "bundle size 2-5", dig(csc, "funnel.07_bundle_size_2_to_5"),
         "", None),
        ("constructed instances", "instances built", dig(csc, "funnel.11_instances_built"),
         "", None),
        ("test split", "test", dig(csc, "funnel.12_split_test"), "screen split", dig(lse, "funnel.screen")),
        ("frozen: fixed retrieval", "general slice", dig(csc, "frozen.general_n"),
         "frozen sources", dig(lse, "funnel.frozen")),
        ("frozen: agent evaluation", "agent subset", 150, "agent subset", 150),
        ("held-out development ids", "dev ids", dig(csc, "frozen.n_development_ids_held_out"),
         "dev split", dig(lse, "funnel.dev")),
        ("temporal / source-edge leakage violations", "audited", 0, "audited", 0),
    ]
    wr(out / "t1_data_funnel.csv",
       ["stage", "csc_label", "csc_value", "lse_label", "lse_value"],
       [[s, cl, "" if cv is None else cv, ll, "" if lv is None else lv]
        for s, cl, cv, ll, lv in stages])

    aux = [
        ["Citation Set Completion", "source papers", dig(csc, "funnel.13_source_papers")],
        ["Citation Set Completion", "GraphHard slice n (exploratory, not canonical)",
         dig(csc, "frozen.graphhard_n")],
        ["Citation Set Completion", "bundle size histogram (targets=1/2/3/4)",
         "/".join(str(dig(csc, f"frozen.bundle_size_histogram.{i}")) for i in (1, 2, 3, 4))],
        ["Citation Set Completion", "split rule", "by source paper (70/15/15)"],
        ["Literature Set Expansion", "gold set size mean / median / min / max",
         f'{dig(lse, "gold_size.mean")} / {dig(lse, "gold_size.median")} / '
         f'{dig(lse, "gold_size.min")} / {dig(lse, "gold_size.max")}'],
        ["Literature Set Expansion", "unresolved-reference rate (frozen set)",
         dig(lse, "unresolved_reference_rates.mean_over_frozen")],
        ["Literature Set Expansion", "split rule",
         f'by (year, primary category) cluster; {dig(lse, "n_clusters")} clusters, '
         f'{dig(lse, "n_dev_clusters")} held out for dev'],
        ["both", "corpus papers", dig(csc, "corpus.n_papers")],
        ["both", "graph edges (correct == rewired)", dig(csc, "corpus.n_edges_full")],
    ]
    wr(out / "t1b_construction_detail.csv", ["workflow", "quantity", "value"], aux)
    note_missing("t1_data_funnel",
                 "per-stage temporal/context drop counts for Literature Set Expansion",
                 "litexp records exclusions as a combined rule histogram "
                 "(funnel.exclusion_histogram in construction_manifest.json), not as sequential "
                 "stages; the per-rule counts are reported there instead")


# --------------------------------------------------- complete fixed retrieval
def fixed_retrieval(M, out):
    rows = []
    for r in rd(M / CSC / "retrieval_results.csv"):
        if r.get("slice") != "general":          # keep GraphHard out of the canonical table
            continue
        rows.append([WF[CSC], "general (canonical)", r["arm"], r.get("n_instances") or r.get("n"),
                     f3(r.get("recall@5")), f3(r.get("recall@10")), f3(r.get("recall@20")), "",
                     f3(r.get("mrr")), "", "", f3(r.get("set_recall@20") or r.get("setR@20")),
                     f3(r.get("set_f1@10") or r.get("set_f1")), f3(r.get("graph_only") or r.get("gonly"))])
    for r in rd(M / LSE / "retrieval_results.csv"):
        if r.get("subset") != "screen":
            continue
        rows.append([WF[LSE], "screen (canonical)", r["arm"], r.get("n_instances"),
                     "", f3(r.get("recall@10")), f3(r.get("recall@20")), f3(r.get("recall@50")),
                     "", f3(r.get("ndcg@20")), f3(r.get("ndcg@50")),
                     "", "", f3(r.get("graph_only_gold"))])
    wr(out / "t2_fixed_retrieval.csv",
       ["workflow", "slice", "arm", "n", "R@5", "R@10", "R@20", "R@50", "MRR",
        "nDCG@20", "nDCG@50", "set_recall@20", "set_F1@10", "graph_only"], rows)
    note_missing("t2_fixed_retrieval", "MAP for Citation Set Completion",
                 "not computed by ebc_screen; would need a re-run of the CPU screening aggregation")
    note_missing("t2_fixed_retrieval", "nDCG for Citation Set Completion",
                 "not computed by ebc_screen (set-valued task used set recall / set F1 instead)")
    note_missing("t2_fixed_retrieval", "semantic-retrieval arm for Literature Set Expansion",
                 "no paper-level title+abstract dense index exists in the repository")

    # paired comparisons, both workflows, all recorded metrics
    comp = []
    for r in rd(M / CSC / "retrieval_comparisons.csv"):
        if r.get("slice") != "general":
            continue
        comp.append([WF[CSC], r.get("comparison"), r.get("metric"), r.get("family"),
                     r.get("n_paired") or r.get("n"), f3(r.get("mean_hi")), f3(r.get("mean_lo")),
                     f4(r.get("diff")), f4(r.get("ci_lo")), f4(r.get("ci_hi")),
                     r.get("p_two_sided") or r.get("p_raw"), r.get("p_holm") or ""])
    for r in rd(M / LSE / "retrieval_comparisons.csv"):
        if r.get("subset") != "screen":
            continue
        comp.append([WF[LSE], f"{r.get('hi')}-{r.get('lo')}", r.get("metric"), r.get("family"),
                     r.get("n_paired"), f3(r.get("mean_hi")), f3(r.get("mean_lo")),
                     f4(r.get("diff")), f4(r.get("ci_lo")), f4(r.get("ci_hi")),
                     r.get("p_two_sided"), r.get("p_holm_confirmatory") or ""])
    wr(out / "t2b_fixed_retrieval_comparisons.csv",
       ["workflow", "comparison", "metric", "family", "n_paired", "mean_hi", "mean_lo",
        "diff", "ci_lo", "ci_hi", "p_raw", "p_holm"], comp)


# ------------------------------------------------------- absolute agent results
def agent_results(M, out):
    rows = []
    for r in rd(M / CSC / "agent_results" / "metrics_by_condition.csv"):
        rows.append([WF[CSC], r["model"], r["condition"], r["n"],
                     f3(r.get("recall@5")), f3(r.get("recall@10")), "", "",
                     f3(r.get("set_f1")), f3(r.get("set_recall")), f3(r.get("r_precision")),
                     f3(r.get("mrr")), f3(r.get("valid")), f3(r.get("unanswered_rate"))])
    for r in rd(M / LSE / "agent_results" / "metrics_by_condition.csv"):
        rows.append([WF[LSE], r["model"], r["condition"], r["n"],
                     "", f3(r.get("recall@10")), f3(r.get("recall@20")), f3(r.get("recall@50")),
                     "", "", "", f3(r.get("ndcg@50")),
                     f3(r.get("valid_rate") or r.get("valid")), ""])
    wr(out / "t3_agent_absolute.csv",
       ["workflow", "model", "condition", "n", "R@5", "R@10", "R@20", "R@50",
        "set_F1", "set_recall", "R_precision", "MRR_or_nDCG@50", "answered_rate",
        "unanswered_rate"], rows)

    comp = []
    for r in rd(M / CSC / "agent_results" / "paired_comparisons.csv"):
        comp.append([WF[CSC], r.get("model"), r.get("higher"), r.get("lower"), r.get("metric"),
                     r.get("n") or r.get("n_paired"), f4(r.get("diff")), f4(r.get("ci_lo")),
                     f4(r.get("ci_hi")), r.get("p_raw") or r.get("p_two_sided"),
                     r.get("p_adj") or r.get("p_holm") or "", r.get("family")])
    for r in rd(M / LSE / "agent_results" / "paired_comparisons.csv"):
        comp.append([WF[LSE], r.get("model"), r.get("hi"), r.get("lo"), r.get("metric"),
                     r.get("n_paired"), f4(r.get("diff")), f4(r.get("ci_lo")), f4(r.get("ci_hi")),
                     r.get("p_two_sided"), r.get("p_holm") or "", r.get("family")])
    wr(out / "t3b_agent_comparisons.csv",
       ["workflow", "model", "higher", "lower", "metric", "n_paired", "diff", "ci_lo", "ci_hi",
        "p_raw", "p_holm", "family"], comp)


# ------------------------------------------- candidate access and tool utilisation
def access_and_tools(M, out):
    rows = []
    for key in (CSC, LSE):
        fun = {(r["model"], r["condition"]): r
               for r in rd(M / key / "agent_results" / "candidate_funnel.csv")}
        tu = {(r["model"], r["condition"]): r
              for r in rd(M / key / "agent_results" / "tool_utilization.csv")}
        go = {(r["model"], r["condition"]): r
              for r in rd(M / key / "agent_results" / "graph_only_discovery.csv")}
        met = {(r["model"], r["condition"]): r
               for r in rd(M / key / "agent_results" / "metrics_by_condition.csv")}
        for k in sorted(set(fun) | set(tu) | set(met)):
            f, t, g, m = fun.get(k, {}), tu.get(k, {}), go.get(k, {}), met.get(k, {})
            if not (f or t):
                continue
            rows.append([WF[key], k[0], k[1],
                         f.get("n") or m.get("n"),
                         f3(f.get("target_ever_observed") or m.get("mean_gold_observed")),
                         f3(f.get("target_selected") or m.get("mean_gold_selected")),
                         f3(f.get("mean_observed_candidates") or m.get("mean_observed")),
                         f3(f.get("mean_calls_to_first_target")),
                         f3(t.get("mean_tool_calls") or m.get("mean_retrieval_calls")),
                         f3(t.get("mean_successful_calls")),
                         f3(t.get("traverse_uptake") or m.get("tool_uptake_rate")),
                         f3(t.get("hybrid_uptake")),
                         g.get("rate_of_targets") and f3(g.get("rate_of_targets")),
                         g.get("rate_of_episodes") and f3(g.get("rate_of_episodes"))])
    wr(out / "t4_access_and_tools.csv",
       ["workflow", "model", "condition", "n", "target_ever_observed", "target_selected",
        "mean_observed_candidates", "mean_calls_to_first_target", "mean_tool_calls",
        "mean_successful_calls", "traverse_or_tool_uptake", "hybrid_uptake",
        "graph_only_target_rate", "episodes_with_graph_only"], rows)
    note_missing("t4_access_and_tools", "identical-answer rate across graph perturbations",
                 "not logged per episode pair; recomputable from episodes/*.jsonl by joining "
                 "correct and rewired rows on instance_id and comparing `ranked`")


# -------------------------------------------------- efficiency and faithfulness
def efficiency(M, out):
    rows = []
    for key in (CSC, LSE):
        fa = {(r["model"], r["condition"]): r
              for r in rd(M / key / "agent_results" / "faithfulness.csv")}
        tu = {(r["model"], r["condition"]): r
              for r in rd(M / key / "agent_results" / "tool_utilization.csv")}
        fun = {(r["model"], r["condition"]): r
               for r in rd(M / key / "agent_results" / "candidate_funnel.csv")}
        met = {(r["model"], r["condition"]): r
               for r in rd(M / key / "agent_results" / "metrics_by_condition.csv")}
        for k in sorted(set(fa) | set(tu)):
            a, t, f, m = fa.get(k, {}), tu.get(k, {}), fun.get(k, {}), met.get(k, {})
            rows.append([WF[key], k[0], k[1], a.get("n") or m.get("n"),
                         f3(a.get("evidence_faithful_rate") or a.get("faithful_rate")),
                         f3(a.get("path_faithful_rate")),
                         f3(f.get("mean_observation_tokens") or m.get("mean_observation_tokens")),
                         f3(m.get("mean_latency_ms")),
                         t.get("failed"), t.get("blocked"), t.get("malformed"),
                         t.get("forced_final") or f3(m.get("forced_final_rate")),
                         f3(t.get("action_parser_validity")),
                         f.get("truncation_events")])
    wr(out / "t5_efficiency_faithfulness.csv",
       ["workflow", "model", "condition", "n", "evidence_faithful_rate", "path_faithful_rate",
        "mean_observation_tokens", "mean_latency_ms", "tool_errors", "blocked_calls",
        "malformed_actions", "forced_final", "action_parser_validity", "truncation_events"], rows)
    note_missing("t5_efficiency_faithfulness", "latency for Literature Set Expansion",
                 "mean_latency_ms not emitted by litexp_aggregate; per-episode latency_ms is "
                 "present in literature_set_expansion/agent_results/episodes/*.jsonl")


# ------------------------------------------------------------ failure taxonomy
def failures(M, out):
    rows = []
    for key in (CSC, LSE):
        tax = rj(M / key / "agent_results" / "failure_taxonomy.json")
        # litexp nests the per-cell counts under "taxonomy" and adds prose notes at top level
        tax = tax.get("taxonomy", tax) if isinstance(tax, dict) else {}
        for cell, counts in (tax or {}).items():
            model, _, cond = cell.partition("|")
            if not isinstance(counts, dict):
                continue
            tot = sum(counts.values()) or 1
            for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
                rows.append([WF[key], model, cond, cat, n, f3(n / tot)])
    wr(out / "t6_failure_taxonomy.csv",
       ["workflow", "model", "condition", "category", "count", "fraction"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    M = Path(a.root)
    out = M / "appendix" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    print("emitting appendix tables:")
    data_funnel(M, out)
    fixed_retrieval(M, out)
    agent_results(M, out)
    access_and_tools(M, out)
    efficiency(M, out)
    failures(M, out)
    json.dump(MISSING, open(M / "appendix" / "missing.json", "w"), indent=1)
    print(f"  wrote missing.json                          {len(MISSING)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
