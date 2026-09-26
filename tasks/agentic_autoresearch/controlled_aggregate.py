"""Aggregate controlled_interaction_v1 Task-A results (runbook 2.3-2.5, 8.2-8.3).

Recomputes every number from the raw episode rows only. Writes:
  metrics_by_condition.csv  anytime_metrics.csv  candidate_funnel.csv
  tool_utilization.csv      faithfulness.csv     paired_comparisons.csv
  claim_decisions.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
from pathlib import Path

from .evaluate import paired_bootstrap_ci, paired_permutation_p, holm

N_RESAMPLES = 10000
SEED = 20260925
PRIMARY = "top1"
AGENTS = ["flat_controlled_agent", "correct_graph_controlled_agent",
          "rewired_graph_controlled_agent", "compact_hybrid_controlled_agent"]
# natural-action counterpart of each controlled condition (previous run, different interface)
NATURAL_OF = {"flat_controlled_agent": "flat_search_agent",
              "correct_graph_controlled_agent": "correct_connectivity_agent",
              "rewired_graph_controlled_agent": "rewired_connectivity_agent",
              "compact_hybrid_controlled_agent": "compact_hybrid_agent"}


def load_rows(root):
    rows = []
    for sub in ("episodes", "matched_pools"):
        for f in glob.glob(str(Path(root) / sub / "*" / "*.jsonl")):
            rows += [json.loads(l) for l in open(f)]
    return rows


def load_natural(path):
    rows = []
    for f in glob.glob(str(Path(path) / "episodes" / "*" / "*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    return rows


def wr(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def paired(a, b, key):
    """Strictly paired on instance_id; returns (ids, xa, xb)."""
    ka = {r["instance_id"]: r for r in a}
    kb = {r["instance_id"]: r for r in b}
    ids = sorted(set(ka) & set(kb))
    return ids, [ka[i]["metrics"][key] for i in ids], [kb[i]["metrics"][key] for i in ids]


def compare(a, b, key, hi, lo, model, family):
    ids, xa, xb = paired(a, b, key)
    if not ids:
        return None
    diff = mean(xa) - mean(xb)
    ci = paired_bootstrap_ci(xa, xb, n_boot=N_RESAMPLES, seed=SEED)
    lo_ci, hi_ci = ci["lo"], ci["hi"]
    p = paired_permutation_p(xa, xb, n_perm=N_RESAMPLES, seed=SEED)
    return dict(model=model, higher=hi, lower=lo, metric=key, n=len(ids),
                mean_higher=round(mean(xa), 4), mean_lower=round(mean(xb), 4),
                diff=round(diff, 4), ci_lo=round(lo_ci, 4), ci_hi=round(hi_ci, 4),
                p_raw=p, family=family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--natural", default="results/agentic_autoresearch/20260924_182602_a26e2e")
    a = ap.parse_args()
    root = Path(a.root)
    rows = load_rows(root)
    nat = load_natural(Path(a.natural)) if Path(a.natural).exists() else []

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["model"], r["condition"])].append(r)
    natby = collections.defaultdict(list)
    for r in nat:
        natby[(r["model"], r["condition"])].append(r)
    models = sorted({r["model"] for r in rows})

    # ---- metrics by condition -------------------------------------------------
    mrows = []
    for (m, c), v in sorted(by.items()):
        mrows.append([m, c, len(v),
                      round(mean(r["metrics"]["top1"] for r in v), 4),
                      round(mean(r["metrics"]["mrr"] for r in v), 4),
                      round(mean(r["metrics"]["recall_at5"] for r in v), 4),
                      round(mean(r["metrics"]["valid"] for r in v), 4),
                      round(mean(1 - r["metrics"]["valid"] for r in v), 4),
                      round(mean(r["latency_ms"] for r in v), 1)])
    wr(root / "metrics_by_condition.csv",
       ["model", "condition", "n", "top1", "mrr", "recall_at5", "validity", "unanswered_rate",
        "mean_latency_ms"], mrows)

    # ---- candidate funnel -----------------------------------------------------
    frows = []
    for (m, c), v in sorted(by.items()):
        gold_sel = mean(r["gold"][0] in r["ranked"] for r in v)
        frows.append([m, c, len(v),
                      round(mean(r["gold_observed"] for r in v), 4),
                      round(mean(r.get("gold_opened", False) for r in v), 4),
                      round(gold_sel, 4),
                      round(mean(len(r["observed_candidates"]) for r in v), 2),
                      round(mean(r["observation_tokens"] for r in v), 1),
                      mean([r["calls_to_first_gold"] for r in v
                            if r.get("calls_to_first_gold") is not None] or [0]),
                      sum(r["truncation_events"] for r in v)])
    wr(root / "candidate_funnel.csv",
       ["model", "condition", "n", "gold_ever_observed", "gold_opened", "gold_selected",
        "mean_observed_candidates", "mean_observation_tokens", "mean_calls_to_first_gold",
        "truncation_events"], frows)

    # ---- graph-only gold discovery: gold observed here but not in the flat condition ----
    gorows = []
    for m in models:
        flat = {r["instance_id"] for r in by[(m, "flat_controlled_agent")] if r["gold_observed"]}
        for c in ("correct_graph_controlled_agent", "rewired_graph_controlled_agent",
                  "compact_hybrid_controlled_agent"):
            v = by[(m, c)]
            if not v:
                continue
            only = [r["instance_id"] for r in v if r["gold_observed"] and r["instance_id"] not in flat]
            gorows.append([m, c, len(v), len(only), round(len(only) / len(v), 4)])
    wr(root / "graph_only_discovery.csv",
       ["model", "condition", "n", "gold_found_not_found_by_flat", "rate"], gorows)

    # ---- tool utilization -----------------------------------------------------
    trows = []
    for (m, c), v in sorted(by.items()):
        if not any(r["tool_calls"] for r in v):
            continue
        up = collections.Counter()
        for r in v:
            for k, n in r.get("uptake", {}).items():
                up[k] += n
        trows.append([m, c, len(v),
                      round(mean(len(r["tool_calls"]) for r in v), 3),
                      round(mean(r["calls_succeeded"] for r in v), 3),
                      sum(r["calls_attempted"] for r in v), sum(r["calls_parsed"] for r in v),
                      sum(r["calls_dispatched"] for r in v), sum(r["calls_succeeded"] for r in v),
                      sum(r["blocked_calls"] for r in v), sum(r["failed_calls"] for r in v),
                      sum(r["malformed_actions"] for r in v), sum(r["repairs"] for r in v),
                      sum(r["premature_finalize_refusals"] for r in v),
                      sum(r.get("forced_final", False) for r in v),
                      round(mean(r["uptake"].get("traverse", 0) >= 1 for r in v), 4),
                      round(mean(r["uptake"].get("hybrid_search", 0) >= 1 for r in v), 4)])
    wr(root / "tool_utilization.csv",
       ["model", "condition", "n", "mean_tool_calls", "mean_successful_calls", "attempted",
        "parsed", "dispatched", "succeeded", "blocked", "failed", "malformed", "repairs",
        "premature_finalize_refusals", "forced_final", "traverse_uptake", "hybrid_uptake"], trows)

    # ---- faithfulness / mutually exclusive failure categories (runbook 8.2) ----
    frows2, tax = [], collections.defaultdict(collections.Counter)
    for (m, c), v in sorted(by.items()):
        for r in v:
            g = r["gold"][0]
            if not r["valid"]:
                cat = "invalid_answer_or_parser_failure"
            elif g in r["ranked"][:1]:
                cat = "correct_and_faithful" if r.get("evidence_faithful") else "correct_cites_unseen"
            elif g not in r["observed_candidates"]:
                cat = "gold_absent_from_retrieved"
            elif r.get("gold_opened"):
                cat = "gold_opened_but_not_selected"
            elif g in r["observed_candidates"]:
                cat = "gold_observed_but_not_opened"
            else:
                cat = "wrong_from_adequate_evidence"
            tax[(m, c)][cat] += 1
        n = len(v)
        cited = [r for r in v if r.get("cited_evidence")]
        frows2.append([m, c, n, len(cited),
                       sum(1 for r in cited if r.get("evidence_faithful")),
                       sum(1 for r in cited if r.get("evidence_faithful") is False),
                       round(mean(bool(r.get("evidence_faithful")) for r in cited), 4) if cited else None])
    wr(root / "faithfulness.csv",
       ["model", "condition", "n", "n_with_cited_evidence", "faithful", "unfaithful",
        "faithful_rate"], frows2)
    json.dump({f"{m}|{c}": dict(v) for (m, c), v in tax.items()},
              open(root / "failure_taxonomy.json", "w"), indent=1)

    # ---- anytime metrics (runbook 1.4 / 2.3) ----------------------------------
    arows = []
    for (m, c), v in sorted(by.items()):
        for k in ("1", "2", "4", "8"):
            got = [r["anytime"][k] for r in v if r.get("anytime", {}).get(k)]
            if not got:
                continue
            arows.append([m, c, k, len(got),
                          round(mean(x["metrics"]["top1"] for x in got), 4),
                          round(mean(x["metrics"]["mrr"] for x in got), 4),
                          round(mean(x["metrics"]["recall_at5"] for x in got), 4),
                          round(mean(x["gold_observed"] for x in got), 4)])
    wr(root / "anytime_metrics.csv",
       ["model", "condition", "after_n_calls", "n", "top1", "mrr", "recall_at5",
        "gold_observed_rate"], arows)

    # ---- paired comparisons ---------------------------------------------------
    out = []
    for m in models:
        prim = []
        c1 = compare(by[(m, "correct_graph_controlled_agent")],
                     by[(m, "rewired_graph_controlled_agent")], PRIMARY,
                     "correct_graph_controlled_agent", "rewired_graph_controlled_agent", m, "preregistered")
        c2 = compare(by[(m, "compact_hybrid_controlled_agent")],
                     by[(m, "correct_hybrid_one_shot_ctrl")], PRIMARY,
                     "compact_hybrid_controlled_agent", "correct_hybrid_one_shot_ctrl", m, "preregistered")
        prim += [x for x in (c1, c2) if x]
        for c in AGENTS:                                    # vs matched-pool replay
            x = compare(by[(m, c)], by[(m, f"matched_pool_replay::{c}")], PRIMARY,
                        c, f"matched_pool_replay::{c}", m, "preregistered")
            if x:
                prim.append(x)
        for c in AGENTS:                                    # controlled vs natural interface
            x = compare(by[(m, c)], natby[(m, NATURAL_OF[c])], PRIMARY,
                        c, f"natural::{NATURAL_OF[c]}", m, "preregistered")
            if x:
                prim.append(x)
        adj = holm({f"{x['higher']}>{x['lower']}": x["p_raw"] for x in prim})
        for x in prim:
            x["p_adj"] = adj[f"{x['higher']}>{x['lower']}"]["p_adj"]
            x["significant"] = x["p_adj"] < 0.05 and x["ci_lo"] > 0
        out += prim
        # secondary, unadjusted
        sec = [("correct_graph_controlled_agent", "flat_controlled_agent"),
               ("compact_hybrid_controlled_agent", "flat_controlled_agent"),
               ("compact_hybrid_controlled_agent", "correct_graph_controlled_agent"),
               ("correct_graph_controlled_agent", "bm25_one_shot_ctrl")]
        for hi, lo in sec:
            for k in (PRIMARY, "mrr"):
                x = compare(by[(m, hi)], by[(m, lo)], k, hi, lo, m, "secondary_unadjusted")
                if x:
                    x["p_adj"], x["significant"] = x["p_raw"], (x["p_raw"] < 0.05 and x["ci_lo"] > 0)
                    out.append(x)
    cols = ["model", "higher", "lower", "metric", "n", "mean_higher", "mean_lower", "diff",
            "ci_lo", "ci_hi", "p_raw", "p_adj", "family", "significant"]
    wr(root / "paired_comparisons.csv", cols, [[x.get(c) for c in cols] for x in out])

    # ---- claim decisions (runbook 2.5) ----------------------------------------
    dec = {}
    for m in models:
        v = by[(m, "correct_graph_controlled_agent")]
        uptake = mean(r["uptake"].get("traverse", 0) >= 1 for r in v) if v else 0.0
        c1 = next((x for x in out if x["model"] == m and x["higher"] == "correct_graph_controlled_agent"
                   and x["lower"] == "rewired_graph_controlled_agent"), None)
        if uptake < 0.90:
            conn = "Inconclusive uptake"
        elif c1 and c1["significant"]:
            conn = "Supported"
        else:
            conn = "Not supported"
        c6r = next((x for x in out if x["model"] == m
                    and x["higher"] == "compact_hybrid_controlled_agent"
                    and x["lower"].startswith("matched_pool_replay")), None)
        seq = "Supported" if (c6r and c6r["significant"]) else "Not supported"
        c6h = next((x for x in out if x["model"] == m
                    and x["higher"] == "compact_hybrid_controlled_agent"
                    and x["lower"] == "correct_graph_controlled_agent"), None)
        iface = "Supported" if (c6h and c6h["significant"]) else "Not supported"
        dec[m] = dict(traversal_uptake=round(uptake, 4),
                      connectivity_helps_when_traversal_controlled=conn,
                      sequential_interaction_beats_matched_pool=seq,
                      compiled_hybrid_beats_raw_traversal=iface)
    json.dump(dict(protocol="controlled_interaction_v1", note=(
        "Explicitly post-hoc protocol with a disclosed mandatory minimum research phase. "
        "Not preregistered before the natural-action run."), by_model=dec),
        open(root / "claim_decisions.json", "w"), indent=1)

    print(f"rows={len(rows)} conditions={len(by)} comparisons={len(out)}")
    for r in mrows:
        print("  ", r[:7])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
