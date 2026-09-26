"""Aggregate the EBC agent experiment (runbook 3.5, 8.2-8.3).

Recomputes every reported number from the raw episode rows only. Nothing is read from a previously
aggregated file, so any table can be rebuilt from `episodes/` + `matched_pools/` alone.

    python -m tasks.agentic_autoresearch.ebc_aggregate --root <...>/agent_results

Writes into --root:
    metrics_by_condition.csv   anytime_metrics.csv        candidate_funnel.csv
    tool_utilization.csv       faithfulness.csv           graph_only_discovery.csv
    paired_comparisons.csv     failure_taxonomy.json      claim_decisions.json

DECLARED PRIMARY CAUSAL COMPARISON, fixed before any inference (see ebc_agent.PRIMARY_COMPARISON):
    correct_traversal_agent > rewired_traversal_agent on macro target Recall@10.
Confirmatory family (Holm-adjusted together):
    1. correct_traversal_agent   > rewired_traversal_agent          [primary]
    2. compact_hybrid_agent      > correct_hybrid_one_shot
    3. each agent condition      > its matched_pool_replay
Everything else is reported unadjusted and labelled secondary.

A stored graph path is retrieval provenance, not semantic proof that the paper supports the
paragraph; `path_faithful` only certifies that the path exists and is legal in the condition's own
pre-cutoff graph.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
from pathlib import Path

import numpy as np

from .ebc_agent import AGENTS, ANYTIME_AT, ONESHOT, PRIMARY_COMPARISON, PROTOCOL
from .evaluate import holm

N_RESAMPLES = 10000
SEED = 20260925
PRIMARY_METRIC = PRIMARY_COMPARISON[2]          # "recall@10"
METRIC_KEY = {"recall@5": "recall_at5", "recall@10": "recall_at10", "set_f1": "set_f1",
              "set_recall": "set_recall", "r_precision": "r_precision", "mrr": "mrr",
              "hit": "hit", "valid": "valid"}
REPORTED = ["recall@10", "recall@5", "set_f1", "set_recall", "r_precision", "mrr", "valid"]

UPTAKE_GATE = 0.90                              # runbook 9.1: below this, comparisons are
#                                                 "Inconclusive uptake", never "Not supported"


# ---------------------------------------------------------------- statistics

def paired_boot(a, b, n_boot=N_RESAMPLES, seed=SEED, alpha=0.05, chunk=1000):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = d.size
    if n == 0:
        return None
    rs = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        means[done:done + m] = d[rs.integers(0, n, size=(m, n))].mean(axis=1)
        done += m
    return dict(diff=float(d.mean()), lo=float(np.quantile(means, alpha / 2)),
                hi=float(np.quantile(means, 1 - alpha / 2)), n=int(n))


def paired_perm_p(a, b, n_perm=N_RESAMPLES, seed=SEED, chunk=1000):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = d.size
    if n == 0:
        return None
    obs = abs(d.mean())
    rs = np.random.default_rng(seed + 1)
    count, done = 0, 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = rs.integers(0, 2, size=(m, n)) * 2 - 1
        count += int((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).sum())
        done += m
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------- io

def load_rows(root, smoke=False):
    rows, seen = [], set()
    subs = ("smoke", "matched_pools_smoke") if smoke else ("episodes", "matched_pools")
    for sub in subs:
        for f in sorted(glob.glob(str(Path(root) / sub / "*" / "*.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                key = (r["model"], r["condition"], r["instance_id"])
                if key in seen:                 # duplicate keys must never be double-counted
                    continue
                seen.add(key)
                rows.append(r)
    return rows


def wr(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def paired(a, b, metric):
    """Strictly paired on instance_id; identical ids in both arms."""
    key = METRIC_KEY[metric]
    ka = {r["instance_id"]: r for r in a}
    kb = {r["instance_id"]: r for r in b}
    ids = sorted(set(ka) & set(kb))
    return ids, [ka[i]["metrics"][key] for i in ids], [kb[i]["metrics"][key] for i in ids]


def compare(a, b, metric, hi, lo, model, family):
    ids, xa, xb = paired(a, b, metric)
    if not ids:
        return None
    ci = paired_boot(xa, xb)
    return dict(model=model, higher=hi, lower=lo, metric=metric, n=len(ids),
                mean_higher=round(mean(xa), 4), mean_lower=round(mean(xb), 4),
                diff=round(ci["diff"], 4), ci_lo=round(ci["lo"], 4), ci_hi=round(ci["hi"], 4),
                p_raw=paired_perm_p(xa, xb), family=family, n_boot=N_RESAMPLES, boot_seed=SEED,
                is_primary=bool((hi, lo, metric) == PRIMARY_COMPARISON))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="agent_results directory")
    ap.add_argument("--smoke", action="store_true", help="aggregate the DEV smoke rows instead")
    a = ap.parse_args()
    root = Path(a.root)
    rows = load_rows(root, smoke=a.smoke)
    if not rows:
        print("no episode rows found under", root)
        return 1

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["model"], r["condition"])].append(r)
    models = sorted({r["model"] for r in rows})
    suffix = "_smoke" if a.smoke else ""

    # ---- invariant checks recomputed from the rows themselves ----------------
    inv = {}
    for m in models:
        c, w = by[(m, "correct_traversal_agent")], by[(m, "rewired_traversal_agent")]
        if c and w:
            hc = {r["instance_id"]: r["prompt_schema_hash"] for r in c}
            hw = {r["instance_id"]: r["prompt_schema_hash"] for r in w}
            shared = sorted(set(hc) & set(hw))
            inv[m] = dict(
                paired_n=len(shared),
                prompt_schema_identical=all(hc[i] == hw[i] for i in shared),
                graph_hashes_differ=len({r["graph_variant_hash"] for r in c}
                                        | {r["graph_variant_hash"] for r in w}) == 2,
                degree_hashes_match=({r["graph_degree_hash"] for r in c}
                                     == {r["graph_degree_hash"] for r in w}),
                observations_not_universally_identical=any(
                    {r["instance_id"]: r["matched_pool_hash"] for r in c}.get(i)
                    != {r["instance_id"]: r["matched_pool_hash"] for r in w}.get(i)
                    for i in shared))

    # ---- metrics by condition ------------------------------------------------
    mrows = []
    for (m, c), v in sorted(by.items()):
        hard = [r for r in v if r.get("graphhard")]
        mrows.append([m, c, len(v)]
                     + [round(mean(r["metrics"][METRIC_KEY[k]] for r in v), 4) for k in REPORTED]
                     + [round(mean(1 - r["metrics"]["valid"] for r in v), 4),
                        round(mean(r["latency_ms"] for r in v), 1),
                        len(hard) or None])
    wr(root / f"metrics_by_condition{suffix}.csv",
       ["model", "condition", "n"] + REPORTED + ["unanswered_rate", "mean_latency_ms",
                                                 "n_graphhard"], mrows)

    # ---- candidate funnel ----------------------------------------------------
    frows = []
    for (m, c), v in sorted(by.items()):
        n_t = sum(r["n_targets"] for r in v) or 1
        frows.append([m, c, len(v), n_t,
                      round(sum(r["n_targets_observed"] for r in v) / n_t, 4),
                      round(sum(len(r.get("targets_opened") or []) for r in v) / n_t, 4),
                      round(sum(r["n_targets_selected"] for r in v) / n_t, 4),
                      round(mean(len(r["observed_candidates"]) for r in v), 2),
                      round(mean(r["observation_tokens"] for r in v), 1),
                      round(mean([r["calls_to_first_target"] for r in v
                                  if r.get("calls_to_first_target") is not None] or [0]), 3),
                      sum(1 for r in v if r.get("calls_to_first_target") is None),
                      sum(r["truncation_events"] for r in v)])
    wr(root / f"candidate_funnel{suffix}.csv",
       ["model", "condition", "n", "n_targets", "target_ever_observed", "target_opened",
        "target_selected", "mean_observed_candidates", "mean_observation_tokens",
        "mean_calls_to_first_target", "episodes_never_observing_a_target",
        "truncation_events"], frows)

    # ---- graph-only target discovery: found here, not observed by the flat agent ----
    gorows = []
    for m in models:
        flat = {r["instance_id"]: set(r.get("targets_observed") or [])
                for r in by[(m, "flat_controlled_agent")]}
        for c in ("correct_traversal_agent", "rewired_traversal_agent", "compact_hybrid_agent"):
            v = by[(m, c)]
            if not v:
                continue
            n_only = sum(len(set(r.get("targets_observed") or []) - flat.get(r["instance_id"], set()))
                         for r in v)
            eps = sum(1 for r in v
                      if set(r.get("targets_observed") or []) - flat.get(r["instance_id"], set()))
            sel_only = sum(len({t for t in (r.get("targets_selected") or [])}
                               - flat.get(r["instance_id"], set())) for r in v)
            n_t = sum(r["n_targets"] for r in v) or 1
            gorows.append([m, c, len(v), n_t, n_only, round(n_only / n_t, 4),
                           eps, round(eps / len(v), 4), sel_only, round(sel_only / n_t, 4)])
    wr(root / f"graph_only_discovery{suffix}.csv",
       ["model", "condition", "n", "n_targets", "targets_observed_not_by_flat",
        "rate_of_targets", "episodes_with_graph_only_target", "rate_of_episodes",
        "targets_selected_not_observed_by_flat", "selected_rate"], gorows)

    # ---- tool utilization ----------------------------------------------------
    trows = []
    for (m, c), v in sorted(by.items()):
        if not any(r["tool_calls"] for r in v):
            continue
        trows.append([m, c, len(v),
                      round(mean(len(r["tool_calls"]) for r in v), 3),
                      round(mean(r["calls_succeeded"] for r in v), 3),
                      round(mean(r["calls_succeeded"] >= 1 for r in v), 4),
                      sum(r["calls_attempted"] for r in v), sum(r["calls_parsed"] for r in v),
                      sum(r["calls_dispatched"] for r in v), sum(r["calls_succeeded"] for r in v),
                      sum(r["blocked_calls"] for r in v), sum(r["failed_calls"] for r in v),
                      sum(r["malformed_actions"] for r in v), sum(r["repairs"] for r in v),
                      sum(r["premature_finalize_refusals"] for r in v),
                      sum(bool(r.get("forced_final")) for r in v),
                      round(mean(r["calls_parsed"] / max(1, r["calls_attempted"]) for r in v), 4),
                      round(mean(r["uptake"].get("traverse", 0) >= 1 for r in v), 4),
                      round(mean(r["uptake"].get("hybrid_search", 0) >= 1 for r in v), 4),
                      round(mean(r["uptake"].get("search", 0) >= 1 for r in v), 4)])
    wr(root / f"tool_utilization{suffix}.csv",
       ["model", "condition", "n", "mean_tool_calls", "mean_successful_calls",
        "episodes_with_a_successful_call", "attempted", "parsed", "dispatched", "succeeded",
        "blocked", "failed", "malformed", "repairs", "premature_finalize_refusals",
        "forced_final", "action_parser_validity", "traverse_uptake", "hybrid_uptake",
        "search_uptake"], trows)

    # ---- faithfulness + mutually exclusive failure categories (runbook 8.2) ---
    frows2, tax = [], collections.defaultdict(collections.Counter)
    for (m, c), v in sorted(by.items()):
        for r in v:
            t = set(r["targets"])
            if not r["valid"]:
                cat = "invalid_answer_or_parser_failure"
            elif t & set(r["ranked"]):
                cat = ("recovered_and_faithful" if r.get("evidence_faithful")
                       else "recovered_cites_unseen")
            elif not (t & set(r["observed_candidates"])):
                cat = "no_target_ever_retrieved"
            elif t & set(r.get("targets_opened") or []):
                cat = "target_opened_but_not_selected"
            else:
                cat = "target_observed_but_not_selected"
            tax[(m, c)][cat] += 1
        cited = [r for r in v if r.get("cited_evidence")]
        withpath = [r for r in v if r.get("retrieval_provenance_path")]
        frows2.append([m, c, len(v), len(cited),
                       sum(1 for r in cited if r.get("evidence_faithful")),
                       sum(1 for r in cited if r.get("evidence_faithful") is False),
                       round(mean(bool(r.get("evidence_faithful")) for r in cited), 4) if cited else None,
                       len(withpath),
                       sum(len(r["retrieval_provenance_path"]) for r in withpath),
                       sum(1 for r in withpath for p in r["retrieval_provenance_path"]
                           if p.get("path_verified_in_condition_graph")),
                       round(mean(bool(r.get("path_faithful")) for r in withpath), 4) if withpath else None])
    wr(root / f"faithfulness{suffix}.csv",
       ["model", "condition", "n", "n_with_cited_evidence", "evidence_faithful",
        "evidence_unfaithful", "evidence_faithful_rate", "n_with_graph_path",
        "graph_derived_targets", "paths_verified_in_condition_graph", "path_faithful_rate"], frows2)
    json.dump({f"{m}|{c}": dict(v) for (m, c), v in tax.items()},
              open(root / f"failure_taxonomy{suffix}.json", "w"), indent=1)

    # ---- anytime curves (never back-filled: only rows the agent actually emitted) ----
    arows = []
    for (m, c), v in sorted(by.items()):
        for k in (str(x) for x in ANYTIME_AT):
            got = [r["anytime"][k] for r in v if r.get("anytime", {}).get(k)]
            if not got:
                continue
            arows.append([m, c, k, len(got),
                          round(mean(x["metrics"]["recall_at10"] for x in got), 4),
                          round(mean(x["metrics"]["recall_at5"] for x in got), 4),
                          round(mean(x["metrics"]["set_f1"] for x in got), 4),
                          round(mean(x["metrics"]["mrr"] for x in got), 4),
                          round(mean(x["n_targets_observed"] > 0 for x in got), 4)])
    wr(root / f"anytime_metrics{suffix}.csv",
       ["model", "condition", "after_n_calls", "n", "recall@10", "recall@5", "set_f1", "mrr",
        "any_target_observed_rate"], arows)

    # ---- paired comparisons --------------------------------------------------
    out = []
    for m in models:
        fam = []
        c1 = compare(by[(m, PRIMARY_COMPARISON[0])], by[(m, PRIMARY_COMPARISON[1])],
                     PRIMARY_METRIC, PRIMARY_COMPARISON[0], PRIMARY_COMPARISON[1], m,
                     "confirmatory")
        if c1:
            fam.append(c1)
        c2 = compare(by[(m, "compact_hybrid_agent")], by[(m, "correct_hybrid_one_shot")],
                     PRIMARY_METRIC, "compact_hybrid_agent", "correct_hybrid_one_shot", m,
                     "confirmatory")
        if c2:
            fam.append(c2)
        for c in AGENTS:
            x = compare(by[(m, c)], by[(m, f"matched_pool_replay::{c}")], PRIMARY_METRIC,
                        c, f"matched_pool_replay::{c}", m, "confirmatory")
            if x:
                fam.append(x)
        adj = holm({f"{x['higher']}>{x['lower']}@{x['metric']}": x["p_raw"] for x in fam})
        for x in fam:
            k = f"{x['higher']}>{x['lower']}@{x['metric']}"
            x["p_adj"] = adj[k]["p_adj"]
            x["significant"] = bool(adj[k]["reject_0_05"] and x["ci_lo"] > 0)
        out += fam
        # secondary, unadjusted, reported as such
        sec = [("correct_traversal_agent", "flat_controlled_agent"),
               ("compact_hybrid_agent", "flat_controlled_agent"),
               ("compact_hybrid_agent", "correct_traversal_agent"),
               ("correct_traversal_agent", "bm25_one_shot"),
               ("correct_hybrid_one_shot", "bm25_one_shot")]
        for hi, lo in sec:
            for k in (PRIMARY_METRIC, "set_f1", "mrr"):
                x = compare(by[(m, hi)], by[(m, lo)], k, hi, lo, m, "secondary_unadjusted")
                if x:
                    x["p_adj"] = x["p_raw"]
                    x["significant"] = bool(x["p_raw"] < 0.05 and x["ci_lo"] > 0)
                    out.append(x)
        # the primary contrast on every reported metric, secondary/unadjusted
        for k in REPORTED:
            if k == PRIMARY_METRIC:
                continue
            x = compare(by[(m, PRIMARY_COMPARISON[0])], by[(m, PRIMARY_COMPARISON[1])], k,
                        PRIMARY_COMPARISON[0], PRIMARY_COMPARISON[1], m, "secondary_unadjusted")
            if x:
                x["p_adj"] = x["p_raw"]
                x["significant"] = bool(x["p_raw"] < 0.05 and x["ci_lo"] > 0)
                out.append(x)
    cols = ["model", "higher", "lower", "metric", "n", "mean_higher", "mean_lower", "diff",
            "ci_lo", "ci_hi", "p_raw", "p_adj", "family", "is_primary", "significant",
            "n_boot", "boot_seed"]
    wr(root / f"paired_comparisons{suffix}.csv", cols, [[x.get(c) for c in cols] for x in out])

    # ---- claim decisions -----------------------------------------------------
    dec = {}
    for m in models:
        v = by[(m, PRIMARY_COMPARISON[0])]
        uptake = mean(r["uptake"].get("traverse", 0) >= 1 for r in v) if v else 0.0
        prim = next((x for x in out if x["model"] == m and x["is_primary"]), None)
        if uptake < UPTAKE_GATE:
            conn = "Inconclusive uptake"
        elif prim and prim["significant"]:
            conn = "Supported"
        else:
            conn = "Not supported"
        rep = next((x for x in out if x["model"] == m and x["higher"] == "compact_hybrid_agent"
                    and x["lower"].startswith("matched_pool_replay")), None)
        one = next((x for x in out if x["model"] == m and x["higher"] == "compact_hybrid_agent"
                    and x["lower"] == "correct_hybrid_one_shot"), None)
        iface = next((x for x in out if x["model"] == m
                      and x["higher"] == "compact_hybrid_agent"
                      and x["lower"] == "correct_traversal_agent"), None)
        dec[m] = dict(
            traversal_uptake=round(uptake, 4),
            graph_supports_evidence_bundle_completion=conn,
            sequential_interaction_beats_matched_pool=(
                "Supported" if (rep and rep["significant"]) else "Not supported"),
            compact_hybrid_beats_one_shot_hybrid=(
                "Supported" if (one and one["significant"]) else "Not supported"),
            compiled_hybrid_beats_raw_traversal=(
                "Supported" if (iface and iface["significant"]) else "Not supported"),
            invariants=inv.get(m))
    json.dump(dict(
        protocol=PROTOCOL, primary_comparison=(f"{PRIMARY_COMPARISON[0]} > "
                                               f"{PRIMARY_COMPARISON[1]} on macro target "
                                               f"{PRIMARY_METRIC}"),
        declared_before_inference=True, n_resamples=N_RESAMPLES, seed=SEED,
        uptake_gate=UPTAKE_GATE,
        note=("Explicitly post-hoc controlled protocol with a disclosed mandatory minimum research "
              "phase. Low traversal uptake is reported as inconclusive uptake, never as evidence "
              "against graph utility. A stored graph path is retrieval provenance, not semantic "
              "proof that the paper supports the paragraph."),
        by_model=dec), open(root / f"claim_decisions{suffix}.json", "w"), indent=1)

    print(f"rows={len(rows)} model_condition_cells={len(by)} comparisons={len(out)}")
    print(f"{'model':16s} {'condition':34s} {'n':>4s} " +
          " ".join(f"{k:>10s}" for k in REPORTED))
    for r in mrows:
        print(f"{r[0]:16s} {r[1]:34s} {r[2]:4d} " +
              " ".join(f"{x:10.4f}" for x in r[3:3 + len(REPORTED)]))
    for m, d in dec.items():
        print(f"\n[{m}] traversal_uptake={d['traversal_uptake']} "
              f"graph_supports_EBC={d['graph_supports_evidence_bundle_completion']}")
        print("  invariants:", json.dumps(d["invariants"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
