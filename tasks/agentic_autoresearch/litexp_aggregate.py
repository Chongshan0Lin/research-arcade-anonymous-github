"""Aggregate the Literature Set Expansion snowballing agent experiment (runbook §4.4, §8.2-8.3).

Recomputes every reported number from the raw episode rows only. Nothing is read from a previously
aggregated file, so any table can be rebuilt from `episodes/` + `matched_pools/` alone.

    python -m tasks.agentic_autoresearch.litexp_aggregate --root <...>/agent_results

Writes into --root:
    metrics_by_condition.csv   anytime_metrics.csv       candidate_funnel.csv
    tool_utilization.csv       faithfulness.csv          graph_only_discovery.csv
    recall_cost_curve.csv      unsupported_inclusions.csv
    rationale_faithfulness.csv paired_comparisons.csv
    failure_taxonomy.json      claim_decisions.json

DECLARED PRIMARY CAUSAL COMPARISON, fixed before any inference
(see `litexp_agent.PRIMARY_COMPARISON`):
    correct_snowballing_agent > rewired_snowballing_agent on Recall@50.

Confirmatory family (Holm-adjusted together, and ONLY these):
    1. correct_snowballing_agent        > rewired_snowballing_agent          [primary]
    2. compact_hybrid_literature_agent  > correct_hybrid_fixed_list
    3. each agent condition             > its matched_pool_replay
Everything else is reported unadjusted and labelled `secondary_unadjusted`.

WEAK SUPERVISION: recall is measured against the source paper's own bibliography, an incomplete
and selective weak relevance label, not exhaustive ground truth. Absolute recall understates true
topical recall; only the paired between-condition differences, which share the identical label set
per instance, are interpretable as system comparisons.

IDS ARE SCORED INDEPENDENTLY OF PROSE QUALITY (runbook §4.4). Rationale text never enters an
effectiveness metric; `rationale_faithfulness.csv` reports only whether the ids a rationale refers
to were actually observed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
from pathlib import Path

import numpy as np

from .evaluate import holm
from .litexp_agent import (AGENTS, ANYTIME_AT, CONDITIONS, CONFIRMATORY_FAMILY, FIXED,
                           PRIMARY_COMPARISON, PROTOCOL, SCORE_KS, SCORING_NOTE, WEAK_LABEL_NOTE)

N_RESAMPLES = 10000
SEED = 20260925
PRIMARY_METRIC = PRIMARY_COMPARISON[2]              # "recall@50"
REPORTED = [f"recall@{k}" for k in SCORE_KS] + [f"ndcg@{k}" for k in SCORE_KS] + \
           ["map_at_50", "hit", "valid"]
UPTAKE_GATE = 0.90        # runbook §9.1: below this, a comparison is "Inconclusive uptake",
#                           never "Not supported"


# ---------------------------------------------------------------- statistics

def paired_boot(a, b, n_boot=N_RESAMPLES, seed=SEED, alpha=0.05, chunk=1000):
    """Paired bootstrap CI of mean(a) - mean(b) over the SAME instances."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = d.size
    if n == 0:
        return None
    rs = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rs.integers(0, n, size=(m, n))
        means[done:done + m] = d[idx].mean(axis=1)
        done += m
    means.sort()
    return dict(diff=float(d.mean()),
                lo=float(means[int((alpha / 2) * n_boot)]),
                hi=float(means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]),
                n=int(n), n_boot=int(n_boot), seed=seed)


def paired_perm_p(a, b, n_perm=N_RESAMPLES, seed=SEED, chunk=1000):
    """Two-sided paired randomisation test (sign flips) on the mean difference."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = d.size
    if n == 0:
        return None
    obs = abs(d.mean())
    rs = np.random.default_rng(seed)
    count, done = 0, 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = rs.integers(0, 2, size=(m, n)) * 2 - 1
        count += int((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).sum())
        done += m
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------- io

def load_rows(root, smoke=False):
    """All episode + matched-pool rows, de-duplicated on the exact resume key."""
    root = Path(root)
    subs = (["smoke", "matched_pools_smoke"] if smoke else ["episodes", "matched_pools"])
    rows, seen, dupes = [], set(), 0
    for sub in subs:
        for f in sorted(glob.glob(str(root / sub / "*" / "*.jsonl"))):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                key = r.get("resume_key") or f"{r.get('model')}|{r.get('condition')}|{r.get('instance_id')}"
                if key in seen:
                    dupes += 1
                    continue
                seen.add(key)
                r["_key"] = key
                rows.append(r)
    return rows, dupes


def wr(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else 0.0


# ---------------------------------------------------------------- comparisons

def paired(by_cond, hi, lo, metric):
    """Values for the instances BOTH conditions completed, in a fixed shared order."""
    A, B = by_cond.get(hi, {}), by_cond.get(lo, {})
    ids = sorted(set(A) & set(B))
    return ids, [A[i]["metrics"].get(metric, 0.0) for i in ids], \
        [B[i]["metrics"].get(metric, 0.0) for i in ids]


def compare(by_cond, hi, lo, metric, model, family):
    ids, a, b = paired(by_cond, hi, lo, metric)
    if not ids:
        return None
    ci = paired_boot(a, b)
    p = paired_perm_p(a, b)
    return dict(model=model, family=family, hi=hi, lo=lo, metric=metric, n_paired=len(ids),
                mean_hi=round(mean(a), 6), mean_lo=round(mean(b), 6),
                diff=round(ci["diff"], 6), ci_lo=round(ci["lo"], 6), ci_hi=round(ci["hi"], 6),
                p_two_sided=p, positive_ci=bool(ci["lo"] > 0),
                ci_excludes_zero=bool(ci["lo"] > 0 or ci["hi"] < 0),
                n_boot=N_RESAMPLES, n_perm=N_RESAMPLES, seed=SEED,
                is_primary=bool((hi, lo, metric) == tuple(PRIMARY_COMPARISON)))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="agent_results directory")
    ap.add_argument("--smoke", action="store_true", help="aggregate the DEV smoke rows instead")
    a = ap.parse_args()
    root = Path(a.root)
    rows, dupes = load_rows(root, a.smoke)
    if not rows:
        print("no episode rows found under", root)
        return 1

    models = sorted({r["model"] for r in rows})
    all_conds = sorted({r["condition"] for r in rows})
    by_model = {m: collections.defaultdict(dict) for m in models}
    for r in rows:
        by_model[r["model"]][r["condition"]][r["instance_id"]] = r

    # ---- metrics by condition (never pooled across models: runbook 8.3)
    mrows, uptake_by = [], {}
    for m in models:
        for c in all_conds:
            rs = list(by_model[m][c].values())
            if not rs:
                continue
            is_agent = c in AGENTS
            used_tool = mean([float(r.get("n_retrieval_calls", 0) > 0) for r in rs]) if is_agent else None
            uptake_by[(m, c)] = used_tool
            d = dict(model=m, condition=c, n=len(rs),
                     family=("agent" if is_agent else "fixed_list" if c in FIXED
                             else "matched_pool_replay"),
                     valid_rate=round(mean([float(r.get("valid", False)) for r in rs]), 4),
                     tool_uptake_rate=(round(used_tool, 4) if used_tool is not None else ""),
                     mean_retrieval_calls=round(mean([r.get("n_retrieval_calls", 0) for r in rs]), 3),
                     mean_review_adds=round(mean([r.get("review_adds", 0) for r in rs]), 3),
                     mean_observation_tokens=round(mean([r.get("observation_tokens", 0) for r in rs]), 1),
                     mean_observed=round(mean([r.get("n_observed", 0) for r in rs]), 2),
                     mean_gold=round(mean([r.get("n_gold", 0) for r in rs]), 2),
                     mean_gold_observed=round(mean([r.get("n_gold_observed", 0) for r in rs]), 3),
                     mean_gold_selected=round(mean([r.get("n_gold_selected", 0) for r in rs]), 3),
                     mean_graph_only_gold=round(mean([r.get("n_graph_only_gold", 0) for r in rs]), 3),
                     mean_padded_ids=round(mean([r.get("padded_ids", 0) for r in rs]), 2),
                     forced_final_rate=round(mean([float(r.get("forced_final", False)) for r in rs]), 4))
            for k in REPORTED:
                d[k] = round(mean([r["metrics"].get(k, 0.0) for r in rs]), 6)
            mrows.append(d)
    wr(root / "metrics_by_condition.csv",
       ["model", "condition", "family", "n", "valid_rate", "tool_uptake_rate"] + REPORTED
       + ["mean_retrieval_calls", "mean_review_adds", "mean_observation_tokens", "mean_observed",
          "mean_gold", "mean_gold_observed", "mean_gold_selected", "mean_graph_only_gold",
          "mean_padded_ids", "forced_final_rate"], mrows)

    # ---- anytime metrics (never back-filled)
    arows = []
    for m in models:
        for c in AGENTS:
            rs = list(by_model[m][c].values())
            for k in ANYTIME_AT:
                vals = [r.get("anytime", {}).get(str(k)) for r in rs]
                vals = [v for v in vals if v]
                if not vals:
                    continue
                arows.append(dict(model=m, condition=c, after_retrieval_calls=k, n=len(vals),
                                  **{mk: round(mean([v["metrics"].get(mk, 0.0) for v in vals]), 6)
                                     for mk in (PRIMARY_METRIC, "recall@20", "recall@10")},
                                  mean_gold_observed=round(
                                      mean([v.get("n_gold_observed", 0) for v in vals]), 3)))
    wr(root / "anytime_metrics.csv",
       ["model", "condition", "after_retrieval_calls", "n", PRIMARY_METRIC, "recall@20",
        "recall@10", "mean_gold_observed"], arows)

    # ---- recall-cost curve: recall per retrieval call and per observation token
    crows = []
    for m in models:
        for c in all_conds:
            rs = list(by_model[m][c].values())
            if not rs:
                continue
            calls = mean([r.get("n_retrieval_calls", 0) for r in rs])
            toks = mean([r.get("observation_tokens", 0) for r in rs])
            rec = mean([r["metrics"].get(PRIMARY_METRIC, 0.0) for r in rs])
            crows.append(dict(model=m, condition=c, n=len(rs), metric=PRIMARY_METRIC,
                              mean_recall=round(rec, 6),
                              mean_retrieval_calls=round(calls, 3),
                              mean_observation_tokens=round(toks, 1),
                              recall_per_retrieval_call=round(rec / calls, 6) if calls else "",
                              recall_per_1k_observation_tokens=round(rec / (toks / 1000), 6)
                              if toks else ""))
    wr(root / "recall_cost_curve.csv",
       ["model", "condition", "n", "metric", "mean_recall", "mean_retrieval_calls",
        "mean_observation_tokens", "recall_per_retrieval_call",
        "recall_per_1k_observation_tokens"], crows)

    # ---- candidate funnel
    frows = []
    for m in models:
        for c in all_conds:
            rs = list(by_model[m][c].values())
            if not rs:
                continue
            ng = sum(r.get("n_gold", 0) for r in rs)
            frows.append(dict(
                model=m, condition=c, n=len(rs), gold_total=ng,
                gold_observed=sum(r.get("n_gold_observed", 0) for r in rs),
                gold_opened=sum(len(set(r.get("gold", [])) & set(r.get("opened", []))) for r in rs),
                gold_selected=sum(r.get("n_gold_selected", 0) for r in rs),
                gold_observed_rate=round(sum(r.get("n_gold_observed", 0) for r in rs) / ng, 4) if ng else "",
                gold_selected_rate=round(sum(r.get("n_gold_selected", 0) for r in rs) / ng, 4) if ng else "",
                mean_calls_to_first_gold=round(mean(
                    [r.get("calls_to_first_gold") for r in rs
                     if r.get("calls_to_first_gold") is not None]), 3)))
    wr(root / "candidate_funnel.csv",
       ["model", "condition", "n", "gold_total", "gold_observed", "gold_opened", "gold_selected",
        "gold_observed_rate", "gold_selected_rate", "mean_calls_to_first_gold"], frows)

    # ---- tool utilisation
    trows = []
    for m in models:
        for c in AGENTS:
            rs = list(by_model[m][c].values())
            if not rs:
                continue
            verbs = sorted({v for r in rs for v in r.get("uptake", {})})
            trows.append(dict(
                model=m, condition=c, n=len(rs),
                attempted=sum(r.get("calls_attempted", 0) for r in rs),
                parsed=sum(r.get("calls_parsed", 0) for r in rs),
                dispatched=sum(r.get("calls_dispatched", 0) for r in rs),
                succeeded=sum(r.get("calls_succeeded", 0) for r in rs),
                blocked=sum(r.get("blocked_calls", 0) for r in rs),
                failed=sum(r.get("failed_calls", 0) for r in rs),
                malformed=sum(r.get("malformed_actions", 0) for r in rs),
                repairs=sum(r.get("repairs", 0) for r in rs),
                premature_finalize_refusals=sum(r.get("premature_finalize_refusals", 0) for r in rs),
                forced_final=sum(int(bool(r.get("forced_final"))) for r in rs),
                any_retrieval_rate=round(mean([float(r.get("n_retrieval_calls", 0) > 0) for r in rs]), 4),
                uptake_by_verb=json.dumps({v: sum(r.get("uptake", {}).get(v, 0) for r in rs)
                                           for v in verbs})))
    wr(root / "tool_utilization.csv",
       ["model", "condition", "n", "attempted", "parsed", "dispatched", "succeeded", "blocked",
        "failed", "malformed", "repairs", "premature_finalize_refusals", "forced_final",
        "any_retrieval_rate", "uptake_by_verb"], trows)

    # ---- faithfulness / unsupported inclusions / rationale faithfulness
    fa, ui, ra = [], [], []
    for m in models:
        for c in all_conds:
            rs = list(by_model[m][c].values())
            if not rs:
                continue
            fa.append(dict(model=m, condition=c, n=len(rs),
                           evidence_faithful_rate=round(mean(
                               [float(r["evidence_faithful"]) for r in rs
                                if r.get("evidence_faithful") is not None]), 4),
                           n_with_evidence=sum(1 for r in rs
                                               if r.get("evidence_faithful") is not None)))
            tot_sel = sum(len(r.get("scored_ranking", [])) for r in rs)
            uns = sum(len(r.get("unsupported_inclusions", [])) for r in rs)
            ui.append(dict(model=m, condition=c, n=len(rs), selected_ids=tot_sel,
                           unsupported_ids=uns,
                           unsupported_rate=round(uns / tot_sel, 6) if tot_sel else "",
                           episodes_with_any_unsupported=sum(
                               1 for r in rs if r.get("unsupported_inclusions"))))
            with_r = [r for r in rs if r.get("rationales")]
            ra.append(dict(model=m, condition=c, n=len(rs), episodes_with_rationales=len(with_r),
                           mean_rationales_per_episode=round(
                               mean([len(r.get("rationales", {})) for r in rs]), 3),
                           rationale_ids_observed_rate=round(mean(
                               [float(r["rationale_ids_observed"]) for r in rs
                                if r.get("rationale_ids_observed") is not None]), 4),
                           note=SCORING_NOTE))
    wr(root / "faithfulness.csv",
       ["model", "condition", "n", "evidence_faithful_rate", "n_with_evidence"], fa)
    wr(root / "unsupported_inclusions.csv",
       ["model", "condition", "n", "selected_ids", "unsupported_ids", "unsupported_rate",
        "episodes_with_any_unsupported"], ui)
    wr(root / "rationale_faithfulness.csv",
       ["model", "condition", "n", "episodes_with_rationales", "mean_rationales_per_episode",
        "rationale_ids_observed_rate", "note"], ra)

    # ---- graph-only discovery: gold first seen through a graph tool
    grows = []
    for m in models:
        for c in AGENTS:
            rs = list(by_model[m][c].values())
            if not rs:
                continue
            uniq = set()
            for r in rs:
                uniq |= set(r.get("graph_only_gold", []))
            grows.append(dict(model=m, condition=c, n=len(rs),
                              mean_graph_only_gold=round(
                                  mean([r.get("n_graph_only_gold", 0) for r in rs]), 4),
                              episodes_with_graph_only_gold=sum(
                                  1 for r in rs if r.get("n_graph_only_gold", 0) > 0),
                              unique_graph_only_gold=len(uniq),
                              graph_backing=rs[0].get("graph_backing") or ""))
    wr(root / "graph_only_discovery.csv",
       ["model", "condition", "n", "mean_graph_only_gold", "episodes_with_graph_only_gold",
        "unique_graph_only_gold", "graph_backing"], grows)

    # ---- failure taxonomy (runbook 8.2): mutually exclusive primary category per episode
    tax = collections.defaultdict(collections.Counter)
    for r in rows:
        g, obs, sel = set(r.get("gold", [])), set(r.get("observed_candidates", [])), \
            set(r.get("scored_ranking", []))
        if r.get("parser_problem") == "parse_failed":
            k = "invalid_answer_parser_failure"
        elif not r.get("valid"):
            k = f"invalid::{r.get('parser_problem')}"
        elif not g:
            k = "no_gold"
        elif not (g & obs):
            k = "gold_absent_from_all_retrieved"
        elif r.get("truncation_events") and not (g & sel):
            k = "gold_retrieved_but_truncated"
        elif (g & obs) and not (g & sel):
            k = "gold_observed_but_not_selected"
        elif g <= sel:
            k = "all_gold_recovered"
        else:
            k = "partially_correct"
        tax[f"{r['model']}|{r['condition']}"][k] += 1
    tax_path = root / "failure_taxonomy.json"
    with open(tax_path, "w") as f:
        json.dump(dict(note=("mutually exclusive primary categories; overlapping diagnostic flags "
                             "are in the per-episode rows"),
                       weak_label_note=WEAK_LABEL_NOTE,
                       taxonomy={k: dict(v) for k, v in tax.items()}), f, indent=1)

    # ---- paired comparisons
    comps = []
    for m in models:
        bc = by_model[m]
        for hi, lo, metric in CONFIRMATORY_FAMILY:
            c = compare(bc, hi, lo, metric, m, "confirmatory")
            if c:
                comps.append(c)
        secondary = []
        for hi, lo, metric in CONFIRMATORY_FAMILY:
            for mk in REPORTED:
                if mk != metric:
                    secondary.append((hi, lo, mk))
        for a_ in AGENTS:
            for f_ in FIXED:
                secondary.append((a_, f_, PRIMARY_METRIC))
        for hi, lo, metric in secondary:
            c = compare(bc, hi, lo, metric, m, "secondary_unadjusted")
            if c:
                comps.append(c)
        # Holm within the confirmatory family only, per model
        pv = {f"{c['hi']}>{c['lo']}@{c['metric']}": c["p_two_sided"]
              for c in comps if c["model"] == m and c["family"] == "confirmatory"}
        adj = holm(pv)
        for c in comps:
            if c["model"] == m and c["family"] == "confirmatory":
                c["p_holm"] = adj[f"{c['hi']}>{c['lo']}@{c['metric']}"]["p_adj"]
                c["significant_holm"] = adj[f"{c['hi']}>{c['lo']}@{c['metric']}"]["reject_0_05"]
            elif c["model"] == m:
                c["p_holm"] = ""
                c["significant_holm"] = ""
    wr(root / "paired_comparisons.csv",
       ["model", "family", "hi", "lo", "metric", "n_paired", "mean_hi", "mean_lo", "diff",
        "ci_lo", "ci_hi", "p_two_sided", "p_holm", "significant_holm", "positive_ci",
        "ci_excludes_zero", "is_primary", "n_boot", "n_perm", "seed"], comps)

    # ---- claim decisions
    claims = {}
    for m in models:
        hi, lo, metric = PRIMARY_COMPARISON
        c = next((x for x in comps if x["model"] == m and x["hi"] == hi and x["lo"] == lo
                  and x["metric"] == metric and x["family"] == "confirmatory"), None)
        up_hi, up_lo = uptake_by.get((m, hi)), uptake_by.get((m, lo))
        low_uptake = (up_hi is not None and up_hi < UPTAKE_GATE) or \
                     (up_lo is not None and up_lo < UPTAKE_GATE)
        if c is None:
            dec = "Not run"
        elif low_uptake:
            dec = "Inconclusive uptake"
        elif c["positive_ci"] and c.get("significant_holm"):
            dec = "Supported"
        elif c["ci_hi"] < 0:
            dec = "Not supported (opposite direction)"
        else:
            dec = "Not supported"
        claims[m] = dict(
            claim="Graph supports literature snowballing",
            scope=f"{hi} vs {lo} on {metric}, model {m}",
            decision=dec, comparison=c,
            tool_uptake=dict(hi=up_hi, lo=up_lo, gate=UPTAKE_GATE,
                             note=("uptake below the gate means correct-vs-rewired equality is a "
                                   "failed intervention-uptake check, NEVER evidence that graph "
                                   "connectivity is useless")),
            weak_label_note=WEAK_LABEL_NOTE, scoring_note=SCORING_NOTE)
    with open(root / "claim_decisions.json", "w") as f:
        json.dump(dict(protocol=PROTOCOL, primary_comparison=list(PRIMARY_COMPARISON),
                       confirmatory_family=[list(x) for x in CONFIRMATORY_FAMILY],
                       holm_scope=("confirmatory family only, per model; never pooled "
                                   "across models"),
                       duplicate_rows_skipped=dupes, models=models,
                       conditions_present=all_conds, claims=claims), f, indent=1)

    print(f"models={models} conditions={len(all_conds)} rows={len(rows)} dup_skipped={dupes}")
    for d in mrows:
        print(f"  {d['model']:28s} {d['condition']:42s} n={d['n']:4d} "
              f"{PRIMARY_METRIC}={d[PRIMARY_METRIC]:.4f} valid={d['valid_rate']:.3f} "
              f"uptake={d['tool_uptake_rate']}")
    for m, cl in claims.items():
        print(f"  CLAIM[{m}]: {cl['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
