"""Metrics and paired statistics for both tasks.

Ranking metrics are computed per instance and stored before aggregation, so every reported number
can be recomputed and paired across conditions.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict

from .common import RESULTS, read_jsonl, write_json


# ---------------- ranking metrics ----------------

def rr(ranked, gold):
    for i, x in enumerate(ranked, 1):
        if x in gold:
            return 1.0 / i
    return 0.0


def top1(ranked, gold):
    return float(bool(ranked) and ranked[0] in gold)


def recall_at_k(ranked, gold, k):
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def hit_at_k(ranked, gold, k):
    return float(bool(set(ranked[:k]) & set(gold)))


def ndcg_at_k(ranked, gold, k):
    gold = set(gold)
    dcg = sum(1.0 / math.log2(i + 1) for i, x in enumerate(ranked[:k], 1) if x in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


def oracle_normalized_recall(ranked, gold, k):
    """Recall@k divided by the best achievable recall@k (|gold| often exceeds k)."""
    if not gold:
        return 0.0
    best = min(len(gold), k) / len(gold)
    return recall_at_k(ranked, gold, k) / best if best else 0.0


def tolerant_recall_at_k(ranked, gold, k, tol=1):
    """Paragraph-index tolerant recall: p7 counts for gold p6/p7/p8 (anchor off-by-one noise)."""
    def num(x):
        try:
            return int(str(x).lstrip("p"))
        except ValueError:
            return None
    gnums = [num(g) for g in gold]
    gnums = [g for g in gnums if g is not None]
    if not gnums:
        return 0.0
    hit = set()
    for x in ranked[:k]:
        n = num(x)
        if n is None:
            continue
        for g in gnums:
            if abs(g - n) <= tol:
                hit.add(g)
    return len(hit) / len(gnums)


def metrics_for(task, ranked, gold, extra=None, faithful=None):
    """`faithful=False` (the agent cited evidence it never observed) zeroes the strict variants."""
    m = {}
    if task == "literature_evidence":
        m["top1"] = top1(ranked, gold)
        m["mrr"] = rr(ranked, gold)
        m["recall@5"] = float(bool(set(ranked[:5]) & set(gold)))
    else:
        for k in (5, 10, 20):
            m[f"ndcg@{k}"] = ndcg_at_k(ranked, gold, k)
        m["hit@5"] = hit_at_k(ranked, gold, 5)
        for k in (5, 10):
            m[f"recall@{k}"] = recall_at_k(ranked, gold, k)
            m[f"tolerant_recall@{k}"] = tolerant_recall_at_k(ranked, gold, k)
            m[f"oracle_norm_recall@{k}"] = oracle_normalized_recall(ranked, gold, k)
    primary = PRIMARY_METRIC[task]
    m[f"{primary}_strict"] = 0.0 if faithful is False else m.get(primary, 0.0)
    m["evidence_faithful"] = 1.0 if faithful is not False else 0.0
    m.update(extra or {})
    return m


# ---------------- paired statistics ----------------

def paired_bootstrap_ci(a, b, n_boot=2000, seed=0, alpha=0.05):
    """CI of mean(a) - mean(b) over paired observations."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return dict(diff=None, lo=None, hi=None, n=0)
    diffs = [x - y for x, y in zip(a, b)]
    obs = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        boots.append(s / n)
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return dict(diff=obs, lo=lo, hi=hi, n=n)


def paired_permutation_p(a, b, n_perm=2000, seed=0):
    """Two-sided paired randomization test on the mean difference (sign flips)."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return None
    diffs = [x - y for x, y in zip(a, b)]
    obs = abs(sum(diffs) / n)
    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s / n) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def holm(pvals: dict):
    """Holm-Bonferroni correction. Returns {name: (p_raw, p_adj, reject@0.05)}."""
    items = sorted([(k, v) for k, v in pvals.items() if v is not None], key=lambda x: x[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = max(prev, min(1.0, (m - i) * p))
        prev = adj
        out[k] = dict(p_raw=p, p_adj=adj, reject_0_05=adj < 0.05)
    for k, v in pvals.items():
        if v is None:
            out[k] = dict(p_raw=None, p_adj=None, reject_0_05=False)
    return out


# ---------------- claim decision (fixed rule, spec 8) ----------------

PRIMARY_METRIC = {"literature_evidence": "top1", "rtloc": "ndcg@10"}
STRUCTURAL_COMPARISONS = [("typed_graph_agent", "flat_agent"),
                          ("typed_graph_agent", "rewired_graph_agent"),
                          ("typed_graph_agent", "type_shuffled_agent")]


def claim_decision(per_instance, task, seed=0, metric=None):
    """per_instance: {condition: {instance_id: metrics}}. Applies the preregistered rule.

    Hypotheses are DIRECTIONAL ("typed > flat"). The permutation test is two-sided, so a
    significant result in the WRONG direction must never count as support: `direction_supported`
    is reported separately and a hypothesis passes only when the observed difference is positive,
    its CI excludes zero, and the corrected two-sided p is significant.
    """
    metric = metric or PRIMARY_METRIC[task]
    tests, cis, direction = {}, {}, {}
    for hi, lo in STRUCTURAL_COMPARISONS:
        if hi not in per_instance or lo not in per_instance:
            tests[f"{hi}>{lo}"] = None
            continue
        ids = sorted(set(per_instance[hi]) & set(per_instance[lo]))
        a = [per_instance[hi][i][metric] for i in ids]
        b = [per_instance[lo][i][metric] for i in ids]
        ci = paired_bootstrap_ci(a, b, seed=seed)
        cis[f"{hi}>{lo}"] = ci
        tests[f"{hi}>{lo}"] = paired_permutation_p(a, b, seed=seed)
        direction[f"{hi}>{lo}"] = dict(
            observed_direction=("positive" if ci["diff"] > 0 else
                                "negative" if ci["diff"] < 0 else "zero"),
            direction_supported=bool(ci["diff"] > 0))
    corrected = holm(tests)
    for k, v in corrected.items():                      # rename for clarity: the test is two-sided
        v["p_two_sided"] = v.pop("p_raw")
        v["p_two_sided_holm"] = v.pop("p_adj")
        v["significant_two_sided"] = v.pop("reject_0_05")
        v.update(direction.get(k, {}))
        v["supports_hypothesis"] = bool(v.get("significant_two_sided") and v.get("direction_supported"))
    passed = [k for k in cis if corrected.get(k, {}).get("supports_hypothesis") and cis[k]["lo"] > 0]
    if len(passed) == len(STRUCTURAL_COMPARISONS) and len(cis) == len(STRUCTURAL_COMPARISONS):
        decision = "Supported"
    elif passed:
        decision = "Mixed"
    else:
        decision = "Not supported"
    return dict(metric=metric, decision=decision, passed=passed, confidence_intervals=cis,
                tests=corrected,
                note="hypotheses are directional; a significant difference in the wrong direction "
                     "is reported as significant_two_sided=True but supports_hypothesis=False")


def aggregate(task, results_dir=None):
    """Read per-instance prediction files and produce aggregate + paired statistics."""
    d = (results_dir or RESULTS / task)
    per_instance = defaultdict(dict)
    for f in sorted(d.glob("predictions_*.jsonl")):
        cond = f.stem.replace("predictions_", "")
        for r in read_jsonl(f):
            per_instance[cond][r["instance_id"]] = r["metrics"]
    if not per_instance:
        return None
    conds = sorted(per_instance)
    common = set.intersection(*[set(per_instance[c]) for c in conds]) if conds else set()
    agg = {}
    for c in conds:
        ms = [per_instance[c][i] for i in sorted(common)]
        keys = sorted({k for m in ms for k in m})
        agg[c] = {k: (sum(m.get(k, 0) for m in ms) / len(ms) if ms else None) for k in keys}
        agg[c]["n"] = len(ms)
    out = dict(task=task, n_paired_instances=len(common), aggregate=agg,
               claim=claim_decision({c: {i: per_instance[c][i] for i in common} for c in conds}, task))
    write_json(d / "aggregate.json", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--task", default=None)
    a = ap.parse_args()
    tasks = ["literature_evidence", "rtloc"] if a.all else [a.task]
    for t in tasks:
        r = aggregate(t)
        print(t, json.dumps(r["claim"], indent=1) if r else "no predictions yet")


if __name__ == "__main__":
    main()
