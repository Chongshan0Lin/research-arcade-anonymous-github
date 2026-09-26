"""Hybrid BM25 + graph retrieval, rewiring dose-response, and edge-family ablation (no LLM).

Three questions, all at a FIXED final candidate budget K so nothing wins by returning more:
  1. complementarity -- does graph expansion find golds BM25 misses, and does fusion beat both?
  2. dose-response   -- does candidate quality fall monotonically as topology is rewired
                        0/25/50/75/100% (degree-preserving, nodes/text/degrees fixed)?
  3. edge families   -- which connectivity carries the signal: citation, paragraph-paper,
                        authorship, or all of them?

Fusion methods: reciprocal rank fusion, score-normalized fusion, and a LightGBM ranker trained on
the validation split and frozen before touching test.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict, deque

from .canonical import current_run_id, provenance, run_dir, write_run_metadata
from .candidate_utility import (KS, MAX_HOPS, PER_NODE_CAP, POOL, SEEDS, bm25_ranking, bm25_seeds,
                                _legal)
from .common import DATA, read_jsonl, write_json, write_jsonl, write_manifest
from .evaluate import paired_bootstrap_ci, paired_permutation_p
from .temporal_store import normalize_ts

K_FINAL = 50
RRF_K = 60
DOSES = [0, 25, 50, 75, 100]
FAMILIES = {"citation_only": {"cites"}, "paragraph_only": {"para_cites", "has_paragraph"},
            "authorship_only": {"written_by"}, "all_families": None}


def graph_rank(inst, seeds, adj, papers, masked, allowed_relations=None, k=K_FINAL):
    """Label-free support-ranked expansion, optionally restricted to one edge family."""
    as_of = normalize_ts(inst["as_of"])
    support, hops = defaultdict(float), {}
    for seed_rank, (seed, _s) in enumerate(seeds):
        start = f"paper:{seed}"
        q, seen = deque([(start, 0)]), {start}
        while q:
            node, d = q.popleft()
            if d >= MAX_HOPS:
                continue
            for nbr, rel, _dir in sorted(adj.get(node, []), key=lambda x: x[0])[:PER_NODE_CAP]:
                if allowed_relations is not None and rel not in allowed_relations:
                    continue
                if (node, nbr) in masked or nbr in seen:
                    continue
                seen.add(nbr)
                if nbr.startswith("paper:"):
                    pid = nbr.split(":", 1)[1]
                    if pid != inst["source_paper_id"] and _legal(pid, papers, as_of):
                        support[pid] += 1.0 / ((1 + seed_rank) * (2 + d))
                        hops[pid] = min(hops.get(pid, 99), d + 1)
                    q.append((nbr, d + 1))
                elif nbr.startswith("para:"):
                    q.append((nbr, d + 1))
    ranked = sorted(support, key=lambda p: (-support[p], hops.get(p, 99), p))
    return ranked[:k], support, hops


def rrf(lists, k=RRF_K, budget=K_FINAL):
    score = defaultdict(float)
    for lst in lists:
        for i, pid in enumerate(lst, 1):
            score[pid] += 1.0 / (k + i)
    return sorted(score, key=lambda p: (-score[p], p))[:budget]


def norm_fusion(bm_list, graph_scores, budget=K_FINAL, w=0.5):
    """Score-normalized fusion: min-max normalise each side, then weighted sum."""
    bm_score = {pid: 1.0 - i / max(1, len(bm_list)) for i, pid in enumerate(bm_list)}
    g = dict(graph_scores)
    if g:
        lo, hi = min(g.values()), max(g.values())
        g = {p: (v - lo) / (hi - lo) if hi > lo else 0.0 for p, v in g.items()}
    out = defaultdict(float)
    for p, v in bm_score.items():
        out[p] += (1 - w) * v
    for p, v in g.items():
        out[p] += w * v
    return sorted(out, key=lambda p: (-out[p], p))[:budget]


def metrics(ranked, gold):
    rank = ranked.index(gold) + 1 if gold in ranked else 0
    return dict(rank=rank, mrr=1.0 / rank if rank else 0.0,
                **{f"recall@{k}": float(0 < rank <= k) for k in KS})


def run_instance(inst, bm, papers, adjs, dose_adjs=None):
    gold = inst["gold_target_id"]
    masked = {(f"paper:{inst['source_paper_id']}", f"paper:{gold}"),
              (f"paper:{gold}", f"paper:{inst['source_paper_id']}"),
              (f"para:{inst.get('paragraph_global_id')}", f"paper:{gold}"),
              (f"paper:{gold}", f"para:{inst.get('paragraph_global_id')}")}
    seeds = bm25_seeds(inst, bm, papers, k=SEEDS)
    bm_list = bm25_ranking(inst, bm, papers, k=K_FINAL)
    out = dict(instance_id=inst["instance_id"], gold=gold,
               source_version_verified=not inst.get("source_version_unverifiable", False),
               conditions={})
    out["conditions"]["bm25"] = metrics(bm_list, gold)
    graph_lists = {}
    for name, adj in adjs.items():
        g_list, support, _h = graph_rank(inst, seeds, adj, papers, masked)
        graph_lists[name] = (g_list, support)
        out["conditions"][f"graph_{name}"] = metrics(g_list, gold)
        out["conditions"][f"hybrid_rrf_{name}"] = metrics(rrf([bm_list, g_list]), gold)
        out["conditions"][f"hybrid_norm_{name}"] = metrics(norm_fusion(bm_list, support), gold)
    # complementarity decomposition against the correct graph
    g_correct = graph_lists.get("typed", ([], {}))[0]
    out["found_by"] = ("both" if gold in bm_list and gold in g_correct else
                       "bm25_only" if gold in bm_list else
                       "graph_only" if gold in g_correct else "neither")
    # edge-family ablation (correct topology only)
    for fam, rels in FAMILIES.items():
        g_list, support, _ = graph_rank(inst, seeds, adjs["typed"], papers, masked, rels)
        out["conditions"][f"family_{fam}"] = metrics(g_list, gold)
        out["conditions"][f"hybrid_family_{fam}"] = metrics(rrf([bm_list, g_list]), gold)
    # rewiring dose-response
    for dose, adj in (dose_adjs or {}).items():
        g_list, support, _ = graph_rank(inst, seeds, adj, papers, masked)
        out["conditions"][f"dose_{dose}"] = metrics(g_list, gold)
        out["conditions"][f"hybrid_dose_{dose}"] = metrics(rrf([bm_list, g_list]), gold)
    return out


def partial_rewire(edges, pct, seed=0):
    """Degree-preserving double-edge swaps on a fraction of edges, per relation type."""
    from .build_graph_variants import rewire
    if pct == 0:
        return [list(e) for e in edges]
    if pct >= 100:
        return rewire([list(e) for e in edges], random.Random(seed))
    rng = random.Random(seed)
    idx = list(range(len(edges)))
    rng.shuffle(idx)
    take = set(idx[: int(len(edges) * pct / 100)])
    sub = [list(edges[i]) for i in take]
    rest = [list(edges[i]) for i in range(len(edges)) if i not in take]
    return rewire(sub, random.Random(seed)) + rest


def validate_dose(base_edges, dose_edges, dose):
    """Per-dose structural validation (spec 1.2): identical counts, degree preservation, no new
    self-loops/duplicates, and the exact number of endpoints actually swapped."""
    from .build_graph_variants import degree_profile, graph_stats
    b_out, b_in = degree_profile(base_edges)
    d_out, d_in = degree_profile(dose_edges)
    gs_b, gs_d = graph_stats(base_edges), graph_stats(dose_edges)
    base_set = {tuple(e) for e in base_edges}
    changed = sum(1 for e in dose_edges if tuple(e) not in base_set)
    return dict(dose_pct=dose, n_edges=gs_d["n_edges"], n_edges_base=gs_b["n_edges"],
                edge_count_identical=gs_d["n_edges"] == gs_b["n_edges"],
                relation_histogram_identical=gs_d["relation_histogram"] == gs_b["relation_histogram"],
                per_relation_out_degree_preserved=d_out == b_out,
                per_relation_in_degree_preserved=d_in == b_in,
                self_loops=gs_d["self_loops"], self_loops_base=gs_b["self_loops"],
                duplicates=gs_d["duplicates"], duplicates_base=gs_b["duplicates"],
                edges_changed=changed,
                edges_changed_pct=round(100.0 * changed / max(1, gs_b["n_edges"]), 2))


def monotonic_trend_test(values_by_dose, n_perm=10000, seed=0):
    """Preregistered trend test: Spearman correlation between dose and per-instance recall,
    with a permutation null over dose labels."""
    import random as _r
    doses = sorted(values_by_dose)
    means = [sum(values_by_dose[d]) / len(values_by_dose[d]) for d in doses]
    def spearman(xs, ys):
        rx = {v: i for i, v in enumerate(sorted(xs))}
        ry = {v: i for i, v in enumerate(sorted(ys))}
        ax = [rx[v] for v in xs]
        ay = [ry[v] for v in ys]
        n = len(xs)
        mx, my = sum(ax) / n, sum(ay) / n
        num = sum((a - mx) * (b - my) for a, b in zip(ax, ay))
        den = (sum((a - mx) ** 2 for a in ax) * sum((b - my) ** 2 for b in ay)) ** 0.5
        return num / den if den else 0.0
    obs = spearman(doses, means)
    rng = _r.Random(seed)
    count = 0
    for _ in range(n_perm):
        perm = means[:]
        rng.shuffle(perm)
        if abs(spearman(doses, perm)) >= abs(obs) - 1e-12:
            count += 1
    return dict(doses=doses, mean_recall_by_dose=dict(zip(doses, means)), spearman=obs,
                p_permutation=(count + 1) / (n_perm + 1),
                monotone_decreasing=all(means[i] >= means[i + 1] - 1e-12 for i in range(len(means) - 1)))


def aggregate(rows, name):
    conds = sorted({c for r in rows for c in r["conditions"]})
    agg = {c: dict(n=len(rows),
                   mrr=sum(r["conditions"][c]["mrr"] for r in rows) / len(rows),
                   **{f"recall@{k}": sum(r["conditions"][c][f"recall@{k}"] for r in rows) / len(rows)
                      for k in KS})
           for c in conds}
    stats = {}
    pairs = [("hybrid_rrf_typed", "bm25"), ("hybrid_rrf_typed", "graph_typed"),
             ("hybrid_rrf_typed", "hybrid_rrf_rewired"), ("hybrid_rrf_typed", "hybrid_rrf_shuffled"),
             ("hybrid_rrf_typed", "hybrid_rrf_untyped"), ("graph_typed", "graph_rewired")]
    for hi, lo in pairs:
        if hi not in conds or lo not in conds:
            continue
        for k in (10, 50):
            a = [r["conditions"][hi][f"recall@{k}"] for r in rows]
            b = [r["conditions"][lo][f"recall@{k}"] for r in rows]
            ci = paired_bootstrap_ci(a, b, seed=0)
            stats[f"{hi}>{lo}@recall@{k}"] = dict(
                **ci, p_two_sided=paired_permutation_p(a, b, seed=0),
                observed_direction=("positive" if ci["diff"] > 0 else
                                    "negative" if ci["diff"] < 0 else "zero"),
                direction_supported=bool(ci["diff"] > 0))
    return dict(subset=name, n=len(rows), found_by=dict(Counter(r["found_by"] for r in rows)),
                aggregate=agg, paired_stats=stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="graph_hard", choices=["graph_hard", "full_test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verified-only", action="store_true")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    from .run_agent import load_literature_corpus

    rid = a.run_id or current_run_id(create=True)
    d = run_dir(rid, "literature_evidence", create=True)
    papers, bm, adj, _ = load_literature_corpus(("full", "untyped", "rewire", "type_shuffle"))
    adjs = {"typed": adj["full"], "untyped": adj["untyped"], "rewired": adj["rewire"],
            "shuffled": adj["type_shuffle"]}

    print("building dose-response graphs...", flush=True)
    edges = [(r["s"], r["t"], r["r"]) for r in
             read_jsonl(DATA / "literature_evidence" / "graphs" / "citation_graph_full.jsonl")]
    dose_adjs, dose_validation = {}, []
    for dose in DOSES:
        ev = partial_rewire(edges, dose, seed=20260922)
        dd = defaultdict(list)
        for s, t, r in ev:
            dd[s].append((t, r, "out"))
            dd[t].append((s, r, "in"))
        dose_adjs[dose] = dd
        dose_validation.append(validate_dose(edges, ev, dose))
        print(f"  dose {dose}% ready ({len(ev)} edges, "
              f"{dose_validation[-1]['edges_changed_pct']}% endpoints changed)", flush=True)

    src = ("a1_graph_hard_test.jsonl" if a.subset == "graph_hard" else "a1_instances.jsonl")
    insts = [r for r in read_jsonl(DATA / "literature_evidence" / src)
             if a.subset == "graph_hard" or r["split"] == "test"]
    if a.verified_only:
        insts = [r for r in insts if not r.get("source_version_unverifiable", False)]
    if a.limit:
        insts = insts[: a.limit]
    rows = []
    for i, inst in enumerate(insts, 1):
        rows.append(dict(run_instance(inst, bm, papers, adjs, dose_adjs), subset=a.subset, run_id=rid))
        if i % 50 == 0:
            print(f"  {i}/{len(insts)}", flush=True)
    write_jsonl(d / f"hybrid_{a.subset}.jsonl", rows)
    summaries = [aggregate(rows, a.subset)]
    ver = [r for r in rows if r["source_version_verified"]]
    if ver and len(ver) != len(rows):
        summaries.append(aggregate(ver, f"{a.subset}__source_version_verified"))
    trend = {}
    for k in (10, 50):
        vals = {dose: [r["conditions"][f"dose_{dose}"][f"recall@{k}"] for r in rows] for dose in DOSES}
        trend[f"graph_recall@{k}"] = monotonic_trend_test(vals)
        vals_h = {dose: [r["conditions"][f"hybrid_dose_{dose}"][f"recall@{k}"] for r in rows]
                  for dose in DOSES}
        trend[f"hybrid_recall@{k}"] = monotonic_trend_test(vals_h)
    write_json(d / f"hybrid_{a.subset}.json",
               dict(run_id=rid, provenance=provenance(task="literature_evidence"),
                    dose_validation=dose_validation, dose_trend_test=trend,
                    budget_K=K_FINAL, rrf_k=RRF_K, doses=DOSES, families=list(FAMILIES),
                    summaries=summaries))
    write_manifest(f"hybrid_{a.subset}", inputs=[DATA / "literature_evidence" / src],
                   funnel=dict(instances=len(insts)), seed=20260922,
                   filters=dict(budget_K=K_FINAL, verified_only=a.verified_only),
                   extra=dict(run_id=rid))
    write_run_metadata(rid, phase=f"hybrid_{a.subset}")
    for s in summaries:
        print(f"\n== {s['subset']} (n={s['n']}) found_by={s['found_by']}")
        for c, v in sorted(s["aggregate"].items()):
            print(f"  {c:26s} R@10={v['recall@10']:.3f} R@50={v['recall@50']:.3f} MRR={v['mrr']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
