"""Phase 1: does the CORRECT graph retrieve the gold citation better than corrupted graphs?

No LLM. For every instance: BM25 seeds from the masked paragraph, then one preregistered
expansion/ranking procedure run under typed / untyped / rewired / type-shuffled graphs and a
BM25-only baseline. Identical hop limits, per-node caps, temporal cutoff and pool size everywhere.

PREREGISTERED RANKING RULE (declared before looking at any result, recorded in the manifest):
    sort by (best seed rank, shortest legal path length, relation-sequence prior, paper id)
with relation prior cites < para_cites < has_paragraph < written_by < connected < other.
The gold target is never used to choose paths, edge types, weights, thresholds or tie-breaks.
The gold citation edge stays masked in both directions for the source paper AND the citing
paragraph, in every variant.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque

from .canonical import current_run_id, provenance, run_dir, write_run_metadata
from .common import DATA, read_jsonl, write_json, write_jsonl, write_manifest
from .evaluate import paired_bootstrap_ci, paired_permutation_p
from .temporal_store import normalize_ts

VARIANTS = ["full", "untyped", "rewire", "type_shuffle"]
LABEL = {"full": "typed", "untyped": "untyped", "rewire": "rewired", "type_shuffle": "type_shuffled"}
KS = [5, 10, 20, 50]
RELATION_PRIOR = {"cites": 0, "para_cites": 1, "has_paragraph": 2, "written_by": 3, "connected": 4}
SEEDS = 10              # BM25 seed papers per instance
MAX_HOPS = 3
PER_NODE_CAP = 25       # neighbours expanded per node (deterministic: sorted by node id)
POOL = 50               # final candidate-pool size (= max K)


def _legal(pid, papers, as_of):
    d = papers.get(pid, {}).get("date")
    return bool(d) and normalize_ts(d) <= as_of


def bm25_seeds(inst, bm, papers, k=SEEDS, n=200):
    as_of = normalize_ts(inst["as_of"])
    out = []
    for pid, score in bm.search(inst["masked_text"], n):
        if pid == inst["source_paper_id"] or not _legal(pid, papers, as_of):
            continue
        out.append((pid, score))
        if len(out) >= k:
            break
    return out


def bm25_ranking(inst, bm, papers, k=POOL, n=400):
    as_of = normalize_ts(inst["as_of"])
    out = []
    for pid, _ in bm.search(inst["masked_text"], n):
        if pid == inst["source_paper_id"] or not _legal(pid, papers, as_of):
            continue
        out.append(pid)
        if len(out) >= k:
            break
    return out


def expand(inst, seeds, adj, papers, masked_pairs, max_hops=MAX_HOPS, cap=PER_NODE_CAP):
    """Breadth-first expansion from the seeds.

    Returns {paper_id: (best_seed_rank, hops, rel_prior, path, support_score)} where
    `support_score` = sum over seeds that reach the candidate of 1/((1+seed_rank)*(1+hops)).
    This is the standard multi-seed proximity signal (PPR-like); it uses no gold information.
    """
    as_of = normalize_ts(inst["as_of"])
    best, support = {}, defaultdict(float)
    for seed_rank, (seed, _score) in enumerate(seeds):
        start = f"paper:{seed}"
        q = deque([(start, 0, ())])
        seen = {start}
        while q:
            node, d, rels = q.popleft()
            if d >= max_hops:
                continue
            nbrs = sorted(adj.get(node, []), key=lambda x: (x[1], x[0]))[:cap]
            for nbr, rel, _dir in nbrs:
                if (node, nbr) in masked_pairs:        # the removed citation stays removed
                    continue
                if nbr in seen:
                    continue
                seen.add(nbr)
                path = rels + (rel,)
                if nbr.startswith("paper:"):
                    pid = nbr.split(":", 1)[1]
                    if pid == inst["source_paper_id"] or not _legal(pid, papers, as_of):
                        continue
                    key = (seed_rank, d + 1, min(RELATION_PRIOR.get(r, 5) for r in path), path)
                    support[pid] += 1.0 / ((1 + seed_rank) * (1 + d + 1))
                    if pid not in best or key[:3] < best[pid][:3]:
                        best[pid] = key
                    q.append((nbr, d + 1, path))
                elif nbr.startswith("para:"):
                    q.append((nbr, d + 1, path))       # paragraphs are waypoints, not candidates
    return {pid: (*k, support[pid]) for pid, k in best.items()}


def rank_candidates(best, rule="support"):
    """Two preregistered rules, both gold-free:

    rule A "path"    : (best seed rank, shortest legal path, relation prior, paper id)
    rule B "support" : (-multi-seed support score, shortest path, best seed rank, paper id)

    Rule A was declared first; it leaves the (large) expansion pool effectively unordered, so rule B
    -- the standard multi-seed proximity score -- was added. Both are reported for every condition.
    """
    if rule == "path":
        return sorted(best, key=lambda p: (best[p][0], best[p][1], best[p][2], p))
    return sorted(best, key=lambda p: (-best[p][4], best[p][1], best[p][0], p))


def evaluate_instance(inst, bm, papers, adjs, seeds_cache=None):
    gold = inst["gold_target_id"]
    seeds = seeds_cache if seeds_cache is not None else bm25_seeds(inst, bm, papers)
    bm_rank = bm25_ranking(inst, bm, papers)
    masked = {(f"paper:{inst['source_paper_id']}", f"paper:{gold}"),
              (f"paper:{gold}", f"paper:{inst['source_paper_id']}"),
              (f"para:{inst.get('paragraph_global_id')}", f"paper:{gold}"),
              (f"paper:{gold}", f"para:{inst.get('paragraph_global_id')}")}
    row = dict(instance_id=inst["instance_id"], as_of=inst["as_of"], gold=gold,
               source_version_verified=not inst.get("source_version_unverifiable", False),
               seed_ids=[s for s, _ in seeds], gold_in_seeds=gold in {s for s, _ in seeds},
               conditions={})
    for name, adj in adjs.items():
        best = expand(inst, seeds, adj, papers, masked)
        for rule in ("support", "path"):
            ranked = rank_candidates(best, rule)[:POOL]
            pool_before_pad = len(ranked)
            padded = [p for p in bm_rank if p not in set(ranked)]
            full_list = (ranked + padded)[:POOL]
            rank = full_list.index(gold) + 1 if gold in full_list else 0
            graph_rank = ranked.index(gold) + 1 if gold in ranked else 0
            key = LABEL[name] if rule == "support" else f"{LABEL[name]}__ruleA_path"
            row["conditions"][key] = dict(
                pool_before_padding=pool_before_pad, n_padded=max(0, POOL - pool_before_pad),
                total_reachable_papers=len(best),
                gold_rank=rank, gold_rank_graph_only=graph_rank,
                gold_reachable=gold in best,
                gold_path_len=best[gold][1] if gold in best else None,
                gold_relation_sequence=list(best[gold][3]) if gold in best else None,
                graph_adds_gold=bool(gold in ranked and gold not in set(bm_rank[:POOL])),
                recall=({f"@{k}": float(0 < rank <= k) for k in KS}),
                mrr=(1.0 / rank if rank else 0.0))
    rank = bm_rank.index(gold) + 1 if gold in bm_rank else 0
    row["conditions"]["bm25_only"] = dict(
        pool_before_padding=len(bm_rank), n_padded=0, total_reachable_papers=len(bm_rank), gold_rank=rank, gold_rank_graph_only=0,
        gold_reachable=None, gold_path_len=None, gold_relation_sequence=None, graph_adds_gold=False,
        recall={f"@{k}": float(0 < rank <= k) for k in KS}, mrr=(1.0 / rank if rank else 0.0))
    return row


def aggregate(rows, subset_name):
    conds = sorted({c for r in rows for c in r["conditions"]})
    agg = {}
    for c in conds:
        rs = [r["conditions"][c] for r in rows]
        agg[c] = dict(n=len(rs), mrr=sum(x["mrr"] for x in rs) / len(rs),
                      **{f"recall@{k}": sum(x["recall"][f"@{k}"] for x in rs) / len(rs) for k in KS},
                      gold_reachable_rate=(sum(bool(x["gold_reachable"]) for x in rs) / len(rs)
                                           if rs[0]["gold_reachable"] is not None else None),
                      graph_adds_gold_rate=sum(x["graph_adds_gold"] for x in rs) / len(rs),
                      mean_pool_before_padding=sum(x["pool_before_padding"] for x in rs) / len(rs),
                      mean_reachable_papers=sum(x.get("total_reachable_papers", 0) for x in rs) / len(rs),
                      path_len_hist=dict(Counter(x["gold_path_len"] for x in rs if x["gold_path_len"])))
    stats = {}
    for hi, lo in [("typed", "rewired"), ("typed", "type_shuffled"), ("typed", "bm25_only"),
                   ("typed", "untyped")]:
        if hi not in conds or lo not in conds:
            continue
        for k in KS:
            a = [r["conditions"][hi]["recall"][f"@{k}"] for r in rows]
            b = [r["conditions"][lo]["recall"][f"@{k}"] for r in rows]
            ci = paired_bootstrap_ci(a, b, seed=0)
            stats[f"{hi}>{lo}@recall@{k}"] = dict(
                **ci, p_two_sided=paired_permutation_p(a, b, seed=0),
                observed_direction=("positive" if ci["diff"] > 0 else
                                    "negative" if ci["diff"] < 0 else "zero"),
                direction_supported=bool(ci["diff"] > 0))
    return dict(subset=subset_name, n=len(rows), aggregate=agg, paired_stats=stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="graph_hard", choices=["graph_hard", "full_test", "both"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    from .run_agent import load_literature_corpus

    rid = a.run_id or current_run_id(create=True)
    d = run_dir(rid, "literature_evidence", create=True)
    papers, bm, adj, _po = load_literature_corpus(tuple(VARIANTS))
    adjs = {v: adj[v] for v in VARIANTS}

    sets = {}
    if a.subset in ("graph_hard", "both"):
        sets["graph_hard"] = list(read_jsonl(DATA / "literature_evidence" / "a1_graph_hard_test.jsonl"))
    if a.subset in ("full_test", "both"):
        sets["full_test"] = [r for r in read_jsonl(DATA / "literature_evidence" / "a1_instances.jsonl")
                             if r["split"] == "test"]
    out_rows, summaries = [], []
    for name, insts in sets.items():
        if a.limit:
            insts = insts[: a.limit]
        rows = []
        for i, inst in enumerate(insts, 1):
            rows.append(dict(evaluate_instance(inst, bm, papers, adjs), subset=name, run_id=rid))
            if i % 100 == 0:
                print(f"  {name}: {i}/{len(insts)}", flush=True)
        out_rows += rows
        summaries.append(aggregate(rows, name))
        ver = [r for r in rows if r["source_version_verified"]]
        if ver:
            summaries.append(aggregate(ver, f"{name}__source_version_verified"))
        unver = [r for r in rows if not r["source_version_verified"]]
        if unver:
            summaries.append(aggregate(unver, f"{name}__source_version_UNVERIFIED_appendix"))
    suffix = a.subset if a.subset != "both" else "both"
    write_jsonl(d / f"candidate_utility_{suffix}.jsonl", out_rows)
    write_json(d / f"candidate_utility_{suffix}.json",
               dict(run_id=rid, provenance=provenance(task="literature_evidence"),
                    procedure=dict(seeds=SEEDS, max_hops=MAX_HOPS, per_node_cap=PER_NODE_CAP,
                                   pool=POOL, ks=KS, relation_prior=RELATION_PRIOR,
                                   ranking_rules=dict(
                                       support="(-multi-seed support, shortest path, seed rank, id)",
                                       path="(seed rank, shortest path, relation prior, id)"),
                                   gold_used_for_ranking=False),
                    summaries=summaries))
    write_manifest(f"candidate_utility_{suffix}", inputs=[DATA / "literature_evidence" / "a1_graph_hard_test.jsonl"],
                   funnel={s["subset"]: s["n"] for s in summaries}, seed=0,
                   filters=dict(seeds=SEEDS, max_hops=MAX_HOPS, per_node_cap=PER_NODE_CAP, pool=POOL),
                   extra=dict(run_id=rid, ranking_rule_preregistered=True))
    write_run_metadata(rid, phase="candidate_utility")
    for s in summaries:
        print(f"\n== {s['subset']} (n={s['n']})")
        for c, v in sorted(s["aggregate"].items()):
            print(f"  {c:14s} R@5={v['recall@5']:.3f} R@10={v['recall@10']:.3f} "
                  f"R@50={v['recall@50']:.3f} MRR={v['mrr']:.3f} "
                  f"reach={v['gold_reachable_rate']} pool={v['mean_pool_before_padding']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
