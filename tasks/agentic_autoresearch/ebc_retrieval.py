"""EBC retrieval primitives: anchor-centred graph expansion, exact <=2-hop reachability, arms.

Shared by ebc_build (structural GraphHard labelling) and ebc_screen (comparative screening) so the
two always agree bit-for-bit. Nothing here uses the target labels for ranking.

PREREGISTERED ranking rule (declared before any comparative run):
    score(p) = AA_ONE_HOP_BONUS * 1[p in N(anchor)]
             + sum over legal intermediates m in N(anchor) & N(p) of 1 / log(1 + deg(m))
    ties broken by a fixed, label-free node ordering.
This is Adamic-Adar proximity from the single revealed anchor. It uses no gold information, no
relation types (untyped connectivity), and is applied identically to every graph variant.

Papers are identified by their version-stripped arXiv id: paper:X and paper:Xv2 are one paper with
complementary edges, so scores are accumulated per version-stripped group, never per node.
"""
from __future__ import annotations

import numpy as np

from .ebc_common import AA_ONE_HOP_BONUS, BUDGET_POOL, K_FINAL, KS, RRF_K, base_id, rrf

EXPAND_DEPTH = max(K_FINAL, BUDGET_POOL)


def _union_nbrs(g, nodes):
    if len(nodes) == 0:
        return np.empty(0, dtype=np.int32)
    if len(nodes) == 1:
        return g.nbrs(int(nodes[0]))
    return np.unique(np.concatenate([g.nbrs(int(v)) for v in nodes]))


def _gather(g, nodes):
    """Concatenated neighbour lists of `nodes` plus the matching per-node repeat counts."""
    counts = (g.indptr[nodes + 1] - g.indptr[nodes]).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int32), counts
    starts = g.indptr[nodes]
    ends = np.cumsum(counts)
    offs = np.repeat(starts - (ends - counts), counts) + np.arange(total, dtype=np.int64)
    return g.indices[offs], counts


def graph_expand(corpus, g, anchor_nodes, legal, k=EXPAND_DEPTH):
    """<=2-hop Adamic-Adar expansion from the anchor. Returns (ranked base ids, group score, hop).

    `legal` is the per-instance boolean node mask from EBCCorpus.legal_nodes: it already removes
    future nodes and every node emitted by the source paper (so no edge created by s is traversed,
    and the evaluation paragraph itself can never create an anchor->target relation).
    """
    n = len(corpus.nodes)
    anchor_nodes = np.asarray(anchor_nodes, dtype=np.int32)
    empty = (np.zeros(corpus.n_groups), np.zeros(corpus.n_groups, dtype=np.int8))
    if anchor_nodes.size == 0:
        return [], *empty
    a_group = int(corpus.group[anchor_nodes[0]])
    na = _union_nbrs(g, anchor_nodes)
    na = na[legal[na]]
    na = na[~np.isin(na, anchor_nodes)]
    score = np.zeros(n, dtype=np.float64)
    hop = np.zeros(n, dtype=np.int8)
    if na.size:
        flat, counts = _gather(g, na)
        if flat.size:
            w = 1.0 / np.log(1.0 + np.maximum(g.degree[na], 1.0))
            score += np.bincount(flat, weights=np.repeat(w, counts), minlength=n)
            hop[flat] = 2
        score[na] += AA_ONE_HOP_BONUS
        hop[na] = 1
    cand = np.flatnonzero((score > 0) & corpus.is_paper & legal)
    if cand.size == 0:
        return [], *empty
    gsc = np.bincount(corpus.group[cand], weights=score[cand], minlength=corpus.n_groups)
    ghop = np.zeros(corpus.n_groups, dtype=np.int8)
    np.maximum.at(ghop, corpus.group[cand], 3 - hop[cand])     # 3-hop so max() == min hop
    gsc[a_group] = 0.0                                          # the anchor is revealed, not a target
    gc = np.flatnonzero(gsc > 0)
    gc = gc[np.lexsort((gc, -gsc[gc]))][:k]
    return [corpus.group_base[i] for i in gc], gsc, (3 - ghop)


def reach_2hop(corpus, g, anchor_nodes, target_nodes, legal):
    """Exact (uncapped) reachability of the target from the anchor within 1-2 legal hops.

    Returns (hops or None, witness_intermediate_node_name or None). A 2-hop witness must itself be
    legal, so no path runs through the source paper, one of its paragraphs, or a future node.
    """
    anchor_nodes = np.asarray(anchor_nodes, dtype=np.int32)
    target_nodes = np.asarray(target_nodes, dtype=np.int32)
    if anchor_nodes.size == 0 or target_nodes.size == 0:
        return None, None
    na = _union_nbrs(g, anchor_nodes)
    if na.size and np.isin(target_nodes, na).any():
        return 1, None
    na = na[legal[na]]
    nt = _union_nbrs(g, target_nodes)
    if na.size == 0 or nt.size == 0:
        return None, None
    common = np.intersect1d(na, nt)
    if common.size == 0:
        return None, None
    common = common[legal[common]]
    common = common[~np.isin(common, anchor_nodes) & ~np.isin(common, target_nodes)]
    if common.size == 0:
        return None, None
    return 2, corpus.nodes[int(common.min())]


# ---------------------------------------------------------------- metrics

def instance_metrics(ranked, targets, bm25_top20=None):
    t = set(targets)
    nt = len(t)
    m = {}
    for k in KS:
        hits = len(set(ranked[:k]) & t)
        m[f"recall@{k}"] = hits / nt
        m[f"set_recall@{k}"] = float(hits == nt)
        m[f"hit@{k}"] = float(hits > 0)
    h10 = len(set(ranked[:10]) & t)
    p10, r10 = h10 / 10.0, h10 / nt
    m["set_f1@10"] = (2 * p10 * r10 / (p10 + r10)) if (p10 + r10) else 0.0
    h20 = len(set(ranked[:20]) & t)
    p20, r20 = h20 / 20.0, h20 / nt
    m["set_f1@20"] = (2 * p20 * r20 / (p20 + r20)) if (p20 + r20) else 0.0
    m["r_precision"] = len(set(ranked[:nt]) & t) / nt
    m["mrr"] = 0.0
    for i, p in enumerate(ranked, 1):
        if p in t:
            m["mrr"] = 1.0 / i
            break
    m["graph_only_discoveries"] = (len((set(ranked[:20]) & t) - set(bm25_top20))
                                   if bm25_top20 is not None else 0)
    m["n_targets"] = nt
    return m


# ---------------------------------------------------------------- arms

def run_arms(corpus, inst, rng_pick):
    """Run every screening arm for one instance. Returns (arm -> ranked list, diagnostics)."""
    s = inst["source_paper_id"]
    as_of_i = inst["as_of_int"]
    legal = corpus.legal_nodes(as_of_i, s)
    anchor = inst["anchor_paper_id"]
    a_nodes = corpus.nodes_for_paper(anchor)
    targets = inst["target_paper_ids"]
    drop = {s, base_id(s), anchor, base_id(anchor)}

    def clean(lst):
        return [p for p in lst if p not in drop][:K_FINAL]

    bm = clean(corpus.bm25_rank(inst["masked_text"], as_of_i, s, exclude=[anchor], k=K_FINAL + 5))
    anchor_text = f"{corpus.title.get(anchor) or ''} {corpus.abstract.get(anchor) or ''}"
    at = clean(corpus.bm25_rank(anchor_text, as_of_i, s, exclude=[anchor], k=K_FINAL + 5))

    out = {"bm25": bm, "anchor_text_only": at}
    diag = dict(anchor_in_graph=bool(a_nodes.size), bm25_top20=bm[:20])

    correct_full = []
    for label, variant in (("correct", "full"), ("rewired", "rewire")):
        g = corpus.graphs[variant]
        ranked = [] if not a_nodes.size else graph_expand(corpus, g, a_nodes, legal)[0]
        ranked = [p for p in ranked if p not in drop]
        if label == "correct":
            correct_full = ranked
        out[f"graph_{label}"] = ranked[:K_FINAL]
        out[f"hybrid_{label}"] = clean(rrf([bm, ranked[:K_FINAL]], k=RRF_K, budget=K_FINAL + 5))

    # ---- random-anchor control: a uniformly drawn legal in-graph paper replaces the anchor ----
    g = corpus.graphs["full"]
    banned = drop | set(targets)
    pool = inst["_random_pool"]
    rnd_node = None
    for _ in range(64):
        if not len(pool):
            break
        cand = int(pool[rng_pick.randrange(len(pool))])
        if legal[cand] and corpus.group_base[corpus.group[cand]] not in banned:
            rnd_node = cand
            break
    r_ranked = ([] if rnd_node is None else
                [p for p in graph_expand(corpus, g, corpus.nodes_for_paper(
                    corpus.group_base[corpus.group[rnd_node]]), legal)[0] if p not in drop])
    out["graph_random_anchor"] = r_ranked[:K_FINAL]
    out["hybrid_random_anchor"] = clean(rrf([bm, r_ranked[:K_FINAL]], k=RRF_K, budget=K_FINAL + 5))
    diag["random_anchor_id"] = None if rnd_node is None else corpus.group_base[corpus.group[rnd_node]]

    # ---- degree-matched random anchor: isolates anchor identity from anchor degree ----
    dm_node = None
    if a_nodes.size and len(pool):
        adeg = float(g.degree[a_nodes].sum())
        gap = np.abs(g.degree[pool] - adeg)
        mm = min(256, gap.size)
        near = np.argpartition(gap, mm - 1)[:mm]
        for j in near[np.lexsort((pool[near], gap[near]))]:
            cand = int(pool[j])
            if legal[cand] and corpus.group_base[corpus.group[cand]] not in banned:
                dm_node = cand
                break
    dm_ranked = ([] if dm_node is None else
                 [p for p in graph_expand(corpus, g, corpus.nodes_for_paper(
                     corpus.group_base[corpus.group[dm_node]]), legal)[0] if p not in drop])
    out["graph_degree_matched_anchor"] = dm_ranked[:K_FINAL]
    diag["degree_matched_anchor_id"] = (None if dm_node is None
                                        else corpus.group_base[corpus.group[dm_node]])
    diag["anchor_degree"] = float(g.degree[a_nodes].sum()) if a_nodes.size else None

    # ---- reachability / budget diagnostics ----
    budget = set(correct_full[:BUDGET_POOL])
    diag["budget_pool"] = correct_full[:BUDGET_POOL]
    diag["reach"] = {}
    for t in targets:
        t_nodes = corpus.nodes_for_paper(t)
        rec = {}
        for label, variant in (("correct", "full"), ("rewired", "rewire")):
            h, w = reach_2hop(corpus, corpus.graphs[variant], a_nodes, t_nodes, legal)
            rec[label] = dict(hops=h, witness=w)
        rec["in_budget_pool"] = t in budget
        diag["reach"][t] = rec
    diag["legal_mask"] = legal
    return out, diag
