"""Graph variants for both tasks: full / flat / untyped / rewire / type_shuffle.

Invariants enforced for every variant (spec section 6):
  * identical node IDs and node text;
  * only relations differ;
  * rewiring is independent per relation type and preserves per-type in/out degree sequences;
  * type shuffle preserves endpoints and the global count of each relation type;
  * no new self-loops or duplicate edges unless present in the original;
  * seeds recorded in the manifest, sanity report written to perturbation_sanity.json.

RT-Loc variants are derived from the existing per-instance graphs in data/provenance/graphs
(rt_loc_full.jsonl) so node text is byte-identical to the provenance pipeline.
Literature-evidence variants are derived from the densified citation graph.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from .common import DATA, PROV, read_jsonl, write_json, write_jsonl, write_manifest, DEFAULT_SEED

VARIANTS = ("full", "flat", "untyped", "rewire", "type_shuffle")


# ---------------- perturbation primitives (edge list = list of [src, dst, rel]) ----------------

def flat(edges, rng):
    return []


def untyped(edges, rng):
    return [[a, b, "connected"] for a, b, _ in edges]


def rewire(edges, rng, swaps_per_edge=10):
    """Degree-preserving double-edge swaps within each relation type.

    A swap (a->b, c->d) => (a->d, c->b) keeps every out-degree at the source and in-degree at the
    target, and is rejected if it would create a self-loop or a duplicate edge.
    """
    by_rel = defaultdict(list)
    for a, b, r in edges:
        by_rel[r].append([a, b])
    out = []
    for r, es in by_rel.items():
        seen = {(a, b) for a, b in es}
        had_self_loop = any(a == b for a, b in es)
        n = len(es)
        for _ in range(swaps_per_edge * n):
            if n < 2:
                break
            i, j = rng.randrange(n), rng.randrange(n)
            if i == j:
                continue
            (a, b), (c, d) = es[i], es[j]
            if (not had_self_loop and (a == d or c == b)) or (a, d) in seen or (c, b) in seen:
                continue
            seen.discard((a, b))
            seen.discard((c, d))
            seen.add((a, d))
            seen.add((c, b))
            es[i], es[j] = [a, d], [c, b]
        out += [[a, b, r] for a, b in es]
    return out


def type_shuffle(edges, rng):
    """Same endpoints; relation labels permuted (global per-type counts preserved)."""
    rels = [r for _, _, r in edges]
    rng.shuffle(rels)
    return [[a, b, r] for (a, b, _), r in zip(edges, rels)]


PERTURB = {"full": lambda e, rng: [list(x) for x in e], "flat": flat, "untyped": untyped,
           "rewire": rewire, "type_shuffle": type_shuffle}


# ---------------- sanity checks ----------------

def degree_profile(edges):
    out_d, in_d = defaultdict(Counter), defaultdict(Counter)
    for a, b, r in edges:
        out_d[r][a] += 1
        in_d[r][b] += 1
    return ({r: sorted(c.values()) for r, c in out_d.items()},
            {r: sorted(c.values()) for r, c in in_d.items()})


def graph_stats(edges):
    return dict(n_edges=len(edges), relation_histogram=dict(Counter(r for _, _, r in edges)),
                self_loops=sum(1 for a, b, _ in edges if a == b),
                duplicates=len(edges) - len({(a, b, r) for a, b, r in edges}))


def sanity_report(base_edges, variants, base_nodes, variant_nodes, seed):
    b_out, b_in = degree_profile(base_edges)
    rep = dict(seed=seed, base=graph_stats(base_edges), variants={})
    for name, edges in variants.items():
        v_out, v_in = degree_profile(edges)
        r = graph_stats(edges)
        r["identical_node_ids"] = variant_nodes[name] == base_nodes
        if name == "rewire":
            r["per_type_out_degree_preserved"] = v_out == b_out
            r["per_type_in_degree_preserved"] = v_in == b_in
        if name == "type_shuffle":
            r["relation_histogram_preserved"] = r["relation_histogram"] == rep["base"]["relation_histogram"]
            r["endpoints_preserved"] = sorted((a, b) for a, b, _ in edges) == sorted(
                (a, b) for a, b, _ in base_edges)
        if name == "untyped":
            r["endpoints_preserved"] = sorted((a, b) for a, b, _ in edges) == sorted(
                (a, b) for a, b, _ in base_edges)
        rep["variants"][name] = r
    return rep


# ---------------- RT-Loc: per-instance graphs ----------------

def build_rtloc_variants(seed=DEFAULT_SEED, variants=VARIANTS):
    src = PROV / "graphs" / "rt_loc_full.jsonl"
    base = list(read_jsonl(src))
    out_dir = DATA / "rtloc" / "graphs"
    all_base_edges, agg = [], {v: [] for v in variants}
    node_sets, base_nodes = {}, []
    for v in variants:
        rows = []
        for k, g in enumerate(base):
            rng = random.Random(f"{seed}/{v}/{g['inst_id']}".__hash__() & 0xFFFFFFFF)
            edges = [[a, b, r] for a, b, r in g["edges"]]
            rows.append(dict(inst_id=g["inst_id"], paper=g["paper"], nodes=g["nodes"],
                             edges=PERTURB[v](edges, rng), gold=g["gold"], variant=v))
        write_jsonl(out_dir / f"rt_loc_{v}.jsonl", rows)
        # namespace endpoints by instance: node ids like "p1" recur in every instance graph, so a
        # naive union would report cross-instance repeats as duplicate edges / merged degrees
        agg[v] = [[f"{r['inst_id']}|{a}", f"{r['inst_id']}|{b}", rel]
                  for r in rows for a, b, rel in r["edges"]]
        node_sets[v] = sorted({f"{r['inst_id']}|{n}" for r in rows for n in r["nodes"]})
    all_base_edges = [[f"{g['inst_id']}|{a}", f"{g['inst_id']}|{b}", rel]
                      for g in base for a, b, rel in g["edges"]]
    base_nodes = sorted({f"{g['inst_id']}|{n}" for g in base for n in g["nodes"]})
    # node text must be identical across variants (checked against the source file)
    text_ok = True
    for v in variants:
        rows = list(read_jsonl(out_dir / f"rt_loc_{v}.jsonl"))
        for g0, g1 in zip(base, rows):
            if {k: n["text"] for k, n in g0["nodes"].items()} != {k: n["text"] for k, n in g1["nodes"].items()} \
                    or g0["gold"] != g1["gold"]:
                text_ok = False
                break
    rep = sanity_report(all_base_edges, agg, base_nodes, node_sets, seed)
    rep["node_text_and_gold_identical_across_variants"] = text_ok
    rep["n_instances"] = len(base)
    write_json(out_dir / "perturbation_sanity.json", rep)
    write_manifest("rtloc_graph_variants", inputs=[src], funnel=dict(instances=len(base)),
                   filters=dict(source="data/provenance/graphs/rt_loc_full.jsonl"), seed=seed,
                   graph_variant=list(variants), extra=dict(sanity=rep))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["rtloc"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args()
    if "rtloc" in a.tasks:
        rep = build_rtloc_variants(a.seed, tuple(a.variants))
        print("rtloc variants:", {k: v["n_edges"] for k, v in rep["variants"].items()})
        print("node text/gold identical:", rep["node_text_and_gold_identical_across_variants"])
    if "literature_evidence" in a.tasks:
        from .build_literature_evidence import build_citation_graph_variants
        rep = build_citation_graph_variants(a.seed, tuple(a.variants))
        print("literature_evidence variants:", {k: v["n_edges"] for k, v in rep["variants"].items()})


if __name__ == "__main__":
    main()
