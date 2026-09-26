"""EBC construction, leakage audit and frozen slices (runbook sections 3.1-3.3).

CPU only, no LLM, no GPU.

    python -m tasks.agentic_autoresearch.ebc_build --out <master>/evidence_bundle_completion

Stages
  1. build     : real paragraphs with 2-5 resolved citations -> one anchor + 1-4 targets
  2. split     : train/validation/test by SOURCE PAPER (no source paper spans two splits)
  3. label     : structural GraphHard labelling from (a) BM25 top-20 and (b) exact <=2-hop
                 reachability in the CORRECT pre-cutoff graph. No comparative arm is executed
                 here, so freezing happens strictly before any correct-vs-rewired result exists.
  4. freeze    : frozen_ids.jsonl + SHA-256 of every frozen ID list and input artefact
  5. audit     : construction-level temporal audit rows (source / anchor / target)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .common import DATA, PROV, db_conn, file_hash, git_commit, now_iso, read_jsonl, write_json, write_jsonl
from .ebc_common import (EBC_SEED, EBCCorpus, GRAPH_DIR, INF_DATE, MAX_CITATIONS,
                         MAX_SCREENING_INSTANCES, MIN_CITATIONS, MIN_CONTEXT_TOKENS, PARAMS,
                         SPLIT_FRACTIONS, TASK_NAME, BM25_HARD_TOPK, GRAPHHARD_MAX_HOPS, K_FINAL,
                         base_id, context_tokens, date_int, normalize_all_citations)
from .ebc_retrieval import reach_2hop

OUT_DATA = DATA / "evidence_bundle_completion"
INSTANCES = OUT_DATA / "ebc_instances.jsonl"
LABELS = OUT_DATA / "ebc_structural_labels.jsonl"

AUDIT_FIELDS = [
    "row_kind", "instance_id", "source_paper_id", "as_of", "arm", "rank", "object_id",
    "object_kind", "object_date", "temporally_legal", "strictly_before_cutoff",
    "is_source_paper", "is_anchor", "is_target", "owner_paper_id", "edge_created_by_source",
    "hop", "witness_intermediate", "witness_date", "witness_legal", "violation", "violation_reason",
]


def sha256_list(ids):
    h = hashlib.sha256()
    for i in sorted(ids):
        h.update(i.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------- 1. construction

def build_instances(seed=EBC_SEED, materialize=True):
    funnel = Counter()
    pce = pd.read_csv(PROV / "paragraph_citation_edges.csv", dtype=str)
    funnel["00_paragraph_citation_edges"] = len(pce)
    pce = pce[pce.arxiv_target.notna()]
    funnel["01_resolved_arxiv_target"] = len(pce)
    pce["src"] = pce.citing_arxiv_id.str.replace(r"v\d+$", "", regex=True)
    pce = pce[pce.src != pce.arxiv_target]
    funnel["02_target_not_source"] = len(pce)

    conn = db_conn()
    papers = pd.read_sql("select coalesce(base_arxiv_id, arxiv_id) as base, "
                         "min(submit_date::text) as d, max(title) as t, max(version::text) as v "
                         "from papers group by 1", conn)
    date = dict(zip(papers.base, papers.d.astype(str)))
    title = dict(zip(papers.base, papers.t))
    version = dict(zip(papers.base, papers.v))

    pce["as_of"] = pce.src.map(date)
    pce = pce[pce.as_of.notna()]
    funnel["03_source_date_known"] = len(pce)
    pce["tdate"] = pce.arxiv_target.map(date)

    # ---- per-paragraph bundle, with the strict "every target legal" rule ----
    keep, drop_future, drop_unknown = [], 0, 0
    grp = pce.groupby("paragraph_global_id")
    funnel["04_distinct_paragraphs"] = grp.ngroups
    for pgid, g in grp:
        tg = sorted(set(g.arxiv_target))
        as_of = g.as_of.iloc[0]
        if any(pd.isna(date.get(t)) or date.get(t) is None for t in tg):
            drop_unknown += 1
            continue
        if any(str(date[t]) > str(as_of) for t in tg):
            drop_future += 1
            continue
        if not (MIN_CITATIONS <= len(tg) <= MAX_CITATIONS):
            continue
        keep.append(dict(paragraph_global_id=int(pgid), src=base_id(g.src.iloc[0]), as_of=str(as_of),
                         targets=tg, bib_keys=sorted(set(g.bib_key.dropna()))))
    funnel["05_dropped_unknown_target_date"] = drop_unknown
    funnel["06_dropped_future_target"] = drop_future
    funnel["07_bundle_size_2_to_5"] = len(keep)
    if not materialize:
        return keep, funnel

    # ---- paragraph text ----
    ids = [k["paragraph_global_id"] for k in keep]
    text, section, owner = {}, {}, {}
    for i in range(0, len(ids), 20000):
        chunk = ids[i:i + 20000]
        par = pd.read_sql("select id, paper_arxiv_id, paper_section, content from paragraphs "
                          "where id = any(%(i)s)", conn, params={"i": chunk})
        for r in par.itertuples():
            text[int(r.id)] = r.content
            section[int(r.id)] = r.paper_section
            owner[int(r.id)] = r.paper_arxiv_id

    rows = []
    for k in keep:
        pgid = k["paragraph_global_id"]
        raw = text.get(pgid)
        if raw is None:
            funnel["08_paragraph_text_missing"] += 1
            continue
        masked, n_markers = normalize_all_citations(raw)
        if n_markers == 0:
            funnel["09_no_citation_marker_in_text"] += 1
            continue
        if context_tokens(masked) < MIN_CONTEXT_TOKENS:
            funnel["10_context_too_short"] += 1
            continue
        # anchor: one revealed citation, drawn deterministically (seeded per paragraph)
        r = random.Random(f"{seed}|{pgid}")
        tg = [base_id(x) for x in k["targets"]]
        anchor = tg[r.randrange(len(tg))]
        targets = [t for t in tg if t != anchor]
        masked = masked[:8000]
        low = masked.lower()
        rows.append(dict(
            instance_id=f"EBC-{pgid}", task=TASK_NAME, schema_version="1.0",
            paragraph_global_id=pgid, source_paper_id=k["src"], as_of=k["as_of"],
            as_of_int=date_int(k["as_of"]), section=section.get(pgid),
            masked_text=masked, n_citation_markers=n_markers,
            n_citations=len(tg), all_cited_paper_ids=tg,
            anchor_paper_id=anchor, anchor_title=title.get(anchor),
            anchor_date=date.get(anchor),
            target_paper_ids=targets, target_titles=[title.get(t) for t in targets],
            target_dates=[date.get(t) for t in targets],
            bib_keys=k["bib_keys"],
            source_version_unverifiable=bool(version.get(k["src"]) not in (None, "1", 1, "None")),
            target_title_appears_in_text=[bool(title.get(t) and len(str(title.get(t))) > 12
                                               and str(title.get(t)).lower() in low)
                                          for t in targets],
            dup_group=hashlib.sha256(
                re.sub(r"[^a-z0-9]", "", low)[:400].encode()).hexdigest()[:16]))
    funnel["11_instances_built"] = len(rows)

    # ---- split by source paper; near-duplicate paragraphs stay inside one split ----
    src_keys = sorted({r["source_paper_id"] for r in rows})
    rr = random.Random(seed)
    rr.shuffle(src_keys)
    n = len(src_keys)
    n_tr, n_va = int(n * SPLIT_FRACTIONS[0]), int(n * (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]))
    assign = {k: ("train" if i < n_tr else "validation" if i < n_va else "test")
              for i, k in enumerate(src_keys)}
    dup_split = {}
    for r in rows:
        dup_split.setdefault(r["dup_group"], assign[r["source_paper_id"]])
    for r in rows:
        r["split"] = dup_split[r["dup_group"]]
    for sp in ("train", "validation", "test"):
        funnel[f"12_split_{sp}"] = sum(r["split"] == sp for r in rows)
    funnel["13_source_papers"] = n
    write_jsonl(INSTANCES, rows)
    return rows, funnel


# ---------------------------------------------------------------- 3. structural labelling

def structural_labels(corpus, insts):
    """BM25 top-20 + exact <=2-hop correct-graph reachability. No comparative arm is run."""
    out = []
    g = corpus.graphs["full"]
    for i, inst in enumerate(insts, 1):
        s = inst["source_paper_id"]
        anchor = inst["anchor_paper_id"]
        legal = corpus.legal_nodes(inst["as_of_int"], s)
        a_nodes = corpus.nodes_for_paper(anchor)
        drop = {s, base_id(s), anchor, base_id(anchor)}
        bm = [p for p in corpus.bm25_rank(inst["masked_text"], inst["as_of_int"], s,
                                          exclude=[anchor], k=K_FINAL + 5) if p not in drop]
        top20 = set(bm[:BM25_HARD_TOPK])
        per_t = {}
        for t in inst["target_paper_ids"]:
            t_nodes = corpus.nodes_for_paper(t)
            hops, wit = reach_2hop(corpus, g, a_nodes, t_nodes, legal)
            per_t[t] = dict(in_bm25_top20=t in top20, in_graph=bool(t_nodes.size),
                            reach_hops=hops, witness=wit)
        hard = any((not v["in_bm25_top20"]) and v["reach_hops"] is not None
                   and v["reach_hops"] <= GRAPHHARD_MAX_HOPS for v in per_t.values())
        out.append(dict(instance_id=inst["instance_id"], bm25_top20=bm[:BM25_HARD_TOPK],
                        anchor_in_graph=bool(a_nodes.size), targets=per_t, graphhard=bool(hard)))
        if i % 500 == 0:
            print(f"  labelled {i}/{len(insts)}", flush=True)
    return out


# ---------------------------------------------------------------- 5. construction audit

def construction_audit_rows(corpus, insts):
    """One row per source / anchor / target object of every frozen instance."""
    for inst in insts:
        s, as_of, as_i = inst["source_paper_id"], inst["as_of"], inst["as_of_int"]
        s_group = corpus.source_group(s)
        base = dict(instance_id=inst["instance_id"], source_paper_id=s, as_of=as_of, arm="",
                    rank="", hop="", witness_intermediate="", witness_date="", witness_legal="")
        yield dict(base, row_kind="source", object_id=s, object_kind="paper",
                   object_date=corpus.date.get(s, ""),
                   temporally_legal=True, strictly_before_cutoff=False,
                   is_source_paper=True, is_anchor=False, is_target=False, owner_paper_id=s,
                   edge_created_by_source=True, violation=False,
                   violation_reason="excluded_from_all_candidate_pools")
        for role, pid in ([("anchor", inst["anchor_paper_id"])] +
                          [("target", t) for t in inst["target_paper_ids"]]):
            d = corpus.date.get(pid)
            di = date_int(d)
            legal = di <= as_i
            nodes = corpus.nodes_for_paper(pid)
            grp = int(corpus.group[nodes[0]]) if nodes.size else -1
            own_id = corpus.group_base[grp] if grp >= 0 else ""
            bad = (not legal) or pid == s or grp == s_group
            yield dict(base, row_kind=role, object_id=pid, object_kind="paper",
                       object_date=d or "", temporally_legal=legal,
                       strictly_before_cutoff=di < as_i,
                       is_source_paper=pid == s, is_anchor=role == "anchor",
                       is_target=role == "target", owner_paper_id=own_id,
                       edge_created_by_source=bool(grp == s_group), violation=bool(bad),
                       violation_reason=("" if not bad else
                                         ("target_or_anchor_after_cutoff" if not legal
                                          else "equals_source_paper")))


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="master evidence_bundle_completion directory")
    ap.add_argument("--seed", type=int, default=EBC_SEED)
    ap.add_argument("--max-screening", type=int, default=MAX_SCREENING_INSTANCES)
    ap.add_argument("--rebuild", action="store_true", help="rebuild ebc_instances.jsonl")
    ap.add_argument("--exclude-ids", default=None,
                    help="file of instance ids held out as development examples (runbook 0.2)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.rebuild or not INSTANCES.exists():
        print("[1/5] building instances ...", flush=True)
        rows, funnel = build_instances(a.seed)
    else:
        print("[1/5] reusing", INSTANCES, flush=True)
        rows = list(read_jsonl(INSTANCES))
        funnel = Counter(reused=len(rows))
    print("   funnel:", json.dumps(dict(funnel), indent=1))

    test = [r for r in rows if r["split"] == "test"]
    dev_ids = set()
    if a.exclude_ids and Path(a.exclude_ids).exists():
        dev_ids = {l.strip() for l in open(a.exclude_ids) if l.strip()}
        test = [r for r in test if r["instance_id"] not in dev_ids]
        print(f"   held out {len(dev_ids)} development instance ids (disjoint from evaluation)")
    rr = random.Random(a.seed + 1)
    order = sorted(test, key=lambda r: r["instance_id"])
    rr.shuffle(order)
    screening = sorted(order[:a.max_screening], key=lambda r: r["instance_id"])
    print(f"[2/5] test split n={len(test)} -> frozen screening set n={len(screening)}", flush=True)

    print("[3/5] loading corpus + graphs (full, rewire) ...", flush=True)
    corpus = EBCCorpus(("full", "rewire"))
    print(f"   papers={len(corpus.paper_ids)} nodes={len(corpus.nodes)} "
          f"edges_full={corpus.graphs['full'].n_edges} edges_rewire={corpus.graphs['rewire'].n_edges}",
          flush=True)

    print("[4/5] structural GraphHard labelling (BM25 + correct-graph reachability) ...", flush=True)
    labels = structural_labels(corpus, screening)
    write_jsonl(LABELS, labels)
    lab = {l["instance_id"]: l for l in labels}
    hard_ids = [l["instance_id"] for l in labels if l["graphhard"]]
    gen_ids = [r["instance_id"] for r in screening]

    frozen = []
    for r in screening:
        L = lab[r["instance_id"]]
        slices = ["general"] + (["graphhard"] if L["graphhard"] else [])
        frozen.append(dict(instance_id=r["instance_id"], slices=slices,
                           source_paper_id=r["source_paper_id"], split=r["split"],
                           as_of=r["as_of"], n_citations=r["n_citations"],
                           anchor_paper_id=r["anchor_paper_id"],
                           target_paper_ids=r["target_paper_ids"],
                           n_targets=len(r["target_paper_ids"]),
                           paragraph_global_id=r["paragraph_global_id"]))
    write_jsonl(out / "frozen_ids.jsonl", frozen)

    print("[5/5] construction temporal audit ...", flush=True)
    with open(out / "temporal_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        w.writeheader()
        n_viol = 0
        for row in construction_audit_rows(corpus, screening):
            n_viol += bool(row["violation"])
            w.writerow(row)
    print(f"   construction violations: {n_viol}")

    hashes = dict(
        general_ids_sha256=sha256_list(gen_ids),
        graphhard_ids_sha256=sha256_list(hard_ids),
        frozen_ids_file_sha256=file_hash(out / "frozen_ids.jsonl"),
        ebc_instances_sha256=file_hash(INSTANCES, limit=1 << 30),
        citation_graph_full_sha256=file_hash(GRAPH_DIR / "citation_graph_full.jsonl", limit=1 << 30),
        citation_graph_rewire_sha256=file_hash(GRAPH_DIR / "citation_graph_rewire.jsonl", limit=1 << 30),
        paragraph_citation_edges_sha256=file_hash(PROV / "paragraph_citation_edges.csv", limit=1 << 30),
    )
    bundle_sizes = Counter(len(r["target_paper_ids"]) for r in screening)
    n_hard_targets = sum(len(r["target_paper_ids"]) for r in screening
                         if lab[r["instance_id"]]["graphhard"])
    man = dict(
        name="ebc_construction", task=TASK_NAME, generated_at=now_iso(), git_commit=git_commit(),
        seed=a.seed, parameters=PARAMS, funnel=dict(funnel),
        inputs=dict(paragraph_citation_edges=str(PROV / "paragraph_citation_edges.csv"),
                    papers_table="postgresql://localhost:5433/postgres papers",
                    paragraphs_table="postgresql://localhost:5433/postgres paragraphs",
                    correct_graph=str(GRAPH_DIR / "citation_graph_full.jsonl"),
                    rewired_graph=str(GRAPH_DIR / "citation_graph_rewire.jsonl")),
        task_definition=dict(
            input="source paragraph with EVERY citation marker normalised to [CITATION]",
            revealed="exactly one anchor citation (id + title), drawn uniformly at random per "
                     "paragraph under a fixed per-paragraph seed",
            targets="the remaining 1-4 resolved citations of the same paragraph",
            bundle_size_range=[MIN_CITATIONS, MAX_CITATIONS]),
        leakage_policy=dict(
            targets_and_anchor_available_by_cutoff=True,
            cutoff="papers.submit_date of the source paper (day granularity)",
            source_paper_excluded_from_candidate_pools=True,
            every_edge_emitted_by_source_removed="node mask owner(v) != paper:s removes paper:s and "
                                                 "every paragraph of s, hence every citation and "
                                                 "paragraph edge created by s, in BOTH variants",
            evaluation_paragraph_cannot_create_anchor_target_relation="its para node is owned by s "
                                                                      "and therefore masked",
            traversal_evidence="every traversed node must satisfy avail(v) <= t_s; paper nodes with "
                               "unknown dates are never traversable",
            strict_rule="paragraphs with ANY unresolved-date or post-cutoff citation are dropped "
                        "entirely, so the evaluated bundle is the complete real bundle"),
        splits=dict(by="source paper (+ near-duplicate paragraph group)",
                    fractions=list(SPLIT_FRACTIONS),
                    n_train=funnel.get("12_split_train"), n_validation=funnel.get("12_split_validation"),
                    n_test=funnel.get("12_split_test")),
        frozen=dict(general_n=len(gen_ids), graphhard_n=len(hard_ids),
                    graphhard_target_count=n_hard_targets,
                    bundle_size_histogram={str(k): v for k, v in sorted(bundle_sizes.items())},
                    screening_selection=f"seeded shuffle of the test split, first {a.max_screening}",
                    development_ids_held_out=sorted(dev_ids),
                    n_development_ids_held_out=len(dev_ids),
                    frozen_before_any_comparative_run=True,
                    graphhard_definition=dict(
                        at_least_one_target_outside_bm25_topk=BM25_HARD_TOPK,
                        reachable_from_anchor_within_hops=GRAPHHARD_MAX_HOPS,
                        graph="correct pre-cutoff graph, source-paper edges removed, "
                              "witness intermediate must itself be legal")),
        hashes=hashes,
        corpus=dict(n_papers=len(corpus.paper_ids), n_nodes=len(corpus.nodes),
                    n_edges_full=corpus.graphs["full"].n_edges,
                    n_edges_rewire=corpus.graphs["rewire"].n_edges,
                    n_paper_nodes=int(corpus.is_paper.sum()),
                    n_paper_nodes_with_date=int(((corpus.avail < INF_DATE) & corpus.is_paper).sum())),
    )
    write_json(out / "construction_manifest.json", man)
    print(json.dumps(dict(general=len(gen_ids), graphhard=len(hard_ids), **hashes), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
