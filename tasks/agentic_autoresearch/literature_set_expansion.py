"""Section 4: Literature Set Expansion / Automated Snowballing (CPU-only, no LLM).

TASK
    For a source paper S the system sees ONLY what a researcher would have when formulating a
    related-work search: S's title and abstract (citation markers scrubbed).  It must return a
    ranked set of EARLIER papers.

WEAK SUPERVISION -- READ THIS BEFORE INTERPRETING ANY NUMBER
    The label set is S's own resolved earlier references.  An author's bibliography is
    INCOMPLETE and SELECTIVE: it omits relevant work the authors did not know, chose not to
    cite, or cited without a resolvable arXiv identifier, and it includes work cited for
    reasons other than topical relevance.  It is a REPRODUCIBLE WEAK RELEVANCE LABEL, NOT
    EXHAUSTIVE GROUND TRUTH.  Absolute recall is therefore an underestimate of true topical
    recall and must never be read as "the system missed N relevant papers".  Only the PAIRED
    differences between arms -- which share the identical label set per instance -- are
    interpretable as system comparisons.

PREDECLARED RULES (fixed before any retrieval arm was executed; see construction_manifest.json)
  eligibility
    E1  S is in the paper corpus with a parseable submit_date, a non-empty title, and an
        abstract of >= 200 characters.
    E2  S has >= 5 gold references (resolved arXiv targets that are in the corpus, dated
        <= S's cutoff, and != S).
    E3  S has <= 200 gold references (long survey bibliographies are a different task).
    E4  S's unresolved-reference rate < 0.90, where
            unresolved_rate = 1 - n_gold_raw / n_bib_entries
        with n_bib_entries taken from the full `citations` bibliography table.  Recorded for
        every considered source in temporal_audit.csv whether or not it is excluded.
    E5  S has >= 1 bibliography entry in the resolution table (otherwise nothing to resolve).
  query
    Q1  query text = title + "\n" + abstract, with every LaTeX \\cite*{...} command, bare
        [CITATION]/[MASKED_CITATION] marker and bracketed numeric citation run removed.
        Introduction text was considered and EXCLUDED by predeclared rule: residual bib keys and
        author-year strings in extracted intro text are a citation-marker leakage channel that
        cannot be audited cheaply at this scale.
  temporal / source-edge controls
    T0  PAPER IDENTITY IS RESOLVED ON THE VERSION-STRIPPED BASE ID EVERYWHERE.  The shared
        citation graph mixes spellings: `cites` edges use base ids, but 73,178 of 165,555
        `has_paragraph` edges carry an explicit version, so 6,198 base papers exist as two
        unconnected nodes `paper:X` and `paper:Xv*`.  Every adjacency is rebuilt with paper
        nodes merged onto the base id, identically in every variant, before anything else
        happens.  Without this, masking on the literal source id leaves the source reachable
        through its versioned twin and its own paragraphs stay live (see superseded_v1/).
    T1  a candidate is legal iff its date is known and <= S's cutoff (tasks/.../candidate_utility._legal).
    T2  S itself is never a candidate, under any spelling.
    T3  BLOCKED NODES = {paper:base(S)} U {every paragraph node owned by ANY spelling of S}.
        Graph traversal may neither enter nor leave a blocked node, in any variant.  This
        removes every edge emitted by S: its cites edges, its has_paragraph edges, and every
        para_cites edge from its own paragraphs (which carry its bibliography and its
        co-citation structure).
    T4  the model input never contains the bibliography, citation markers, gold ids, or any
        source-derived co-citation relation.
  splits
    S1  every instance is one source paper, so the split is by source paper by construction.
    S2  additionally a cluster key `{year}-{primary_arxiv_category}` is assigned and a grouped
        20/80 dev/screen split is drawn on CLUSTERS, so no (year, field) cluster spans both.
        Primary screening numbers are reported on `screen`; `dev` is reported descriptively.
  retrieval
    R1  budget K = 50 for every arm; no arm is padded from another arm.
    R2  seeds for graph expansion = top-10 legal BM25 papers; expansion = the existing
        label-free support-ranked BFS (hybrid_retrieval.graph_rank) with MAX_HOPS/PER_NODE_CAP
        from candidate_utility.
    R3  hybrid = reciprocal rank fusion (hybrid_retrieval.rrf, k=60) of the BM25 list and the
        graph list.  correct and rewired hybrids differ ONLY in the backing adjacency.
    R4  semantic retrieval is reported as NOT AVAILABLE: no paper-level title+abstract vector
        index exists in the repository and building one is out of scope (and would need a GPU).

Outputs are append-only and resumable: per-instance rows go to retrieval_per_instance.jsonl
keyed by source_id and are never recomputed unless --force.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

from .candidate_utility import MAX_HOPS, PER_NODE_CAP, _legal
from .common import now_iso, git_commit, read_jsonl, write_json
from .evaluate import (ndcg_at_k, paired_bootstrap_ci, paired_permutation_p, recall_at_k, holm)
from .hybrid_retrieval import RRF_K, rrf
from .temporal_store import normalize_ts

# ---------------------------------------------------------------- predeclared constants
K_FINAL = 50
KS = (20, 50)
N_SEEDS = 10
# BM25 depth scanned before temporal filtering.  It must be deep enough that the BM25 arm can
# actually fill its K=50 budget with LEGAL (pre-cutoff) papers: for an early source paper most of
# the corpus is in its future, so a shallow scan would silently starve the BM25 arm and every
# arm fused with it.  Declared before any arm was scored.
N_BM25_SCAN = 20000
MIN_GOLD = 5
MAX_GOLD = 200
MIN_ABSTRACT_CHARS = 200
MAX_UNRESOLVED_RATE = 0.90
TARGET_N_SOURCES = 1200
DEV_FRACTION = 0.20
FREEZE_SEED = 20260925
BOOT_N = 10000
PERM_N = 10000
STAT_SEED = 20260925

CITE_CMD = re.compile(r"\\[a-zA-Z]*cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\])*\s*\{[^}]*\}")
MARKER = re.compile(r"\[(?:MASKED_)?CITATION\]")
NUMERIC_CITE = re.compile(r"\[\s*\d+(?:\s*[,\u2013-]\s*\d+)*\s*\]")
NONALNUM = re.compile(r"[^a-z0-9]+")
VERSION_SUFFIX = re.compile(r"v\d+$")

ARMS_PRIMARY = ["bm25", "graph_correct", "graph_rewired", "hybrid_correct", "hybrid_rewired"]
ARMS_DESCRIPTIVE = ["graph_correct_citation_only", "graph_correct_paragraph_only",
                    "hybrid_correct_citation_only"]
ALL_ARMS = ARMS_PRIMARY + ARMS_DESCRIPTIVE
FAMILY_RELATIONS = {"graph_correct_citation_only": {"cites"},
                    "graph_correct_paragraph_only": {"para_cites", "has_paragraph"}}


# ---------------------------------------------------------------- small helpers

def base_pid(pid):
    """T0: version-stripped arXiv identity.  '2002.08165v2' -> '2002.08165'."""
    return VERSION_SUFFIX.sub("", str(pid))


def base_node(node):
    """T0 applied to a graph node id.  Only paper nodes carry version suffixes."""
    if node.startswith("paper:"):
        return "paper:" + base_pid(node[6:])
    return node


def load_graphs_base_normalised(variants):
    """Build adjacencies with paper identity resolved on the BASE id, in every variant.

    Returns (adj, para_owner_base, identity_report).  `para_owner_base` maps every paragraph node
    to the BASE id of its owning paper, taken from the unperturbed `full` graph (paragraph
    ownership is ground truth about which paper contains which paragraph, not a relation under
    test, so it must be identical in every condition -- same convention as
    run_agent.load_literature_corpus).
    """
    adj, report = {}, {}
    for v in variants:
        d = defaultdict(list)
        seen = set()
        n_raw = n_selfloop = n_dup = n_renamed = 0
        for e in read_jsonl(GRAPH_DIR / f"citation_graph_{v}.jsonl"):
            n_raw += 1
            s, t, r = base_node(e["s"]), base_node(e["t"]), e["r"]
            if s != e["s"] or t != e["t"]:
                n_renamed += 1
            if s == t:                       # self-loop created by the merge
                n_selfloop += 1
                continue
            if (s, t, r) in seen:            # duplicate created by the merge
                n_dup += 1
                continue
            seen.add((s, t, r))
            d[s].append((t, r, "out"))
            d[t].append((s, r, "in"))
        adj[v] = d
        report[v] = dict(edges_read=n_raw, edges_kept=len(seen), endpoints_renamed=n_renamed,
                         merge_self_loops_dropped=n_selfloop, merge_duplicates_dropped=n_dup,
                         n_nodes=len(d))
    para_owner = {}
    for e in read_jsonl(GRAPH_DIR / "citation_graph_full.jsonl"):
        if e["r"] == "has_paragraph":
            para_owner[e["t"]] = base_pid(e["s"].split(":", 1)[1])
    return adj, para_owner, report


def degree_equality_check(adj_a, adj_b):
    """After merging, correct and rewired must still have element-wise identical per-relation
    degree sequences -- the whole point of the degree-preserving control."""
    def profile(d):
        out, inn = defaultdict(Counter), defaultdict(Counter)
        for node, nbrs in d.items():
            for _nbr, rel, direction in nbrs:
                (out if direction == "out" else inn)[rel][node] += 1
        return ({r: sorted(c.values()) for r, c in out.items()},
                {r: sorted(c.values()) for r, c in inn.items()})
    ao, ai = profile(adj_a)
    bo, bi = profile(adj_b)
    return dict(out_degree_sequences_identical=(ao == bo),
                in_degree_sequences_identical=(ai == bi),
                relations=sorted(set(ao) | set(bo)))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def scrub_query(title, abstract):
    """Q1.  Returns (text, n_markers_removed)."""
    raw = f"{title or ''}\n{abstract or ''}"
    n = len(CITE_CMD.findall(raw)) + len(MARKER.findall(raw)) + len(NUMERIC_CITE.findall(raw))
    out = CITE_CMD.sub(" ", raw)
    out = MARKER.sub(" ", out)
    out = NUMERIC_CITE.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip(), n


def _entropy(labels):
    labels = [x for x in labels if x]
    if not labels:
        return 0.0
    n = len(labels)
    return -sum((c / n) * math.log(c / n, 2) for c in Counter(labels).values())


def norm_title(t):
    return NONALNUM.sub("", str(t or "").lower())[:120]


def average_precision_at_k(ranked, gold, k):
    """AP@k normalised by min(|gold|, k) so it is attainable at the fixed budget."""
    gold = set(gold)
    if not gold:
        return 0.0
    hits, ssum = 0, 0.0
    for i, x in enumerate(ranked[:k], 1):
        if x in gold:
            hits += 1
            ssum += hits / i
    denom = min(len(gold), k)
    return ssum / denom if denom else 0.0


class FastBM25:
    """Vectorised scorer built FROM an existing environment.BM25 index.

    Scores are the same formula and the same per-unique-query-term summation as
    environment.BM25.search; only the summation is done with scipy instead of Python loops.
    Verified against the reference implementation (see construction_manifest.json).
    """

    def __init__(self, bm):
        import numpy as np
        from scipy import sparse
        self.ids = list(bm.ids)
        self.pos = {d: i for i, d in enumerate(self.ids)}
        vocab = {}
        rows, cols, vals = [], [], []
        k1, b, avg = bm.k1, bm.b, bm.avg
        for did, c in bm.tf.items():
            i = self.pos[did]
            L = bm.len[did]
            denom_l = k1 * (1 - b + b * L / avg)
            for w, f in c.items():
                j = vocab.setdefault(w, len(vocab))
                rows.append(i)
                cols.append(j)
                vals.append(bm.idf[w] * f * (k1 + 1) / (f + denom_l))
        self.vocab = vocab
        self.M = sparse.csr_matrix((np.asarray(vals, dtype="float64"),
                                    (np.asarray(rows), np.asarray(cols))),
                                   shape=(len(self.ids), len(vocab))).tocsc()
        self.np = np

    def search(self, query, k):
        np = self.np
        terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if t in self.vocab}
        if not terms:
            return []
        cols = sorted(self.vocab[t] for t in terms)
        s = np.asarray(self.M[:, cols].sum(axis=1)).ravel()
        nz = np.flatnonzero(s)
        if nz.size == 0:
            return []
        take = nz[np.argsort(-s[nz], kind="stable")[: max(k, 1)]]
        out = sorted(((self.ids[i], float(s[i])) for i in take), key=lambda x: (-x[1], str(x[0])))
        return out[:k]

    def verify_against(self, bm, queries, k=50):
        """Top-k agreement with the reference environment.BM25 implementation."""
        rep = []
        for q in queries:
            a = [p for p, _ in bm.search(q, k)]
            b = [p for p, _ in self.search(q, k)]
            rep.append(dict(exact_order_match=a == b,
                            set_overlap=len(set(a) & set(b)) / max(1, len(a))))
        return dict(n_queries=len(rep),
                    exact_order_match_rate=_mean([float(r["exact_order_match"]) for r in rep]),
                    mean_topk_set_overlap=_mean([r["set_overlap"] for r in rep]))


# ---------------------------------------------------------------- graph expansion (blocked)

def graph_rank_blocked(query_seeds, adj, papers, as_of, source_id, blocked,
                       allowed_relations=None, k=K_FINAL, max_hops=MAX_HOPS, cap=PER_NODE_CAP,
                       allow_source_as_waypoint=False):
    """hybrid_retrieval.graph_rank with T3 blocked-node enforcement.

    Returns (ranked_ids, support, audit) where audit counts every traversal attempt that was
    refused because it touched a node emitted by the source paper.

    `allow_source_as_waypoint=True` is ONLY for the negative-control audit: it lets the traversal
    pass THROUGH the source paper node (so the source's own citation edges become live paths)
    while still never scoring the source itself as a candidate.  It is never used for any
    reported arm.
    """
    support, hops = defaultdict(float), {}
    blocked_hits = 0
    future_blocked = 0
    for seed_rank, seed in enumerate(query_seeds):
        start = f"paper:{seed}"
        if start in blocked:
            continue
        q, seen = deque([(start, 0)]), {start}
        while q:
            node, d = q.popleft()
            if d >= max_hops:
                continue
            for nbr, rel, _dir in sorted(adj.get(node, []), key=lambda x: x[0])[:cap]:
                if nbr in blocked:
                    blocked_hits += 1
                    continue
                if allowed_relations is not None and rel not in allowed_relations:
                    continue
                if nbr in seen:
                    continue
                seen.add(nbr)
                if nbr.startswith("paper:"):
                    pid = base_pid(nbr.split(":", 1)[1])
                    if pid == source_id:
                        blocked_hits += 1
                        if not allow_source_as_waypoint:
                            continue
                        q.append((nbr, d + 1))
                        continue
                    if _legal(pid, papers, as_of):
                        support[pid] += 1.0 / ((1 + seed_rank) * (2 + d))
                        hops[pid] = min(hops.get(pid, 99), d + 1)
                    else:
                        future_blocked += 1
                    q.append((nbr, d + 1))
                elif nbr.startswith("para:"):
                    q.append((nbr, d + 1))
    ranked = sorted(support, key=lambda p: (-support[p], hops.get(p, 99), p))[:k]
    return ranked, support, dict(blocked_node_hits=blocked_hits,
                                 future_candidates_filtered=future_blocked,
                                 reachable_legal_papers=len(support))


# ---------------------------------------------------------------- construction

def load_paper_table():
    """Identical id space to run_agent.load_literature_corpus (same SQL)."""
    import pandas as pd
    from .common import db_conn
    conn = db_conn()
    df = pd.read_sql("select coalesce(base_arxiv_id, arxiv_id) as pid, title, abstract, "
                     "min(submit_date::text) as date from papers group by 1,2,3", conn)
    papers = {}
    for r in df.itertuples():
        papers[r.pid] = dict(title=r.title, abstract=r.abstract, date=r.date)
    return papers, conn


def primary_categories(conn):
    """metadata['categories'][0] where available, else the lowest-id linked category."""
    cur = conn.cursor()
    cur.execute("select coalesce(base_arxiv_id, arxiv_id), metadata from papers "
                "where metadata is not null")
    out = {}
    for pid, meta in cur.fetchall():
        try:
            m = ast.literal_eval(meta) if isinstance(meta, str) else meta
            cats = m.get("categories")
            if isinstance(cats, str):
                cats = ast.literal_eval(cats)
            if cats:
                out.setdefault(pid, cats[0])
        except Exception:
            continue
    cur.execute("select pc.paper_arxiv_id, min(c.name) from paper_category pc "
                "join categories c on c.id = pc.category_id group by 1")
    for pid, name in cur.fetchall():
        out.setdefault(pid, name)
    return out


def build(out_dir, limit_sources=None):
    import pandas as pd
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    papers, conn = load_paper_table()
    print(f"corpus papers: {len(papers)}", flush=True)
    cats = primary_categories(conn)

    # full bibliography sizes (denominator of the unresolved-reference rate)
    cur = conn.cursor()
    cur.execute("select regexp_replace(citing_arxiv_id, 'v[0-9]+$', ''), count(*) "
                "from citations group by 1")
    n_bib = {pid: n for pid, n in cur.fetchall()}

    res = pd.read_csv(REPO_PROV / "citation_resolution.csv", dtype=str)
    res["src"] = res.citing_arxiv_id.str.replace(r"v\d+$", "", regex=True)
    res["tgt"] = res.arxiv_target.str.replace(r"v\d+$", "", regex=True)

    by_src_entries = res.groupby("src").size().to_dict()
    resolved = res[res.tgt.notna()]
    by_src_targets = resolved.groupby("src")["tgt"].apply(lambda s: sorted(set(s))).to_dict()

    # T0 exposure audit: how badly the raw graph mixes arXiv id spellings, and how many frozen
    # sources would have escaped a literal-id source mask.  Recorded whether or not it bites.
    raw_paper_nodes, versioned_nodes = set(), set()
    raw_para_owner = {}
    rel_touching_version = Counter()
    for e in read_jsonl(GRAPH_DIR / "citation_graph_full.jsonl"):
        touched = False
        for node in (e["s"], e["t"]):
            if node.startswith("paper:"):
                pid_raw = node[6:]
                raw_paper_nodes.add(pid_raw)
                if VERSION_SUFFIX.search(pid_raw):
                    versioned_nodes.add(pid_raw)
                    touched = True
        if touched:
            rel_touching_version[e["r"]] += 1
        if e["r"] == "has_paragraph":
            raw_para_owner[e["t"]] = e["s"].split(":", 1)[1]
    base_groups = defaultdict(set)
    for pid_raw in raw_paper_nodes:
        base_groups[base_pid(pid_raw)].add(pid_raw)
    identity_exposure = dict(
        raw_paper_nodes=len(raw_paper_nodes), distinct_base_ids=len(base_groups),
        versioned_paper_nodes=len(versioned_nodes),
        base_ids_present_under_both_spellings=sum(1 for g in base_groups.values() if len(g) > 1),
        edges_touching_a_versioned_paper_id=dict(rel_touching_version))

    # graph cites out-edges, used only as an independent cross-check of the gold set
    cites_out = defaultdict(set)
    for e in read_jsonl(GRAPH_DIR / "citation_graph_full.jsonl"):
        if e["r"] == "cites" and e["s"].startswith("paper:"):
            cites_out[base_pid(e["s"].split(":", 1)[1])].add(
                base_pid(e["t"].split(":", 1)[1]))

    audit_rows = []
    eligible = []
    for src in sorted(by_src_entries):
        p = papers.get(src)
        as_of = normalize_ts(p["date"]) if p else None
        n_entries = int(by_src_entries.get(src, 0))
        n_bib_entries = int(n_bib.get(src, n_entries))
        raw_targets = [t for t in by_src_targets.get(src, []) if t and t != src]
        n_gold_raw = len(raw_targets)
        gold, future_refs, offcorpus = [], 0, 0
        if as_of:
            for t in raw_targets:
                if t not in papers:
                    offcorpus += 1
                elif _legal(t, papers, as_of):
                    gold.append(t)
                else:
                    future_refs += 1
        unresolved_rate = 1.0 - (n_gold_raw / n_bib_entries) if n_bib_entries else 1.0
        abstract = (p or {}).get("abstract") or ""
        title = (p or {}).get("title") or ""
        reasons = []
        if p is None:
            reasons.append("E1_not_in_corpus")
        else:
            if not as_of:
                reasons.append("E1_no_parseable_date")
            if not str(title).strip():
                reasons.append("E1_no_title")
            if len(str(abstract)) < MIN_ABSTRACT_CHARS:
                reasons.append("E1_abstract_too_short")
        if n_entries < 1:
            reasons.append("E5_no_bib_entries")
        if len(gold) < MIN_GOLD:
            reasons.append("E2_too_few_gold")
        if len(gold) > MAX_GOLD:
            reasons.append("E3_too_many_gold")
        if unresolved_rate >= MAX_UNRESOLVED_RATE:
            reasons.append("E4_unresolved_rate")
        row = dict(source_id=src, in_corpus=int(p is not None), as_of=as_of or "",
                   n_bib_entries=n_bib_entries, n_resolution_entries=n_entries,
                   n_resolved_targets=n_gold_raw,
                   n_targets_off_corpus=offcorpus, n_targets_after_cutoff=future_refs,
                   n_gold=len(gold), unresolved_rate=round(unresolved_rate, 4),
                   abstract_chars=len(str(abstract)),
                   gold_matches_graph_cites=int(set(gold) == set(cites_out.get(src, set())))
                   if p is not None else 0,
                   n_graph_cites_out=len(cites_out.get(src, set())),
                   primary_category=cats.get(src, "UNKNOWN"),
                   eligible=int(not reasons), exclusion_reasons="|".join(reasons))
        audit_rows.append(row)
        if not reasons:
            eligible.append((src, gold, row))

    with open(out_dir / "temporal_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        w.writeheader()
        w.writerows(audit_rows)
    print(f"considered {len(audit_rows)} sources, eligible {len(eligible)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- freeze
    rng = random.Random(FREEZE_SEED)
    pool = sorted(eligible, key=lambda x: x[0])
    n_take = min(TARGET_N_SOURCES, len(pool)) if limit_sources is None else min(limit_sources, len(pool))
    idx = list(range(len(pool)))
    rng.shuffle(idx)
    chosen = sorted(idx[:n_take])
    frozen = [pool[i] for i in chosen]

    # S2: grouped dev/screen split on (year, primary category) clusters
    clusters = sorted({f"{r['as_of'][:4]}-{r['primary_category']}" for _, _, r in frozen})
    crng = random.Random(FREEZE_SEED + 1)
    corder = clusters[:]
    crng.shuffle(corder)
    csize = Counter(f"{r['as_of'][:4]}-{r['primary_category']}" for _, _, r in frozen)
    dev_clusters, dev_n = set(), 0
    for c in corder:                       # whole clusters until ~DEV_FRACTION of INSTANCES
        if dev_n >= DEV_FRACTION * len(frozen):
            break
        dev_clusters.add(c)
        dev_n += csize[c]

    instances = []
    for src, gold, row in frozen:
        p = papers[src]
        query, n_scrub = scrub_query(p["title"], p["abstract"])
        cluster = f"{row['as_of'][:4]}-{row['primary_category']}"
        instances.append(dict(
            source_id=src, as_of=row["as_of"], cluster=cluster,
            split=("dev" if cluster in dev_clusters else "screen"),
            title=p["title"], query=query, query_markers_removed=n_scrub,
            gold=gold, n_gold=len(gold),
            n_bib_entries=row["n_bib_entries"], unresolved_rate=row["unresolved_rate"],
            primary_category=row["primary_category"]))
    instances.sort(key=lambda r: r["source_id"])

    frozen_path = out_dir / "frozen_ids.jsonl"
    with open(frozen_path, "w") as f:
        for r in instances:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    frozen_hash = sha256_file(frozen_path)

    id_only = out_dir / "frozen_ids_only.txt"
    id_only.write_text("\n".join(r["source_id"] for r in instances) + "\n")

    n_screen = sum(1 for r in instances if r["split"] == "screen")
    manifest = dict(
        task="literature_set_expansion", generated_at=now_iso(), git_commit=git_commit(),
        weak_supervision_statement=(
            "Labels are the source paper's own resolved earlier references. An author's "
            "bibliography is INCOMPLETE and SELECTIVE; this is a reproducible WEAK relevance "
            "label, NOT exhaustive ground truth. Absolute recall understates true topical "
            "recall. Only paired between-arm differences on identical label sets are "
            "interpretable as system comparisons."),
        predeclared_rules=dict(
            eligibility=dict(E1="in corpus, parseable date, non-empty title, abstract >= "
                                f"{MIN_ABSTRACT_CHARS} chars",
                             E2=f"n_gold >= {MIN_GOLD}", E3=f"n_gold <= {MAX_GOLD}",
                             E4=f"unresolved_rate < {MAX_UNRESOLVED_RATE}",
                             E5="at least one bibliography entry in the resolution table"),
            query="title + abstract, LaTeX cite commands / [CITATION] markers / bracketed "
                  "numeric citation runs removed; introduction text excluded by predeclared rule",
            temporal="candidate legal iff date known and <= source cutoff (candidate_utility._legal)",
            source_edge_removal="blocked nodes = {paper:S} U {paragraph nodes owned by S}; "
                                "traversal may neither enter nor leave a blocked node",
            splits="by source paper (one instance per source) plus a grouped 20/80 dev/screen "
                   "split on (year, primary arXiv category) clusters",
            retrieval=dict(budget_K=K_FINAL, seeds=N_SEEDS, max_hops=MAX_HOPS,
                           per_node_cap=PER_NODE_CAP, fusion=f"RRF k={RRF_K}", no_padding=True),
            semantic_arm="NOT AVAILABLE: no paper-level title+abstract vector index exists in "
                         "the repository; building one is out of scope for this CPU-only stage"),
        corpus=dict(n_papers=len(papers),
                    graph_full_sha256=sha256_file(GRAPH_DIR / "citation_graph_full.jsonl"),
                    graph_rewire_sha256=sha256_file(GRAPH_DIR / "citation_graph_rewire.jsonl"),
                    citation_resolution_sha256=sha256_file(REPO_PROV / "citation_resolution.csv")),
        funnel=dict(sources_considered=len(audit_rows),
                    eligible=len(eligible), frozen=len(instances),
                    dev=len(instances) - n_screen, screen=n_screen,
                    exclusion_histogram=dict(Counter(
                        r["exclusion_reasons"] for r in audit_rows if r["exclusion_reasons"]))),
        unresolved_reference_rates=dict(
            mean_over_considered=round(sum(r["unresolved_rate"] for r in audit_rows) / max(1, len(audit_rows)), 4),
            mean_over_frozen=round(sum(r["unresolved_rate"] for r in instances) / max(1, len(instances)), 4)),
        gold_cross_check=dict(
            frozen_sources_whose_gold_equals_graph_cites_out=sum(
                1 for _, _, r in frozen if r["gold_matches_graph_cites"]),
            n_frozen=len(instances),
            note="gold is recomputed from citation_resolution.csv + corpus + temporal filter; "
                 "the graph's cites out-edges are an independent cross-check only and are "
                 "REMOVED from every traversal by rule T3"),
        gold_size=dict(mean=round(sum(r["n_gold"] for r in instances) / max(1, len(instances)), 2),
                       median=sorted(r["n_gold"] for r in instances)[len(instances) // 2] if instances else 0,
                       min=min((r["n_gold"] for r in instances), default=0),
                       max=max((r["n_gold"] for r in instances), default=0)),
        freeze=dict(seed=FREEZE_SEED, target_n=TARGET_N_SOURCES, n_frozen=len(instances),
                    frozen_ids_path=str(frozen_path),
                    frozen_ids_sha256=frozen_hash,
                    frozen_ids_sha256_short=frozen_hash[:16],
                    frozen_ids_only_sha256=sha256_file(id_only)),
        n_clusters=len(clusters), n_dev_clusters=len(dev_clusters))

    # what the v1 literal-id mask would have missed on THIS frozen set
    owner_by_base = defaultdict(set)
    for para, owner_raw in raw_para_owner.items():
        owner_by_base[base_pid(owner_raw)].add(owner_raw)
    exposed_sources, exposed_paras = 0, 0
    for r in instances:
        s = r["source_id"]
        extra = {o for o in owner_by_base.get(s, set()) if o != s}
        if extra:
            exposed_sources += 1
            exposed_paras += sum(1 for _p, o in raw_para_owner.items()
                                 if base_pid(o) == s and o != s)
    manifest["graph_identity"] = dict(
        rule=("T0: paper node identity is resolved on the version-stripped base id at "
              "adjacency-build time, identically in every graph variant"),
        raw_graph_exposure=identity_exposure,
        frozen_set_exposure_under_a_literal_id_mask=dict(
            sources_owning_paragraphs_under_a_versioned_id=exposed_sources,
            n_frozen=len(instances),
            fraction=round(exposed_sources / max(1, len(instances)), 4),
            source_owned_paragraph_nodes_left_unmasked=exposed_paras,
            consequence=("without T0 those source paragraphs stay live, so a traversal can reach "
                         "a paper the source cites, step back into the source's own paragraph, "
                         "and step forward to every other paper cited there -- source-derived "
                         "co-citation leakage forbidden by runbook 4.2"),
            status="NEUTRALISED by T0 in this (v2) build"),
        note=("the direct cites source->gold edges were never affected: 0 of the cites edges "
              "carry a version suffix; the leak channel was has_paragraph/para_cites only"))
    manifest["refreeze"] = dict(
        reason=("the shared citation graph mixed arXiv id spellings, defeating literal-id "
                "source-edge masking; construction was rebuilt from scratch after fixing "
                "identity resolution (rule T0). The re-freeze was NOT performed on the basis of "
                "any observed arm performance."),
        superseded_artifacts_preserved_in="superseded_v1/",
        superseded_frozen_ids_sha256=(
            "5b7eadacd40f892a0f918cef4510649059f01ee11d2c8fa93ca4384a509e4553"),
        eligibility_depends_on_graph=False,
        note=("eligibility and sampling use only the `papers` table and citation_resolution.csv, "
              "so the frozen id set is expected to be unchanged by the graph fix; compare "
              "freeze.frozen_ids_sha256 with superseded_frozen_ids_sha256"))
    manifest["refreeze"]["frozen_ids_unchanged"] = (
        frozen_hash == manifest["refreeze"]["superseded_frozen_ids_sha256"])
    write_json(out_dir / "construction_manifest.json", manifest)
    print(f"frozen {len(instances)} sources  sha256={frozen_hash[:16]}  "
          f"(dev={len(instances)-n_screen}, screen={n_screen})", flush=True)
    return manifest


# ---------------------------------------------------------------- screening

def screen(out_dir, limit=None, force=False):
    from .run_agent import load_literature_corpus
    out_dir = Path(out_dir)
    insts = list(read_jsonl(out_dir / "frozen_ids.jsonl"))
    per_path = out_dir / "retrieval_per_instance.jsonl"
    done = set()
    if per_path.exists() and not force:
        for r in read_jsonl(per_path):
            done.add(r["source_id"])
    todo = [i for i in insts if i["source_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"screening: {len(todo)} to run, {len(done)} already complete", flush=True)
    if not todo:
        return

    t0 = time.time()
    papers, bm, _adj_raw, _po_raw = load_literature_corpus(())
    adj, para_owner, ident = load_graphs_base_normalised(("full", "rewire"))
    deg = degree_equality_check(adj["full"], adj["rewire"])
    write_json(out_dir / "graph_identity_audit.json",
               dict(rule="T0: paper node identity resolved on the version-stripped base id, "
                         "identically in every graph variant",
                    per_variant=ident, degree_preservation_after_merge=deg))
    print(f"corpus + 2 base-normalised graph variants loaded ({time.time()-t0:.0f}s); "
          f"degree preservation after merge: {deg}", flush=True)
    assert deg["out_degree_sequences_identical"] and deg["in_degree_sequences_identical"], \
        "rewired variant is no longer degree-preserving after base-id merging"
    fast = FastBM25(bm)
    print(f"fast BM25 built ({time.time()-t0:.0f}s)", flush=True)
    ver = fast.verify_against(bm, [i["query"] for i in insts[:5]], k=50)
    print(f"FastBM25 vs reference BM25: {ver}", flush=True)
    write_json(out_dir / "bm25_equivalence_check.json",
               dict(**ver, note="FastBM25 is a vectorised scorer built from the reference "
                                "environment.BM25 index (same idf/tf/length normalisation and "
                                "the same per-unique-query-term summation)."))

    # reverse paragraph ownership, for T3
    owned = defaultdict(set)
    for para, owner in para_owner.items():
        owned[base_pid(owner)].add(para)

    adjs = {"correct": adj["full"], "rewired": adj["rewire"]}
    from .common import db_conn
    cat = primary_categories(db_conn())
    print(f"primary categories for {len(cat)} papers "
          "(arXiv primary category is the topical-diversity proxy; the corpus has no venue "
          "field, so venue diversity is not computable here)", flush=True)

    fh = open(per_path, "a")
    for n, inst in enumerate(todo, 1):
        row = score_instance(inst, fast, papers, adjs, owned, cat)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        if n % 25 == 0:
            print(f"  {n}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    fh.close()
    print(f"screening done ({time.time()-t0:.0f}s)", flush=True)


def score_instance(inst, fast, papers, adjs, owned, cat=None):
    src = base_pid(inst["source_id"])
    as_of, gold = inst["as_of"], {base_pid(g) for g in inst["gold"]}
    blocked = {f"paper:{src}"} | set(owned.get(src, ()))

    raw = fast.search(inst["query"], N_BM25_SCAN)
    bm_legal, n_future_bm25, n_self = [], 0, 0
    seen_bm = set()
    for pid, _s in raw:
        pid = base_pid(pid)
        if pid in seen_bm:
            continue
        seen_bm.add(pid)
        if pid == src:
            n_self += 1
            continue
        if not _legal(pid, papers, as_of):
            n_future_bm25 += 1
            continue
        bm_legal.append(pid)
    seeds = bm_legal[:N_SEEDS]
    bm_list = bm_legal[:K_FINAL]

    lists, audits = {"bm25": bm_list}, {}
    for name, a in adjs.items():
        g, sup, aud = graph_rank_blocked(seeds, a, papers, as_of, src, blocked)
        lists[f"graph_{name}"] = g
        lists[f"hybrid_{name}"] = rrf([bm_list, g], k=RRF_K, budget=K_FINAL)
        audits[f"graph_{name}"] = aud
    for arm, rels in FAMILY_RELATIONS.items():
        g, _s, aud = graph_rank_blocked(seeds, adjs["correct"], papers, as_of, src, blocked,
                                        allowed_relations=rels)
        lists[arm] = g
        audits[arm] = aud
    lists["hybrid_correct_citation_only"] = rrf(
        [bm_list, lists["graph_correct_citation_only"]], k=RRF_K, budget=K_FINAL)

    out = dict(source_id=src, as_of=as_of, split=inst["split"], cluster=inst["cluster"],
               n_gold=len(gold), arms={})
    for arm, lst in lists.items():
        lst = lst[:K_FINAL]
        titles = [norm_title(papers.get(p, {}).get("title")) for p in lst]
        seen_t, dups = set(), 0
        for t in titles:
            if t and t in seen_t:
                dups += 1
            seen_t.add(t)
        years = [str(papers.get(p, {}).get("date") or "")[:4] for p in lst]
        cats = [(cat or {}).get(p, "UNKNOWN") for p in lst]
        m = dict(n_returned=len(lst), n_unique=len(set(lst)),
                 duplicate_rate=round(dups / len(lst), 4) if lst else 0.0,
                 gold_found=sorted(set(lst) & gold),
                 n_gold_found=len(set(lst) & gold),
                 recall_per_candidate=round((len(set(lst) & gold) / len(gold)) / K_FINAL, 6)
                 if gold else 0.0,
                 distinct_years=len(set(y for y in years if y)),
                 distinct_categories=len(set(c for c in cats if c != "UNKNOWN")),
                 category_entropy=round(_entropy(cats), 4),
                 same_category_as_source_rate=round(
                     sum(1 for c in cats if c == inst.get("primary_category")) / len(lst), 4)
                 if lst else 0.0,
                 map_at_50=average_precision_at_k(lst, gold, 50))
        for k in KS:
            m[f"recall@{k}"] = recall_at_k(lst, gold, k)
            m[f"ndcg@{k}"] = ndcg_at_k(lst, gold, k)
        # temporal leakage audit: every returned candidate must be legal and != source
        m["leak_future_candidates"] = sum(1 for p in lst if not _legal(p, papers, as_of))
        m["leak_source_in_results"] = sum(1 for p in lst if p == src)
        out["arms"][arm] = m
    for arm in ("graph_correct", "graph_rewired"):
        bm_top = set(lists["bm25"])
        out["arms"][arm]["graph_only_gold"] = sorted(
            (set(lists[arm]) & gold) - bm_top)
        out["arms"][arm]["n_graph_only_gold"] = len(out["arms"][arm]["graph_only_gold"])
    for arm in ("hybrid_correct", "hybrid_rewired"):
        bm_top = set(lists["bm25"])
        out["arms"][arm]["n_graph_only_gold"] = len((set(lists[arm]) & gold) - bm_top)
    out["audits"] = audits
    out["bm25_future_filtered"] = n_future_bm25
    out["bm25_self_filtered"] = n_self
    out["bm25_scanned"] = len(raw)
    out["bm25_legal_available"] = len(bm_legal)
    out["bm25_budget_filled"] = int(len(bm_list) >= K_FINAL)
    out["blocked_nodes"] = len(blocked)
    return out


# ---------------------------------------------------------------- aggregation + gate

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(out_dir):
    out_dir = Path(out_dir)
    rows = list(read_jsonl(out_dir / "retrieval_per_instance.jsonl"))
    seen, dedup = set(), []
    for r in rows:                                   # append-only file: keep first occurrence
        if r["source_id"] in seen:
            continue
        seen.add(r["source_id"])
        dedup.append(r)
    rows = sorted(dedup, key=lambda r: r["source_id"])
    if not rows:
        raise SystemExit("no per-instance rows; run screen first")

    subsets = {"frozen_all": rows,
               "screen": [r for r in rows if r["split"] == "screen"],
               "dev": [r for r in rows if r["split"] == "dev"]}

    metric_cols = (["recall@20", "recall@50", "ndcg@20", "ndcg@50", "map_at_50",
                    "duplicate_rate", "recall_per_candidate", "n_returned", "n_gold_found",
                    "distinct_years", "distinct_categories", "category_entropy",
                    "same_category_as_source_rate"])
    res_rows = []
    for sname, rs in subsets.items():
        if not rs:
            continue
        for arm in ALL_ARMS:
            vals = [r["arms"][arm] for r in rs if arm in r["arms"]]
            if not vals:
                continue
            unique_gold = set()
            for r in rs:
                unique_gold |= set(r["arms"][arm].get("gold_found", []))
            d = dict(subset=sname, arm=arm, family=("primary" if arm in ARMS_PRIMARY
                                                    else "descriptive"),
                     n_instances=len(vals))
            for c in metric_cols:
                d[c] = round(_mean([v[c] for v in vals]), 6)
            d["unique_gold_references_recovered"] = len(unique_gold)
            d["mean_graph_only_gold"] = round(_mean([v.get("n_graph_only_gold", 0) for v in vals]), 4)
            d["leak_future_candidates_total"] = sum(v["leak_future_candidates"] for v in vals)
            d["leak_source_in_results_total"] = sum(v["leak_source_in_results"] for v in vals)
            res_rows.append(d)
    with open(out_dir / "retrieval_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res_rows[0].keys()))
        w.writeheader()
        w.writerows(res_rows)

    # ---- paired comparisons
    CONFIRMATORY = [("hybrid_correct", "bm25"), ("hybrid_correct", "hybrid_rewired")]
    EXPLORATORY = [("hybrid_correct", "graph_correct"), ("graph_correct", "graph_rewired"),
                   ("graph_correct", "bm25"), ("hybrid_rewired", "bm25"),
                   ("hybrid_correct", "hybrid_correct_citation_only")]
    comp_rows = []
    pvals_conf = {}
    for sname, rs in subsets.items():
        if not rs:
            continue
        for fam, pairs in (("confirmatory", CONFIRMATORY), ("exploratory", EXPLORATORY)):
            for hi, lo in pairs:
                for metric in ("recall@50", "recall@20", "ndcg@50", "ndcg@20", "map_at_50"):
                    a = [r["arms"][hi][metric] for r in rs]
                    b = [r["arms"][lo][metric] for r in rs]
                    ci = paired_bootstrap_ci(a, b, n_boot=BOOT_N, seed=STAT_SEED)
                    p = paired_permutation_p(a, b, n_perm=PERM_N, seed=STAT_SEED)
                    key = f"{sname}|{hi}>{lo}|{metric}"
                    row = dict(subset=sname, family=fam, hi=hi, lo=lo, metric=metric,
                               n_paired=ci["n"], mean_hi=round(_mean(a), 6),
                               mean_lo=round(_mean(b), 6), diff=round(ci["diff"], 6),
                               ci_lo=round(ci["lo"], 6), ci_hi=round(ci["hi"], 6),
                               p_two_sided=p,
                               ci_excludes_zero=bool(ci["lo"] > 0 or ci["hi"] < 0),
                               positive_ci=bool(ci["lo"] > 0),
                               n_boot=BOOT_N, n_perm=PERM_N, seed=STAT_SEED)
                    comp_rows.append(row)
                    if fam == "confirmatory" and sname == "screen" and metric == "recall@50":
                        pvals_conf[key] = p
    adj = holm(pvals_conf)
    for r in comp_rows:
        key = f"{r['subset']}|{r['hi']}>{r['lo']}|{r['metric']}"
        r["p_holm_confirmatory"] = adj.get(key, {}).get("p_adj")
    with open(out_dir / "retrieval_comparisons.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
        w.writeheader()
        w.writerows(comp_rows)

    # ---- leakage
    leak = dict(
        future_candidates=sum(v["leak_future_candidates"] for r in rows for v in r["arms"].values()),
        source_in_results=sum(v["leak_source_in_results"] for r in rows for v in r["arms"].values()),
        source_edges_blocked_total=sum(a["blocked_node_hits"] for r in rows
                                       for a in r["audits"].values()),
        instances_with_blocked_nodes=sum(1 for r in rows if r["blocked_nodes"] > 0),
        n_instances=len(rows))
    leak["zero_violations"] = bool(leak["future_candidates"] == 0 and leak["source_in_results"] == 0)

    # ---- gate (mechanical)
    def get(sname, hi, lo, metric="recall@50"):
        for r in comp_rows:
            if (r["subset"], r["hi"], r["lo"], r["metric"]) == (sname, hi, lo, metric):
                return r
        return None

    gate = {}
    for sname in ("screen", "frozen_all"):
        vs_bm = get(sname, "hybrid_correct", "bm25")
        vs_rw = get(sname, "hybrid_correct", "hybrid_rewired")
        if not (vs_bm and vs_rw):
            continue
        c1 = vs_bm["diff"] >= 0.05
        c2 = vs_rw["diff"] >= 0.05
        c3 = vs_bm["positive_ci"]
        c4 = vs_rw["positive_ci"]
        c5 = leak["zero_violations"]
        gate[sname] = dict(
            criteria=dict(
                hybrid_correct_minus_bm25_recall50=vs_bm["diff"],
                hybrid_correct_minus_bm25_ci=[vs_bm["ci_lo"], vs_bm["ci_hi"]],
                hybrid_correct_minus_rewired_recall50=vs_rw["diff"],
                hybrid_correct_minus_rewired_ci=[vs_rw["ci_lo"], vs_rw["ci_hi"]],
                n_paired=vs_bm["n_paired"]),
            checks=dict(beats_bm25_by_0_05=bool(c1), beats_rewired_hybrid_by_0_05=bool(c2),
                        positive_ci_vs_bm25=bool(c3), positive_ci_vs_rewired=bool(c4),
                        zero_leakage=bool(c5)),
            decision=("PASS" if all([c1, c2, c3, c4, c5]) else "FAIL"))
    primary = gate.get("screen") or gate.get("frozen_all")
    decision = dict(
        task="literature_set_expansion", decided_at=now_iso(), git_commit=git_commit(),
        rule=("launch literature agents only if correct hybrid exceeds BM25 by >= 0.05 absolute "
              "Recall@50 AND exceeds the rewired hybrid by >= 0.05 absolute Recall@50, with "
              "positive paired 95% CIs for BOTH differences, and zero temporal/source-edge "
              "leakage (runbook 4.3 / stopping rule 9.5)"),
        primary_subset="screen", by_subset=gate,
        leakage_audit=leak,
        decision=primary["decision"] if primary else "UNDETERMINED",
        agents_launched=False,
        agent_action=("PROCEED to 4.4 (GPU stage, owned by another process)"
                      if primary and primary["decision"] == "PASS"
                      else "STOP: do not run literature agents; report retrieval result only"),
        frozen_ids_sha256=sha256_file(out_dir / "frozen_ids.jsonl"),
        weak_label_caveat=("Recall is measured against the source paper's own bibliography, an "
                           "incomplete and selective weak relevance label, not exhaustive ground "
                           "truth. The gate is a decision rule on PAIRED differences under that "
                           "identical label set."))
    write_json(out_dir / "gate_decision.json", decision)
    write_json(out_dir / "aggregate_summary.json",
               dict(subsets={k: len(v) for k, v in subsets.items()},
                    results=res_rows, leakage=leak, gate=gate))
    return res_rows, comp_rows, decision, leak


def verify(out_dir, n_sample=40):
    """Independent post-hoc audit on a random sample, plus a source-edge positive control.

    1. Re-runs every arm for `n_sample` frozen sources and re-checks EVERY returned id against a
       FRESHLY queried submit_date table (not the cached corpus dict used during scoring).
    2. Positive control: re-runs the correct-graph expansion with source-edge blocking DISABLED.
       If blocking were a no-op the two recalls would be identical; a large gap shows the
       source-emitted edges really were carrying the labels and really were removed.
    """
    from .run_agent import load_literature_corpus
    from .common import db_conn
    out_dir = Path(out_dir)
    insts = list(read_jsonl(out_dir / "frozen_ids.jsonl"))
    rng = random.Random(FREEZE_SEED + 7)
    sample = rng.sample(insts, min(n_sample, len(insts)))

    cur = db_conn().cursor()
    cur.execute("select coalesce(base_arxiv_id, arxiv_id), min(submit_date)::text "
                "from papers group by 1")
    fresh_date = {p: d for p, d in cur.fetchall()}

    papers, bm, _a, _p = load_literature_corpus(())
    adj, para_owner, ident = load_graphs_base_normalised(("full", "rewire"))
    fast = FastBM25(bm)
    owned = defaultdict(set)
    for para, owner in para_owner.items():
        owned[base_pid(owner)].add(para)
    adjs = {"correct": adj["full"], "rewired": adj["rewire"]}
    # un-merged adjacency + literal-id paragraph ownership, purely to replay the v1 bug
    adj_raw_v1 = defaultdict(list)
    owned_v1 = defaultdict(set)
    for e in read_jsonl(GRAPH_DIR / "citation_graph_full.jsonl"):
        adj_raw_v1[e["s"]].append((e["t"], e["r"], "out"))
        adj_raw_v1[e["t"]].append((e["s"], e["r"], "in"))
        if e["r"] == "has_paragraph":
            owned_v1[e["s"].split(":", 1)[1]].add(e["t"])

    viol_future, viol_source, viol_unknown, checked = [], [], [], 0
    ctrl = []
    cites_out = defaultdict(set)
    for e in read_jsonl(GRAPH_DIR / "citation_graph_full.jsonl"):
        if e["r"] == "cites" and e["s"].startswith("paper:"):
            cites_out[base_pid(e["s"].split(":", 1)[1])].add(
                base_pid(e["t"].split(":", 1)[1]))
    gold_not_in_source_edges = 0

    for inst in sample:
        src, as_of, gold = inst["source_id"], inst["as_of"], set(inst["gold"])
        gold_not_in_source_edges += len(gold - cites_out.get(src, set()))
        row = score_instance(inst, fast, papers, adjs, owned)
        # rebuild the lists to audit them directly
        blocked = {f"paper:{src}"} | set(owned.get(src, ()))
        raw = fast.search(inst["query"], N_BM25_SCAN)
        bm_legal = list(dict.fromkeys(
            base_pid(p) for p, _ in raw
            if base_pid(p) != src and _legal(base_pid(p), papers, as_of)))
        lists = {"bm25": bm_legal[:K_FINAL]}
        seeds = bm_legal[:N_SEEDS]
        for name, a in adjs.items():
            g, _s, _a = graph_rank_blocked(seeds, a, papers, as_of, src, blocked)
            lists[f"graph_{name}"] = g
            lists[f"hybrid_{name}"] = rrf([lists["bm25"], g], k=RRF_K, budget=K_FINAL)
        for arm, lst in lists.items():
            for p in lst:
                checked += 1
                d = fresh_date.get(p)
                if d is None:
                    viol_unknown.append((src, arm, p))
                elif normalize_ts(d) > normalize_ts(as_of):
                    viol_future.append((src, arm, p, d, as_of))
                if p == src:
                    viol_source.append((src, arm, p))
        # Control A: let the traversal pass THROUGH the source paper node, so the source's own
        # citation edges become live paths (never scoring the source itself).  This is the
        # "what would source-edge leakage be worth?" measurement.
        g_a, _s, _x = graph_rank_blocked(seeds, adjs["correct"], papers, as_of, src, set(),
                                         allow_source_as_waypoint=True)
        # Control B: faithful replay of the SUPERSEDED v1 masking -- literal-id blocked set on the
        # UN-MERGED adjacency, i.e. the exact bug the coordinator flagged.
        g_b, _s, _x = graph_rank_blocked(seeds, adj_raw_v1, papers, as_of, src,
                                         {f"paper:{inst['source_id']}"}
                                         | set(owned_v1.get(inst["source_id"], ())))
        ctrl.append(dict(source_id=src,
                         recall50_blocked=row["arms"]["graph_correct"]["recall@50"],
                         recall50_source_as_waypoint=recall_at_k(g_a, gold, 50),
                         recall50_v1_literal_id_mask=recall_at_k(g_b, gold, 50)))

    out = dict(
        n_sampled_instances=len(sample), n_returned_ids_rechecked=checked,
        independent_date_source="fresh `select min(submit_date) from papers group by 1` query",
        future_date_violations=len(viol_future), future_violation_examples=viol_future[:10],
        source_in_results_violations=len(viol_source),
        unknown_date_ids=len(viol_unknown), unknown_examples=viol_unknown[:10],
        gold_refs_not_covered_by_a_source_emitted_edge=gold_not_in_source_edges,
        source_edge_controls=dict(
            note=("graph_correct Recall@50 under the reported masking (T0+T3) versus two "
                  "deliberately broken variants. Control A lets traversal pass THROUGH the "
                  "source node so its own citation edges become live paths -- the value of pure "
                  "source-edge leakage. Control B is a faithful replay of the superseded v1 "
                  "masking (literal-id blocked set on the UN-MERGED adjacency), i.e. the exact "
                  "identity bug, so the gap measures how much the bug was actually worth on "
                  "these instances."),
            mean_recall50_reported=round(_mean([c["recall50_blocked"] for c in ctrl]), 4),
            control_A_source_as_waypoint=dict(
                mean_recall50=round(_mean([c["recall50_source_as_waypoint"] for c in ctrl]), 4),
                mean_leak=round(_mean([c["recall50_source_as_waypoint"] - c["recall50_blocked"]
                                       for c in ctrl]), 4),
                n_instances_changed=sum(1 for c in ctrl if abs(
                    c["recall50_source_as_waypoint"] - c["recall50_blocked"]) > 1e-12)),
            control_B_v1_literal_id_mask=dict(
                mean_recall50=round(_mean([c["recall50_v1_literal_id_mask"] for c in ctrl]), 4),
                mean_delta_vs_reported=round(
                    _mean([c["recall50_v1_literal_id_mask"] - c["recall50_blocked"]
                           for c in ctrl]), 4),
                n_instances_changed=sum(1 for c in ctrl if abs(
                    c["recall50_v1_literal_id_mask"] - c["recall50_blocked"]) > 1e-12)),
            n=len(ctrl)),
        verdict=("PASS: zero independent temporal/source-edge leakage violations"
                 if not viol_future and not viol_source
                 else "FAIL: leakage violations found"))
    write_json(out_dir / "independent_leakage_verification.json", out)
    print(json.dumps({k: v for k, v in out.items() if "examples" not in k}, indent=1))
    return out


ARM_LABEL = {
    "bm25": "BM25 (title+abstract query)",
    "graph_correct": "correct citation-neighbourhood expansion",
    "graph_rewired": "degree-preserving rewired expansion",
    "hybrid_correct": "correct BM25 + graph hybrid (RRF)",
    "hybrid_rewired": "BM25 + rewired hybrid (RRF)",
    "graph_correct_citation_only": "graph family: citation edges only (descriptive)",
    "graph_correct_paragraph_only": "graph family: paragraph edges only (descriptive)",
    "hybrid_correct_citation_only": "BM25 + citation-only graph hybrid (descriptive)",
}


def report(out_dir):
    out_dir = Path(out_dir)
    man = json.loads((out_dir / "construction_manifest.json").read_text())
    gate = json.loads((out_dir / "gate_decision.json").read_text())
    res = list(csv.DictReader(open(out_dir / "retrieval_results.csv")))
    comp = list(csv.DictReader(open(out_dir / "retrieval_comparisons.csv")))
    bm25eq = {}
    p = out_dir / "bm25_equivalence_check.json"
    if p.exists():
        bm25eq = json.loads(p.read_text())
    leak = gate["leakage_audit"]
    L = []
    A = L.append
    A("# Literature Set Expansion / Automated Snowballing — retrieval screening (runbook §4)\n")
    A(f"- run directory: `{out_dir}`")
    A(f"- commit: `{man['git_commit']}`")
    A(f"- built: {man['generated_at']}  ·  gated: {gate['decided_at']}")
    A(f"- frozen ID file: `frozen_ids.jsonl`, SHA-256 `{man['freeze']['frozen_ids_sha256']}`")
    A("- **no agents were run and no GPU/LLM was used at this stage.**\n")

    A("## 0. What the label actually is (read first)\n")
    A("> " + man["weak_supervision_statement"].replace("\n", " "))
    A("")
    A("Concretely, over the frozen set the mean **unresolved-reference rate** is "
      f"**{man['unresolved_reference_rates']['mean_over_frozen']:.3f}** "
      f"(mean over all {man['funnel']['sources_considered']} considered sources: "
      f"{man['unresolved_reference_rates']['mean_over_considered']:.3f}). That is, roughly "
      "two thirds of a source paper's bibliography entries never resolve to an in-corpus arXiv "
      "paper and therefore cannot appear in the label set at all. Absolute recall numbers below "
      "are consequently a floor on true topical recall, not a measurement of it. **Only the "
      "paired between-arm differences, which use the identical label set per instance, are "
      "interpretable as system comparisons.**\n")

    A("## 1. Task construction (§4.1)\n")
    A("For each source paper the system is shown **only** the title and the abstract, with LaTeX "
      "`\\cite*{...}` commands, `[CITATION]`/`[MASKED_CITATION]` markers and bracketed numeric "
      "citation runs stripped. It must return a ranked list of earlier papers. Introduction text "
      "was allowed by the runbook but **excluded by predeclared rule** (see "
      "`construction_manifest.json` → `predeclared_rules.query`): residual bib keys and "
      "author-year strings in extracted introductions are an unauditable citation-marker leakage "
      "channel at this scale.\n")
    f = man["funnel"]
    A("| construction funnel | n |")
    A("|---|---|")
    A(f"| source papers considered (have >= 1 resolvable bibliography entry) | {f['sources_considered']} |")
    A(f"| eligible after predeclared rules E1–E5 | {f['eligible']} |")
    A(f"| **frozen** (random sample, seed {man['freeze']['seed']}) | **{f['frozen']}** |")
    A(f"| — dev clusters | {f['dev']} |")
    A(f"| — screen clusters (primary) | {f['screen']} |")
    A("")
    A("Exclusion reasons (applied before any retrieval was run):\n")
    A("| reason | n |")
    A("|---|---|")
    for k, v in sorted(f["exclusion_histogram"].items(), key=lambda x: -x[1]):
        A(f"| `{k}` | {v} |")
    gs = man["gold_size"]
    A("")
    A(f"Gold set size per source: mean {gs['mean']}, median {gs['median']}, "
      f"range {gs['min']}–{gs['max']}.")
    cc = man["gold_cross_check"]
    A(f"Independent cross-check: for {cc['frozen_sources_whose_gold_equals_graph_cites_out']}"
      f"/{cc['n_frozen']} frozen sources the recomputed gold set is exactly the source's `cites` "
      "out-edge set in the citation graph; for the remaining sources the gold set is a strict "
      "**subset** of those out-edges (the extra out-edges are references dated after the cutoff). "
      "No gold reference is reachable through an edge the source did not emit, and every "
      "source-emitted edge is removed by rule T3 below.\n")

    A("## 2. Temporal and source-edge controls (§4.2)\n")
    gi = man.get("graph_identity")
    if gi:
        ex, fx = gi["raw_graph_exposure"], gi["frozen_set_exposure_under_a_literal_id_mask"]
        rf = man.get("refreeze", {})
        A("### 2.0 arXiv-id identity bug — found, quantified, fixed, and re-run\n")
        A("**This section supersedes a first pass whose numbers were discarded.** The shared "
          "citation graph mixes two spellings of the same paper: `cites` edges use "
          "version-stripped ids, but "
          f"{ex['edges_touching_a_versioned_paper_id'].get('has_paragraph', 0):,} of the "
          "`has_paragraph` edges carry an explicit version. "
          f"{ex['base_ids_present_under_both_spellings']:,} base papers therefore exist as two "
          f"unconnected nodes `paper:X` and `paper:Xv*` "
          f"({ex['versioned_paper_nodes']:,} versioned nodes among {ex['raw_paper_nodes']:,} "
          f"paper nodes / {ex['distinct_base_ids']:,} distinct base ids).\n")
        A("Exposure on **this** frozen set, had the source mask keyed on the literal id:\n")
        A("| measure | value |")
        A("|---|---|")
        A(f"| frozen sources owning paragraphs under a versioned id | "
          f"**{fx['sources_owning_paragraphs_under_a_versioned_id']} / {fx['n_frozen']} "
          f"({fx['fraction']*100:.1f}%)** |")
        A(f"| source-owned paragraph nodes that would have stayed live | "
          f"**{fx['source_owned_paragraph_nodes_left_unmasked']:,}** |")
        A(f"| direct `cites` source→gold edges affected | **0** (no `cites` edge carries a "
          "version suffix) |")
        A("")
        A("The live channel was the paragraph one: a traversal could reach a paper the source "
          "cites, step *back* into the source's own paragraph, and step *forward* to every other "
          "paper cited in that same paragraph — precisely the source-derived co-citation leakage "
          "runbook §4.2 forbids, and it was not uniform across instances (18.6% of them).\n")
        A(f"**Fix (rule T0):** {gi['rule']}. `paper:X` and `paper:Xv2` are merged into one node "
          "with merged edge sets; paragraph ownership is recorded against the base id; the "
          "blocked set is `{paper:base(S)}` ∪ {every paragraph owned by *any* spelling of S}; "
          "candidate ids are normalised to base ids. The merge is applied identically to the "
          "correct and the rewired variant, and the degree-preserving control survives it — see "
          "`graph_identity_audit.json`, which records element-wise identical per-relation in/out "
          "degree sequences for the two variants after merging. The existing `rewire` variant was "
          "reused, not rebuilt.\n")
        A(f"**Re-freeze:** {rf.get('reason', '')} Superseded artefacts (including the partial "
          "768-instance first-pass screening file) are preserved unmodified in "
          f"`{rf.get('superseded_artifacts_preserved_in', 'superseded_v1/')}` and were never used "
          "for any reported number. Because eligibility and sampling use only the `papers` table "
          "and `citation_resolution.csv` and never the graph, the rebuilt frozen set is "
          f"**byte-identical** to the first one (same SHA-256 "
          f"`{man['freeze']['frozen_ids_sha256_short']}…`, `frozen_ids_unchanged = "
          f"{rf.get('frozen_ids_unchanged')}`). No frozen id was changed on the basis of observed "
          "arm performance.\n")
    A("### 2.1 Control table\n")
    A("| control | implementation | audited result |")
    A("|---|---|---|")
    A("| paper identity resolved on the version-stripped base id (rule T0) | every adjacency "
      "rebuilt with `paper:X` / `paper:Xv*` merged, identically in every variant | see §2.0 |")
    A("| candidates predate the cutoff | `candidate_utility._legal` on every returned id, in "
      f"every arm | **{leak['future_candidates']} violations** |")
    A(f"| source paper never returned | explicit filter + post-hoc check | "
      f"**{leak['source_in_results']} violations** |")
    A("| all edges emitted by the source removed | blocked-node set = `{paper:S}` ∪ "
      "{paragraph nodes owned by S}; traversal may neither enter nor leave a blocked node, in "
      f"every graph variant | {leak['source_edges_blocked_total']:,} blocked traversal attempts "
      f"across {leak['n_instances']} instances |")
    A("| bibliography / citation markers / gold ids / source co-citations never revealed | input "
      "is title+abstract only, scrubbed; source paragraphs (which carry the bibliography and all "
      "source-derived co-citation structure) are blocked nodes | by construction |")
    A("| split by source paper | one instance = one source paper | by construction |")
    A("| split by venue/year cluster | grouped 20/80 dev/screen split on "
      f"`{{year}}-{{primary arXiv category}}` clusters ({man['n_clusters']} clusters, "
      f"{man['n_dev_clusters']} in dev) | no cluster spans both splits |")
    A("| unresolved-reference rates recorded | `temporal_audit.csv`, one row per considered "
      "source, with exclusion reasons | see §0 |")
    A("")
    A(f"**Leakage verdict: {'ZERO violations' if leak['zero_violations'] else 'VIOLATIONS PRESENT'}.**")
    A("")
    vp = out_dir / "independent_leakage_verification.json"
    if vp.exists():
        v = json.loads(vp.read_text())
        pc = v.get("source_edge_controls") or v.get("source_edge_positive_control")
        A("Independent re-audit (`independent_leakage_verification.json`): "
          f"{v['n_returned_ids_rechecked']:,} returned ids from {v['n_sampled_instances']} "
          "randomly sampled instances were re-checked against a freshly queried `submit_date` "
          f"table — {v['future_date_violations']} future-date violations, "
          f"{v['source_in_results_violations']} source-in-results violations, "
          f"{v['unknown_date_ids']} unknown-date ids. Verdict: **{v['verdict']}**.\n")
        ca, cb = pc.get("control_A_source_as_waypoint"), pc.get("control_B_v1_literal_id_mask")
        if ca and cb:
            A("Two deliberately-broken negative controls on the same sample quantify what the "
              "masking is actually worth (`graph_correct` Recall@50):\n")
            A("| variant | Recall@50 | Δ vs reported | instances changed |")
            A("|---|---|---|---|")
            A(f"| **reported masking (T0 + T3)** | **{pc['mean_recall50_reported']:.4f}** | — | — |")
            A(f"| control A: traversal allowed *through* the source node | "
              f"{ca['mean_recall50']:.4f} | {ca['mean_leak']:+.4f} | {ca['n_instances_changed']}"
              f"/{pc['n']} |")
            A(f"| control B: superseded v1 literal-id mask on the un-merged graph | "
              f"{cb['mean_recall50']:.4f} | {cb['mean_delta_vs_reported']:+.4f} | "
              f"{cb['n_instances_changed']}/{pc['n']} |")
            A("")
            A("Read this honestly in both directions. The identity bug was **real and structural** "
              "— 223/1200 sources had their own paragraphs left unmasked — but under this "
              "particular expansion procedure its measured effect on Recall@50 was small "
              f"({cb['mean_delta_vs_reported']:+.4f} on {cb['n_instances_changed']}/{pc['n']} "
              "sampled instances), because the inherited per-node cap (neighbours sorted by node "
              "id, `para:` sorting after `paper:`) means paragraph neighbours are seldom expanded "
              "from a paper node. The fix was still required: the exposure was non-uniform across "
              "instances, it is exactly the channel §4.2 forbids, and its size under a *different* "
              "traversal policy — such as the §4.4 agent, which can call a co-citation tool "
              "directly — is not bounded by what it happened to be worth here. **All reported "
              "numbers use the corrected masking.**\n")
        A(f"Gold references not covered by any source-emitted `cites` edge, on the sample: "
          f"{v['gold_refs_not_covered_by_a_source_emitted_edge']} — the gold set is a subset of "
          "the removed source edges, so no label survives in the graph as a direct edge.\n")
    A("Note on venue: the corpus has no venue field, so venue diversity is not computable; the "
      "arXiv primary category is used as the topical-diversity proxy and as the cluster key.\n")

    A("## 3. Retrieval-scale screening (§4.3)\n")
    A(f"Budget K = {K_FINAL} for every arm; **no arm is padded from another arm**. Seeds for "
      f"graph expansion are the top-{N_SEEDS} legal BM25 papers; expansion is the existing "
      f"label-free support-ranked BFS (`hybrid_retrieval.graph_rank`, max_hops={MAX_HOPS}, "
      f"per_node_cap={PER_NODE_CAP}); hybrids are reciprocal rank fusion (k={RRF_K}) of the BM25 "
      "list with the corresponding graph list. The correct and rewired arms differ **only** in "
      "the backing adjacency (same nodes, same text, same degrees, same procedure, same budget).\n")
    if bm25eq:
        A(f"BM25 equivalence check: the vectorised scorer reproduces the reference "
          f"`environment.BM25` top-50 exactly on {bm25eq['n_queries']}/{bm25eq['n_queries']} "
          f"probe queries (exact-order match rate {bm25eq['exact_order_match_rate']:.2f}).\n")
    A("**Semantic retrieval arm: NOT RUN.** No paper-level title+abstract vector index exists in "
      "the repository (the only embedding files, `data/citation_prediction/*embedding*.jsonl`, "
      "are paragraph-level over a different citation-prediction subset). This matches the "
      "existing deviation already recorded in `tasks/agentic_autoresearch/README.md` "
      "(\"No dense index exists → `one_shot_semantic` is deferred\"). The runbook forbids "
      "building a new index at this stage, so the arm is recorded as **not available** rather "
      "than silently omitted.\n")

    for subset, title in (("screen", "primary screening split"), ("dev", "dev split (descriptive)"),
                          ("frozen_all", "all frozen sources (descriptive)")):
        rows = [r for r in res if r["subset"] == subset]
        if not rows:
            continue
        n = rows[0]["n_instances"]
        A(f"### 3.{'123'[('screen', 'dev', 'frozen_all').index(subset)]} {title} (n={n})\n")
        A("| arm | R@20 | R@50 | nDCG@20 | nDCG@50 | MAP@50 | unique gold recovered | "
          "mean graph-only gold | dup rate | recall/candidate | distinct cats | cat entropy |")
        A("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in ALL_ARMS:
            r = next((x for x in rows if x["arm"] == arm), None)
            if not r:
                continue
            nm = ARM_LABEL[arm]
            bold = "**" if arm in ("hybrid_correct", "bm25", "hybrid_rewired") else ""
            A(f"| {bold}{nm}{bold} | {float(r['recall@20']):.4f} | {bold}{float(r['recall@50']):.4f}{bold} | "
              f"{float(r['ndcg@20']):.4f} | {float(r['ndcg@50']):.4f} | {float(r['map_at_50']):.4f} | "
              f"{r['unique_gold_references_recovered']} | {float(r['mean_graph_only_gold']):.2f} | "
              f"{float(r['duplicate_rate']):.4f} | {float(r['recall_per_candidate']):.5f} | "
              f"{float(r['distinct_categories']):.1f} | {float(r['category_entropy']):.2f} |")
        A("")
    A("`recall/candidate` is Recall@50 divided by the K=50 budget. `mean graph-only gold` counts "
      "gold references present in the arm's list but absent from the BM25 list. `dup rate` is the "
      "fraction of returned items whose normalised title repeats an earlier item. `distinct cats` "
      "and `cat entropy` are descriptive topical-diversity measures over the arXiv primary "
      "category of the 50 returned papers.\n")
    pm = [r for r in res if r["subset"] == "screen" and r["arm"] == "graph_correct_paragraph_only"]
    if pm:
        A(f"Caveat on the paragraph-only family arm: under the inherited per-node cap "
          f"(`PER_NODE_CAP={PER_NODE_CAP}`, neighbours sorted by node id) paper nodes almost never "
          "expose their paragraph neighbours, so this arm returns "
          f"{float(pm[0]['n_returned']):.1f} candidates on average. It is reported for "
          "completeness only and should not be read as evidence about paragraph connectivity.\n")

    A("## 4. Paired comparisons\n")
    A(f"Paired bootstrap over instances, {BOOT_N:,} resamples, fixed seed {STAT_SEED}; paired "
      f"two-sided permutation test, {PERM_N:,} sign-flips, same seed. Holm adjustment is applied "
      "**only** within the declared confirmatory family (the two Recall@50 gate comparisons on "
      "the screen split).\n")
    for subset in ("screen", "frozen_all"):
        rows = [r for r in comp if r["subset"] == subset and r["metric"] in ("recall@50", "recall@20", "ndcg@50")]
        if not rows:
            continue
        A(f"### {subset}\n")
        A("| family | comparison | metric | n | mean hi | mean lo | diff | 95% CI | p (2-sided) | p Holm | CI > 0 |")
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            ph = r.get("p_holm_confirmatory") or ""
            A(f"| {r['family']} | `{r['hi']}` > `{r['lo']}` | {r['metric']} | {r['n_paired']} | "
              f"{float(r['mean_hi']):.4f} | {float(r['mean_lo']):.4f} | **{float(r['diff']):+.4f}** | "
              f"[{float(r['ci_lo']):+.4f}, {float(r['ci_hi']):+.4f}] | {float(r['p_two_sided']):.4g} | "
              f"{ph} | {'yes' if r['positive_ci'] == 'True' else 'no'} |")
        A("")
    A("A non-significant difference is not evidence of equality. Directional hypotheses are "
      "judged on the sign of the observed difference together with the CI, never on the two-sided "
      "p alone.\n")

    A("## 5. Literature-agent launch gate (§4.3 / stopping rule 9.5)\n")
    A(f"> {gate['rule']}\n")
    prim = gate["by_subset"].get(gate["primary_subset"], {})
    c, ch = prim.get("criteria", {}), prim.get("checks", {})
    A("| gate criterion | required | observed | pass |")
    A("|---|---|---|---|")
    A(f"| correct hybrid − BM25, Recall@50 | ≥ +0.05 | "
      f"{c.get('hybrid_correct_minus_bm25_recall50', float('nan')):+.4f} | "
      f"{'YES' if ch.get('beats_bm25_by_0_05') else 'NO'} |")
    A(f"| 95% paired CI of that difference | strictly positive | "
      f"[{c.get('hybrid_correct_minus_bm25_ci', [0, 0])[0]:+.4f}, "
      f"{c.get('hybrid_correct_minus_bm25_ci', [0, 0])[1]:+.4f}] | "
      f"{'YES' if ch.get('positive_ci_vs_bm25') else 'NO'} |")
    A(f"| correct hybrid − rewired hybrid, Recall@50 | ≥ +0.05 | "
      f"{c.get('hybrid_correct_minus_rewired_recall50', float('nan')):+.4f} | "
      f"{'YES' if ch.get('beats_rewired_hybrid_by_0_05') else 'NO'} |")
    A(f"| 95% paired CI of that difference | strictly positive | "
      f"[{c.get('hybrid_correct_minus_rewired_ci', [0, 0])[0]:+.4f}, "
      f"{c.get('hybrid_correct_minus_rewired_ci', [0, 0])[1]:+.4f}] | "
      f"{'YES' if ch.get('positive_ci_vs_rewired') else 'NO'} |")
    A(f"| temporal / source-edge leakage | zero | future={leak['future_candidates']}, "
      f"source-in-results={leak['source_in_results']} | "
      f"{'YES' if ch.get('zero_leakage') else 'NO'} |")
    A("")
    A(f"**GATE DECISION ({gate['primary_subset']} split, n={c.get('n_paired')}): "
      f"{prim.get('decision', 'UNDETERMINED')}**")
    A("")
    A(f"Action: {gate['agent_action']}. `agents_launched = {gate['agents_launched']}` — "
      "the §4.4 snowballing-agent experiment is GPU work and is owned by another process; this "
      "stage only records the mechanical gate decision and the numbers behind it.\n")
    other = {k: v for k, v in gate["by_subset"].items() if k != gate["primary_subset"]}
    if other:
        A("Secondary (non-decisive) subsets, for transparency:\n")
        A("| subset | hybrid−BM25 R@50 | hybrid−rewired R@50 | decision |")
        A("|---|---|---|---|")
        for k, v in other.items():
            cc2 = v["criteria"]
            A(f"| {k} | {cc2['hybrid_correct_minus_bm25_recall50']:+.4f} | "
              f"{cc2['hybrid_correct_minus_rewired_recall50']:+.4f} | {v['decision']} |")
        A("")

    A("## 6. Reproduction\n")
    A("```bash")
    A("cd $PROJECT_ROOT")
    A("PY=python")
    A(f"OUT={out_dir}")
    A("$PY -m tasks.agentic_autoresearch.literature_set_expansion --stage build     --out-dir $OUT")
    A("$PY -m tasks.agentic_autoresearch.literature_set_expansion --stage screen    --out-dir $OUT")
    A("$PY -m tasks.agentic_autoresearch.literature_set_expansion --stage aggregate --out-dir $OUT")
    A("$PY -m tasks.agentic_autoresearch.literature_set_expansion --stage report    --out-dir $OUT")
    A("```")
    A("`--stage screen` is append-only and resumable: rows already present in "
      "`retrieval_per_instance.jsonl` are skipped, so re-running it resumes rather than "
      "recomputes. Re-running `--stage build` regenerates the identical frozen set (seed "
      f"{FREEZE_SEED}); the frozen IDs were hashed before any arm was compared and were not "
      "changed afterwards.\n")
    A("## 7. Files\n")
    for fn in ["construction_manifest.json", "temporal_audit.csv", "frozen_ids.jsonl",
               "frozen_ids_only.txt", "retrieval_per_instance.jsonl", "retrieval_results.csv",
               "retrieval_comparisons.csv", "gate_decision.json", "aggregate_summary.json",
               "bm25_equivalence_check.json", "graph_identity_audit.json",
               "independent_leakage_verification.json", "report.md",
               "superseded_v1/WHY_SUPERSEDED.md"]:
        pp = out_dir / fn
        A(f"- `{fn}` — {'present' if pp.exists() else 'MISSING'}"
          + (f", {pp.stat().st_size:,} bytes" if pp.exists() else ""))
    A("")
    ap = out_dir / "agent_results" / "agent_freeze_manifest.json"
    if ap.exists():
        am = json.loads(ap.read_text())
        A("## 8. §4.4 agent stage — evaluation set frozen, no inference run here\n")
        A("The gate passed, so §4.4 is cleared. The agent runner (`litexp_agent.py`) and its "
          "aggregator (`litexp_aggregate.py`) are built and dry-run verified offline; **no "
          "inference was started by this stage and no GPU or model server was contacted.**\n")
        A("| item | value |")
        A("|---|---|")
        A(f"| agent evaluation sources frozen | **{am['n']}** |")
        A(f"| drawn from | the **screen** split of `frozen_ids.jsonl` (the split the gate was "
          "decided on) |")
        A(f"| selection | {am['selection']} |")
        A(f"| agent ids SHA-256 | `{am['hashes']['agent_ids_sha256']}` |")
        A(f"| agent ids file SHA-256 | `{am['hashes']['agent_frozen_ids_file_sha256']}` |")
        A(f"| frozen before any inference | {am['frozen_before_any_inference']} |")
        A(f"| declared primary comparison | `{am['declared_primary_comparison']}` |")
        A(f"| smoke slice | {am['dev_slice']} |")
        A("")
        A("Confirmatory family (Holm-adjusted together, and only these):\n")
        for x in am["confirmatory_family"]:
            A(f"- `{x}`")
        A("")
        A("Everything else is reported `secondary_unadjusted`. Ids are scored independently of "
          "prose quality; rationale text never enters an effectiveness metric.\n")
    A("No agent episodes exist yet: `agent_results/` currently holds only the frozen ids and "
      "their hashes. Agent inference is GPU work owned by another process.")
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out_dir / 'report.md'}")


REPO_PROV = Path(__file__).resolve().parents[2] / "data" / "provenance"
GRAPH_DIR = (Path(__file__).resolve().parents[2] / "data" / "agentic_autoresearch" /
             "literature_evidence" / "graphs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["build", "screen", "aggregate", "report", "verify"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-sources", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.stage == "build":
        build(a.out_dir, a.limit_sources)
    elif a.stage == "screen":
        screen(a.out_dir, a.limit, a.force)
    elif a.stage == "report":
        report(a.out_dir)
    elif a.stage == "verify":
        verify(a.out_dir, a.limit or 40)
    else:
        res, comp, dec, leak = aggregate(a.out_dir)
        for r in res:
            if r["subset"] != "screen":
                continue
            print(f"  {r['arm']:32s} R@20={r['recall@20']:.4f} R@50={r['recall@50']:.4f} "
                  f"nDCG@50={r['ndcg@50']:.4f} MAP={r['map_at_50']:.4f}")
        print(json.dumps(dec["by_subset"], indent=1))
        print("GATE:", dec["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
