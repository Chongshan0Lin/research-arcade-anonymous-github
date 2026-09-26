"""Shared corpus / graph / BM25 infrastructure for Evidence Bundle Completion (EBC), runbook section 3.

CPU-only, no LLM, no GPU. Everything here is deterministic given EBC_SEED.

All retrieval and slice parameters in this file are PREREGISTERED: they are declared here, recorded
verbatim into construction_manifest.json, and frozen before any comparative arm is executed. They
are never re-tuned after observing a comparative result.

Design notes
------------
* Connectivity is used UNTYPED (runbook 0.2: "Use untyped connectivity unless a task explicitly
  studies relation semantics"). The adjacency is therefore built from citation_graph_<variant>.jsonl
  while ignoring the relation label, which is bit-identical to the `untyped` variant of the same
  topology.
* The "correct" graph is citation_graph_full.jsonl; the "rewired" control is the existing
  degree-preserving citation_graph_rewire.jsonl (per-relation double-edge swaps, in/out degree
  sequences preserved -- verified in graphs/perturbation_sanity.json). It is reused, not rebuilt.
* BM25 replicates tasks/agentic_autoresearch/environment.py:BM25 exactly (same tokenizer, same
  idf, same k1/b, query terms counted once) but is backed by a scipy CSR matrix so the screening
  can run over thousands of instances on CPU. `verify_bm25_equivalence` checks this numerically.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

from .common import DATA, db_conn

# ---------------------------------------------------------------- preregistered parameters
EBC_SEED = 20260925
TASK_NAME = "evidence_bundle_completion"

MIN_CITATIONS = 2                 # paragraph must resolve to >= 2 distinct arXiv targets
MAX_CITATIONS = 5                 # ... and <= 5 (runbook 3.1)
MIN_CONTEXT_TOKENS = 40           # same filter as the existing A1 builder

MAX_HOPS = 2                      # anchor-centred expansion depth
K_FINAL = 50                      # ranked list length returned by every arm
KS = (5, 10, 20)                  # reported recall cut-offs
RRF_K = 60                        # reciprocal-rank-fusion constant (repo default)
AA_ONE_HOP_BONUS = 2.0            # fixed bonus for direct anchor neighbours in the proximity score

TOOL_BUDGET_CALLS = 8             # declared agent budget: tool calls (runbook 1.2 / 3.4)
CANDIDATES_PER_CALL = 10          # declared agent budget: candidates returned per call
BUDGET_POOL = TOOL_BUDGET_CALLS * CANDIDATES_PER_CALL   # = 80 observable candidates

BM25_HARD_TOPK = 20               # GraphHard: "target outside BM25 top-20"
GRAPHHARD_MAX_HOPS = 2            # GraphHard: "reachable from the anchor within one or two hops"

N_BOOT = 10000                    # paired bootstrap resamples (runbook 8.3)
BOOT_SEED = 20260925

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
MAX_SCREENING_INSTANCES = 5000    # frozen screening-set cap (largest practical, >= 500 required)

TOKEN = re.compile(r"[a-z0-9]+")
CITE_CMD = re.compile(r"\\[a-zA-Z]*cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}")
ALPHA = re.compile(r"[A-Za-z]{2,}")
LATEX_CMD = re.compile(r"\\[a-zA-Z]+\s*")

EBC_DATA = DATA / "evidence_bundle_completion"
GRAPH_DIR = DATA / "literature_evidence" / "graphs"

PARAMS = dict(
    ebc_seed=EBC_SEED, min_citations=MIN_CITATIONS, max_citations=MAX_CITATIONS,
    min_context_tokens=MIN_CONTEXT_TOKENS, max_hops=MAX_HOPS, k_final=K_FINAL, ks=list(KS),
    rrf_k=RRF_K, adamic_adar_one_hop_bonus=AA_ONE_HOP_BONUS,
    tool_budget_calls=TOOL_BUDGET_CALLS, candidates_per_call=CANDIDATES_PER_CALL,
    budget_pool=BUDGET_POOL, bm25_hard_topk=BM25_HARD_TOPK,
    graphhard_max_hops=GRAPHHARD_MAX_HOPS, n_boot=N_BOOT, boot_seed=BOOT_SEED,
    split_fractions=list(SPLIT_FRACTIONS), max_screening_instances=MAX_SCREENING_INSTANCES,
    connectivity="untyped (relation labels ignored; identical to the `untyped` variant)",
    correct_graph="citation_graph_full.jsonl",
    rewired_graph="citation_graph_rewire.jsonl (degree-preserving, pre-existing, reused)",
    date_granularity="day (papers.submit_date is a DATE); availability test is date(x) <= date(t_s)",
    proximity_score="Adamic-Adar over <=2 hops from the anchor + fixed bonus for 1-hop neighbours",
)

INF_DATE = 99999999
VERSION_SUFFIX = re.compile(r"v\d+$")


def base_id(pid):
    """Version-stripped arXiv id.

    The citation graph mixes versioned and unversioned ids: `cites` edges are emitted for the
    version-stripped source, while ~20% of `has_paragraph` / `para_cites` edges carry an explicit
    version (e.g. paper:2002.08165v2). Source-paper masking therefore has to work on the
    version-stripped identity, otherwise a paper's own paragraphs survive under its versioned twin.
    """
    return VERSION_SUFFIX.sub("", pid)


# ---------------------------------------------------------------- text helpers

def normalize_all_citations(text):
    """Replace EVERY citation marker with the neutral token [CITATION].

    EBC reveals the anchor explicitly as structured evidence, so no bibliography key -- anchor or
    target -- is allowed to survive inside the paragraph. Returns (text, n_markers).
    """
    n = 0

    def sub(_m):
        nonlocal n
        n += 1
        return " [CITATION] "

    return CITE_CMD.sub(sub, text), n


def context_tokens(text):
    return len(ALPHA.findall(LATEX_CMD.sub(" ", text)))


def date_int(d):
    """'YYYY-MM-DD...' / datetime.date -> YYYYMMDD int; INF_DATE when unknown/unparseable."""
    if d is None:
        return INF_DATE
    s = str(d).strip()
    if len(s) < 10 or s[4] != "-" or s[7] != "-":
        return INF_DATE
    try:
        return int(s[:4]) * 10000 + int(s[5:7]) * 100 + int(s[8:10])
    except ValueError:
        return INF_DATE


# ---------------------------------------------------------------- BM25

class SparseBM25:
    """BM25 identical to environment.BM25 (k1=1.5, b=0.75, query terms counted once), CSR-backed."""

    def __init__(self, doc_ids, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.ids = list(doc_ids)
        self.pos = {d: i for i, d in enumerate(self.ids)}
        vocab = {}
        rows, cols, tfs, lens = [], [], [], []
        for i, t in enumerate(texts):
            c = {}
            for w in TOKEN.findall(str(t).lower()):
                c[w] = c.get(w, 0) + 1
            lens.append(sum(c.values()) or 1)
            for w, f in c.items():
                j = vocab.get(w)
                if j is None:
                    j = vocab[w] = len(vocab)
                rows.append(i)
                cols.append(j)
                tfs.append(f)
        self.vocab = vocab
        n = len(self.ids)
        lens = np.asarray(lens, dtype=np.float64)
        self.avg = float(lens.mean())
        rows = np.asarray(rows, dtype=np.int32)
        cols = np.asarray(cols, dtype=np.int32)
        tfs = np.asarray(tfs, dtype=np.float64)
        df = np.bincount(cols, minlength=len(vocab)).astype(np.float64)
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        L = lens[rows]
        w = idf[cols] * tfs * (k1 + 1.0) / (tfs + k1 * (1.0 - b + b * L / self.avg))
        self.M = sp.csr_matrix((w, (rows, cols)), shape=(n, len(vocab)))
        self.M.sort_indices()
        self.idf = idf

    def scores(self, query):
        """Dense score vector over all documents (order = self.ids)."""
        terms = {self.vocab[w] for w in set(TOKEN.findall(str(query).lower())) if w in self.vocab}
        if not terms:
            return np.zeros(self.M.shape[0], dtype=np.float64)
        q = np.zeros(self.M.shape[1], dtype=np.float64)
        q[np.fromiter(terms, dtype=np.int64, count=len(terms))] = 1.0
        return self.M.dot(q)


# ---------------------------------------------------------------- graph

class EBCGraph:
    """Undirected, untyped CSR adjacency over the shared node set of one topology variant."""

    def __init__(self, variant, node_index):
        self.variant = variant
        self.node_index = node_index
        src, dst = [], []
        with open(GRAPH_DIR / f"citation_graph_{variant}.jsonl") as f:
            for line in f:
                e = json.loads(line)
                a, b = node_index.get(e["s"]), node_index.get(e["t"])
                if a is None or b is None:
                    continue
                src.append(a)
                dst.append(b)
        a = np.asarray(src, dtype=np.int32)
        b = np.asarray(dst, dtype=np.int32)
        n = len(node_index)
        both_s = np.concatenate([a, b])
        both_t = np.concatenate([b, a])
        order = np.lexsort((both_t, both_s))
        both_s, both_t = both_s[order], both_t[order]
        self.indices = both_t
        self.indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(both_s, minlength=n), out=self.indptr[1:])
        self.degree = np.diff(self.indptr).astype(np.float64)
        self.n_edges = len(a)

    def nbrs(self, v):
        return self.indices[self.indptr[v]:self.indptr[v + 1]]


class EBCCorpus:
    """Papers + BM25 + the shared node index + one CSR adjacency per requested graph variant."""

    def __init__(self, variants=("full", "rewire")):
        import pandas as pd
        conn = db_conn()
        df = pd.read_sql(
            "select coalesce(base_arxiv_id, arxiv_id) as pid, max(title) as title, "
            "max(abstract) as abstract, min(submit_date::text) as date from papers group by 1", conn)
        self.paper_ids = list(df.pid)
        self.title = dict(zip(df.pid, df.title))
        self.abstract = dict(zip(df.pid, df.abstract))
        self.date = dict(zip(df.pid, df.date.astype(str)))
        self.date_i = {p: date_int(d) for p, d in self.date.items()}
        self.bm25 = SparseBM25(
            self.paper_ids,
            [f"{self.title.get(p) or ''} {self.abstract.get(p) or ''}" for p in self.paper_ids])

        # ---- shared node index (identical for every variant: node IDs are invariant) ----
        nodes = set()
        para_owner_node = {}
        with open(GRAPH_DIR / "citation_graph_full.jsonl") as f:
            for line in f:
                e = json.loads(line)
                nodes.add(e["s"])
                nodes.add(e["t"])
                if e["r"] == "has_paragraph":
                    para_owner_node[e["t"]] = e["s"]
        self.nodes = sorted(nodes)
        self.node_index = {n: i for i, n in enumerate(self.nodes)}
        n = len(self.nodes)

        # ---- per-node availability date, owning paper and version-stripped source group ----
        # group[v] = the version-stripped paper that EMITS v's edges (a paper owns itself and its
        # paragraphs, across arXiv versions). Blocking "every citation or paragraph edge created by
        # source paper s" is therefore exactly `group == group[s]`, which also blocks paper:s and
        # every para node of s under any version spelling.
        self.avail = np.full(n, INF_DATE, dtype=np.int64)
        self.owner = np.full(n, -1, dtype=np.int32)
        self.group = np.full(n, -1, dtype=np.int32)
        self.is_paper = np.zeros(n, dtype=bool)
        self.is_author = np.zeros(n, dtype=bool)
        self.paper_pos = np.full(n, -1, dtype=np.int32)        # node id -> BM25 row, -1 if absent
        self.base_group = {}

        def gid(base):
            g = self.base_group.get(base)
            if g is None:
                g = self.base_group[base] = len(self.base_group)
            return g

        for i, name in enumerate(self.nodes):
            kind, val = name.split(":", 1)
            if kind == "paper":
                b = base_id(val)
                self.is_paper[i] = True
                self.owner[i] = i
                self.group[i] = gid(b)
                # a versioned node (paper:XXXXv2) carries the availability date of its base paper
                self.avail[i] = self.date_i.get(val, self.date_i.get(b, INF_DATE))
                p = self.bm25.pos.get(val, self.bm25.pos.get(b))
                if p is not None:
                    self.paper_pos[i] = p
            elif kind == "author":
                self.is_author[i] = True
                self.avail[i] = 0                              # authors carry no timestamp
            # paragraphs are filled in below from their owning paper
        for para, owner in para_owner_node.items():
            pi, oi = self.node_index.get(para), self.node_index.get(owner)
            if pi is None or oi is None:
                continue
            self.owner[pi] = oi
            self.group[pi] = self.group[oi]
            self.avail[pi] = self.avail[oi]
        # paragraph nodes with no has_paragraph edge keep avail=INF -> never traversable
        # canonical paper node per base id: prefer the unversioned spelling
        self.paper_node_of = {}
        for i in np.flatnonzero(self.is_paper):
            val = self.nodes[i].split(":", 1)[1]
            self.paper_node_of[val] = i
            b = base_id(val)
            if b not in self.paper_node_of or b == val:
                self.paper_node_of[b] = i
        self.row_date = np.asarray([self.date_i.get(p, INF_DATE) for p in self.paper_ids],
                                   dtype=np.int64)
        self.row_group = np.asarray([self.base_group.get(base_id(p), -2) for p in self.paper_ids],
                                    dtype=np.int32)

        # group -> base id, and group -> its paper nodes (a base id can appear both as paper:X and
        # paper:Xv2; they carry complementary edges and must be treated as one paper)
        self.group_base = [None] * len(self.base_group)
        for b, gg in self.base_group.items():
            self.group_base[gg] = b
        self.n_groups = len(self.base_group)
        tmp = defaultdict(list)
        for i in np.flatnonzero(self.is_paper):
            tmp[int(self.group[i])].append(int(i))
        self.paper_nodes_of_group = {g: np.asarray(v, dtype=np.int32) for g, v in tmp.items()}

        # paper nodes sorted by availability date -> O(log n) "legal paper pool" per instance,
        # used only by the random-anchor / degree-matched-anchor negative controls
        pn = np.flatnonzero(self.is_paper & (self.avail < INF_DATE))
        order = np.lexsort((pn, self.avail[pn]))
        self.papers_by_date = pn[order]
        self._papers_by_date_key = self.avail[self.papers_by_date]

        self.graphs = {v: EBCGraph(v, self.node_index) for v in variants}

    def nodes_for_paper(self, paper_id):
        """Every node spelling of `paper_id` (unversioned + versioned twins), as an int32 array."""
        g = self.base_group.get(base_id(paper_id))
        if g is None:
            return np.empty(0, dtype=np.int32)
        return self.paper_nodes_of_group.get(g, np.empty(0, dtype=np.int32))

    def legal_paper_pool(self, as_of_i):
        """All paper nodes available by `as_of_i` (source-paper exclusion applied by the caller)."""
        j = int(np.searchsorted(self._papers_by_date_key, as_of_i, side="right"))
        return self.papers_by_date[:j]

    # -------------------------------------------------- per-instance legality masks
    def source_group(self, source_paper):
        return self.base_group.get(base_id(source_paper), -3)

    def legal_nodes(self, as_of_i, source_paper):
        """Boolean mask: node may be traversed / returned for this instance.

        A node is legal iff its availability date is <= the instance cutoff AND it is not emitted
        by the source paper. The second clause (`group != group[s]`) removes paper:s under every
        arXiv-version spelling and every paragraph node owned by s, i.e. every citation edge and
        every paragraph edge created by s, identically in both graph variants. It also guarantees
        that the evaluation paragraph itself can never create an anchor->target relation.
        """
        return (self.avail <= as_of_i) & (self.group != self.source_group(source_paper))

    def bm25_rank(self, query, as_of_i, source_paper, exclude=(), k=K_FINAL):
        """Top-k legal papers by BM25. Excludes the source paper, future papers and `exclude`."""
        sc = self.bm25.scores(query)
        ok = (self.row_date <= as_of_i) & (sc > 0) & (self.row_group != self.source_group(source_paper))
        for x in exclude:
            for key in (x, base_id(x)):
                i = self.bm25.pos.get(key)
                if i is not None:
                    ok[i] = False
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            return []
        if idx.size > k:
            part = idx[np.argpartition(-sc[idx], k)[:k]]
        else:
            part = idx
        part = part[np.lexsort((np.asarray([self.paper_ids[i] for i in part]), -sc[part]))]
        return [self.paper_ids[i] for i in part[:k]]


# ---------------------------------------------------------------- ranking primitives

def rrf(lists, k=RRF_K, budget=K_FINAL):
    score = defaultdict(float)
    for lst in lists:
        for i, pid in enumerate(lst, 1):
            score[pid] += 1.0 / (k + i)
    return sorted(score, key=lambda p: (-score[p], p))[:budget]


def verify_bm25_equivalence(corpus, queries, n_check=5):
    """Numeric check that SparseBM25 reproduces environment.BM25 scores."""
    from .environment import BM25
    sub = corpus.paper_ids[:4000]
    ref = BM25({p: f"{corpus.title.get(p) or ''} {corpus.abstract.get(p) or ''}" for p in sub})
    small = SparseBM25(sub, [f"{corpus.title.get(p) or ''} {corpus.abstract.get(p) or ''}"
                             for p in sub])
    worst = 0.0
    for q in queries[:n_check]:
        a = dict(ref.search(q, 20))
        s = small.scores(q)
        for pid, v in a.items():
            worst = max(worst, abs(v - s[small.pos[pid]]))
    return worst
