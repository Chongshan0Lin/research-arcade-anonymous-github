# Literature Set Expansion / Automated Snowballing — retrieval screening (runbook §4)

- run directory: `results/graph_autoresearch/graphauto_20260925_035919_027d9a/literature_set_expansion`
- commit: `2631ae6d320009d828da4d451556241f7359fde2`
- built: 2026-09-25T09:45:38+00:00  ·  gated: 2026-09-25T10:33:17+00:00
- frozen ID file: `frozen_ids.jsonl`, SHA-256 `5b7eadacd40f892a0f918cef4510649059f01ee11d2c8fa93ca4384a509e4553`
- **no agents were run and no GPU/LLM was used at this stage.**

## 0. What the label actually is (read first)

> Labels are the source paper's own resolved earlier references. An author's bibliography is INCOMPLETE and SELECTIVE; this is a reproducible WEAK relevance label, NOT exhaustive ground truth. Absolute recall understates true topical recall. Only paired between-arm differences on identical label sets are interpretable as system comparisons.

Concretely, over the frozen set the mean **unresolved-reference rate** is **0.680** (mean over all 41054 considered sources: 0.754). That is, roughly two thirds of a source paper's bibliography entries never resolve to an in-corpus arXiv paper and therefore cannot appear in the label set at all. Absolute recall numbers below are consequently a floor on true topical recall, not a measurement of it. **Only the paired between-arm differences, which use the identical label set per instance, are interpretable as system comparisons.**

## 1. Task construction (§4.1)

For each source paper the system is shown **only** the title and the abstract, with LaTeX `\cite*{...}` commands, `[CITATION]`/`[MASKED_CITATION]` markers and bracketed numeric citation runs stripped. It must return a ranked list of earlier papers. Introduction text was allowed by the runbook but **excluded by predeclared rule** (see `construction_manifest.json` → `predeclared_rules.query`): residual bib keys and author-year strings in extracted introductions are an unauditable citation-marker leakage channel at this scale.

| construction funnel | n |
|---|---|
| source papers considered (have >= 1 resolvable bibliography entry) | 41054 |
| eligible after predeclared rules E1–E5 | 27267 |
| **frozen** (random sample, seed 20260925) | **1200** |
| — dev clusters | 291 |
| — screen clusters (primary) | 909 |

Exclusion reasons (applied before any retrieval was run):

| reason | n |
|---|---|
| `E2_too_few_gold|E4_unresolved_rate` | 8887 |
| `E2_too_few_gold` | 3712 |
| `E4_unresolved_rate` | 1181 |
| `E1_abstract_too_short|E2_too_few_gold|E4_unresolved_rate` | 3 |
| `E1_abstract_too_short|E2_too_few_gold` | 2 |
| `E3_too_many_gold` | 1 |
| `E1_abstract_too_short` | 1 |

Gold set size per source: mean 16.57, median 14, range 5–88.
Independent cross-check: for 1108/1200 frozen sources the recomputed gold set is exactly the source's `cites` out-edge set in the citation graph; for the remaining sources the gold set is a strict **subset** of those out-edges (the extra out-edges are references dated after the cutoff). No gold reference is reachable through an edge the source did not emit, and every source-emitted edge is removed by rule T3 below.

## 2. Temporal and source-edge controls (§4.2)

### 2.0 arXiv-id identity bug — found, quantified, fixed, and re-run

**This section supersedes a first pass whose numbers were discarded.** The shared citation graph mixes two spellings of the same paper: `cites` edges use version-stripped ids, but 73,178 of the `has_paragraph` edges carry an explicit version. 6,198 base papers therefore exist as two unconnected nodes `paper:X` and `paper:Xv*` (6,211 versioned nodes among 52,307 paper nodes / 46,096 distinct base ids).

Exposure on **this** frozen set, had the source mask keyed on the literal id:

| measure | value |
|---|---|
| frozen sources owning paragraphs under a versioned id | **223 / 1200 (18.6%)** |
| source-owned paragraph nodes that would have stayed live | **3,070** |
| direct `cites` source→gold edges affected | **0** (no `cites` edge carries a version suffix) |

The live channel was the paragraph one: a traversal could reach a paper the source cites, step *back* into the source's own paragraph, and step *forward* to every other paper cited in that same paragraph — precisely the source-derived co-citation leakage runbook §4.2 forbids, and it was not uniform across instances (18.6% of them).

**Fix (rule T0):** T0: paper node identity is resolved on the version-stripped base id at adjacency-build time, identically in every graph variant. `paper:X` and `paper:Xv2` are merged into one node with merged edge sets; paragraph ownership is recorded against the base id; the blocked set is `{paper:base(S)}` ∪ {every paragraph owned by *any* spelling of S}; candidate ids are normalised to base ids. The merge is applied identically to the correct and the rewired variant, and the degree-preserving control survives it — see `graph_identity_audit.json`, which records element-wise identical per-relation in/out degree sequences for the two variants after merging. The existing `rewire` variant was reused, not rebuilt.

**Re-freeze:** the shared citation graph mixed arXiv id spellings, defeating literal-id source-edge masking; construction was rebuilt from scratch after fixing identity resolution (rule T0). The re-freeze was NOT performed on the basis of any observed arm performance. Superseded artefacts (including the partial 768-instance first-pass screening file) are preserved unmodified in `superseded_v1/` and were never used for any reported number. Because eligibility and sampling use only the `papers` table and `citation_resolution.csv` and never the graph, the rebuilt frozen set is **byte-identical** to the first one (same SHA-256 `5b7eadacd40f892a…`, `frozen_ids_unchanged = True`). No frozen id was changed on the basis of observed arm performance.

### 2.1 Control table

| control | implementation | audited result |
|---|---|---|
| paper identity resolved on the version-stripped base id (rule T0) | every adjacency rebuilt with `paper:X` / `paper:Xv*` merged, identically in every variant | see §2.0 |
| candidates predate the cutoff | `candidate_utility._legal` on every returned id, in every arm | **0 violations** |
| source paper never returned | explicit filter + post-hoc check | **0 violations** |
| all edges emitted by the source removed | blocked-node set = `{paper:S}` ∪ {paragraph nodes owned by S}; traversal may neither enter nor leave a blocked node, in every graph variant | 44,346 blocked traversal attempts across 1200 instances |
| bibliography / citation markers / gold ids / source co-citations never revealed | input is title+abstract only, scrubbed; source paragraphs (which carry the bibliography and all source-derived co-citation structure) are blocked nodes | by construction |
| split by source paper | one instance = one source paper | by construction |
| split by venue/year cluster | grouped 20/80 dev/screen split on `{year}-{primary arXiv category}` clusters (116 clusters, 13 in dev) | no cluster spans both splits |
| unresolved-reference rates recorded | `temporal_audit.csv`, one row per considered source, with exclusion reasons | see §0 |

**Leakage verdict: ZERO violations.**

Independent re-audit (`independent_leakage_verification.json`): 15,000 returned ids from 60 randomly sampled instances were re-checked against a freshly queried `submit_date` table — 0 future-date violations, 0 source-in-results violations, 0 unknown-date ids. Verdict: **PASS: zero independent temporal/source-edge leakage violations**.

Two deliberately-broken negative controls on the same sample quantify what the masking is actually worth (`graph_correct` Recall@50):

| variant | Recall@50 | Δ vs reported | instances changed |
|---|---|---|---|
| **reported masking (T0 + T3)** | **0.3944** | — | — |
| control A: traversal allowed *through* the source node | 0.3984 | +0.0040 | 4/60 |
| control B: superseded v1 literal-id mask on the un-merged graph | 0.3911 | -0.0032 | 3/60 |

Read this honestly in both directions. The identity bug was **real and structural** — 223/1200 sources had their own paragraphs left unmasked — but under this particular expansion procedure its measured effect on Recall@50 was small (-0.0032 on 3/60 sampled instances), because the inherited per-node cap (neighbours sorted by node id, `para:` sorting after `paper:`) means paragraph neighbours are seldom expanded from a paper node. The fix was still required: the exposure was non-uniform across instances, it is exactly the channel §4.2 forbids, and its size under a *different* traversal policy — such as the §4.4 agent, which can call a co-citation tool directly — is not bounded by what it happened to be worth here. **All reported numbers use the corrected masking.**

Gold references not covered by any source-emitted `cites` edge, on the sample: 0 — the gold set is a subset of the removed source edges, so no label survives in the graph as a direct edge.

Note on venue: the corpus has no venue field, so venue diversity is not computable; the arXiv primary category is used as the topical-diversity proxy and as the cluster key.

## 3. Retrieval-scale screening (§4.3)

Budget K = 50 for every arm; **no arm is padded from another arm**. Seeds for graph expansion are the top-10 legal BM25 papers; expansion is the existing label-free support-ranked BFS (`hybrid_retrieval.graph_rank`, max_hops=3, per_node_cap=25); hybrids are reciprocal rank fusion (k=60) of the BM25 list with the corresponding graph list. The correct and rewired arms differ **only** in the backing adjacency (same nodes, same text, same degrees, same procedure, same budget).

BM25 equivalence check: the vectorised scorer reproduces the reference `environment.BM25` top-50 exactly on 5/5 probe queries (exact-order match rate 1.00).

**Semantic retrieval arm: NOT RUN.** No paper-level title+abstract vector index exists in the repository (the only embedding files, `data/citation_prediction/*embedding*.jsonl`, are paragraph-level over a different citation-prediction subset). This matches the existing deviation already recorded in `tasks/agentic_autoresearch/README.md` ("No dense index exists → `one_shot_semantic` is deferred"). The runbook forbids building a new index at this stage, so the arm is recorded as **not available** rather than silently omitted.

### 3.1 primary screening split (n=909)

| arm | R@20 | R@50 | nDCG@20 | nDCG@50 | MAP@50 | unique gold recovered | mean graph-only gold | dup rate | recall/candidate | distinct cats | cat entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **BM25 (title+abstract query)** | 0.1288 | **0.1913** | 0.1553 | 0.1772 | 0.0674 | 1902 | 0.00 | 0.0000 | 0.00383 | 7.5 | 1.87 |
| correct citation-neighbourhood expansion | 0.2833 | 0.4006 | 0.3265 | 0.3663 | 0.1790 | 1549 | 5.06 | 0.0000 | 0.00801 | 5.9 | 1.70 |
| degree-preserving rewired expansion | 0.0959 | 0.1529 | 0.1029 | 0.1240 | 0.0379 | 185 | 2.08 | 0.0000 | 0.00306 | 6.5 | 2.07 |
| **correct BM25 + graph hybrid (RRF)** | 0.2876 | **0.4266** | 0.3103 | 0.3618 | 0.1690 | 2193 | 3.91 | 0.0000 | 0.00853 | 6.8 | 1.79 |
| **BM25 + rewired hybrid (RRF)** | 0.1579 | **0.2516** | 0.1682 | 0.2033 | 0.0726 | 1532 | 1.49 | 0.0000 | 0.00503 | 7.4 | 2.13 |
| graph family: citation edges only (descriptive) | 0.2834 | 0.4005 | 0.3266 | 0.3662 | 0.1789 | 1547 | 0.00 | 0.0000 | 0.00801 | 5.9 | 1.70 |
| graph family: paragraph edges only (descriptive) | 0.1518 | 0.1798 | 0.1695 | 0.1757 | 0.0786 | 1091 | 0.00 | 0.0000 | 0.00360 | 3.9 | 1.31 |
| BM25 + citation-only graph hybrid (descriptive) | 0.2876 | 0.4265 | 0.3103 | 0.3617 | 0.1689 | 2192 | 0.00 | 0.0000 | 0.00853 | 6.8 | 1.79 |

### 3.2 dev split (descriptive) (n=291)

| arm | R@20 | R@50 | nDCG@20 | nDCG@50 | MAP@50 | unique gold recovered | mean graph-only gold | dup rate | recall/candidate | distinct cats | cat entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **BM25 (title+abstract query)** | 0.0949 | **0.1546** | 0.1299 | 0.1492 | 0.0524 | 636 | 0.00 | 0.0000 | 0.00309 | 6.0 | 1.38 |
| correct citation-neighbourhood expansion | 0.2836 | 0.3975 | 0.3472 | 0.3783 | 0.1876 | 731 | 5.80 | 0.0000 | 0.00795 | 5.2 | 1.53 |
| degree-preserving rewired expansion | 0.0923 | 0.1418 | 0.0981 | 0.1159 | 0.0317 | 89 | 2.17 | 0.0000 | 0.00284 | 6.5 | 2.07 |
| **correct BM25 + graph hybrid (RRF)** | 0.2747 | **0.4027** | 0.3144 | 0.3550 | 0.1613 | 924 | 4.51 | 0.0000 | 0.00805 | 5.6 | 1.44 |
| **BM25 + rewired hybrid (RRF)** | 0.1290 | **0.2110** | 0.1472 | 0.1782 | 0.0569 | 513 | 1.54 | 0.0000 | 0.00422 | 6.7 | 1.90 |
| graph family: citation edges only (descriptive) | 0.2838 | 0.3975 | 0.3474 | 0.3784 | 0.1877 | 731 | 0.00 | 0.0000 | 0.00795 | 5.2 | 1.53 |
| graph family: paragraph edges only (descriptive) | 0.1292 | 0.1476 | 0.1518 | 0.1536 | 0.0656 | 412 | 0.00 | 0.0000 | 0.00295 | 3.0 | 0.97 |
| BM25 + citation-only graph hybrid (descriptive) | 0.2747 | 0.4027 | 0.3145 | 0.3550 | 0.1613 | 924 | 0.00 | 0.0000 | 0.00805 | 5.6 | 1.44 |

### 3.3 all frozen sources (descriptive) (n=1200)

| arm | R@20 | R@50 | nDCG@20 | nDCG@50 | MAP@50 | unique gold recovered | mean graph-only gold | dup rate | recall/candidate | distinct cats | cat entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **BM25 (title+abstract query)** | 0.1205 | **0.1824** | 0.1492 | 0.1704 | 0.0638 | 2364 | 0.00 | 0.0000 | 0.00365 | 7.2 | 1.75 |
| correct citation-neighbourhood expansion | 0.2834 | 0.3999 | 0.3315 | 0.3692 | 0.1811 | 1851 | 5.24 | 0.0000 | 0.00800 | 5.8 | 1.66 |
| degree-preserving rewired expansion | 0.0950 | 0.1502 | 0.1017 | 0.1221 | 0.0364 | 199 | 2.10 | 0.0000 | 0.00300 | 6.5 | 2.07 |
| **correct BM25 + graph hybrid (RRF)** | 0.2845 | **0.4208** | 0.3113 | 0.3601 | 0.1671 | 2678 | 4.05 | 0.0000 | 0.00842 | 6.5 | 1.70 |
| **BM25 + rewired hybrid (RRF)** | 0.1509 | **0.2417** | 0.1631 | 0.1972 | 0.0688 | 1896 | 1.50 | 0.0000 | 0.00483 | 7.2 | 2.07 |
| graph family: citation edges only (descriptive) | 0.2835 | 0.3998 | 0.3316 | 0.3692 | 0.1811 | 1850 | 0.00 | 0.0000 | 0.00800 | 5.8 | 1.66 |
| graph family: paragraph edges only (descriptive) | 0.1463 | 0.1720 | 0.1652 | 0.1704 | 0.0755 | 1331 | 0.00 | 0.0000 | 0.00344 | 3.7 | 1.23 |
| BM25 + citation-only graph hybrid (descriptive) | 0.2845 | 0.4208 | 0.3113 | 0.3601 | 0.1671 | 2678 | 0.00 | 0.0000 | 0.00842 | 6.5 | 1.70 |

`recall/candidate` is Recall@50 divided by the K=50 budget. `mean graph-only gold` counts gold references present in the arm's list but absent from the BM25 list. `dup rate` is the fraction of returned items whose normalised title repeats an earlier item. `distinct cats` and `cat entropy` are descriptive topical-diversity measures over the arXiv primary category of the 50 returned papers.

Caveat on the paragraph-only family arm: under the inherited per-node cap (`PER_NODE_CAP=25`, neighbours sorted by node id) paper nodes almost never expose their paragraph neighbours, so this arm returns 20.0 candidates on average. It is reported for completeness only and should not be read as evidence about paragraph connectivity.

## 4. Paired comparisons

Paired bootstrap over instances, 10,000 resamples, fixed seed 20260925; paired two-sided permutation test, 10,000 sign-flips, same seed. Holm adjustment is applied **only** within the declared confirmatory family (the two Recall@50 gate comparisons on the screen split).

### screen

| family | comparison | metric | n | mean hi | mean lo | diff | 95% CI | p (2-sided) | p Holm | CI > 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| confirmatory | `hybrid_correct` > `bm25` | recall@50 | 909 | 0.4266 | 0.1913 | **+0.2353** | [+0.2245, +0.2460] | 9.999e-05 | 0.00019998000199980003 | yes |
| confirmatory | `hybrid_correct` > `bm25` | recall@20 | 909 | 0.2876 | 0.1288 | **+0.1589** | [+0.1499, +0.1678] | 9.999e-05 |  | yes |
| confirmatory | `hybrid_correct` > `bm25` | ndcg@50 | 909 | 0.3618 | 0.1772 | **+0.1846** | [+0.1749, +0.1943] | 9.999e-05 |  | yes |
| confirmatory | `hybrid_correct` > `hybrid_rewired` | recall@50 | 909 | 0.4266 | 0.2516 | **+0.1750** | [+0.1644, +0.1858] | 9.999e-05 | 0.00019998000199980003 | yes |
| confirmatory | `hybrid_correct` > `hybrid_rewired` | recall@20 | 909 | 0.2876 | 0.1579 | **+0.1298** | [+0.1207, +0.1392] | 9.999e-05 |  | yes |
| confirmatory | `hybrid_correct` > `hybrid_rewired` | ndcg@50 | 909 | 0.3618 | 0.2033 | **+0.1585** | [+0.1487, +0.1686] | 9.999e-05 |  | yes |
| exploratory | `hybrid_correct` > `graph_correct` | recall@50 | 909 | 0.4266 | 0.4006 | **+0.0259** | [+0.0180, +0.0337] | 9.999e-05 |  | yes |
| exploratory | `hybrid_correct` > `graph_correct` | recall@20 | 909 | 0.2876 | 0.2833 | **+0.0043** | [-0.0037, +0.0122] | 0.2802 |  | no |
| exploratory | `hybrid_correct` > `graph_correct` | ndcg@50 | 909 | 0.3618 | 0.3663 | **-0.0046** | [-0.0113, +0.0024] | 0.1942 |  | no |
| exploratory | `graph_correct` > `graph_rewired` | recall@50 | 909 | 0.4006 | 0.1529 | **+0.2477** | [+0.2343, +0.2615] | 9.999e-05 |  | yes |
| exploratory | `graph_correct` > `graph_rewired` | recall@20 | 909 | 0.2833 | 0.0959 | **+0.1875** | [+0.1764, +0.1989] | 9.999e-05 |  | yes |
| exploratory | `graph_correct` > `graph_rewired` | ndcg@50 | 909 | 0.3663 | 0.1240 | **+0.2423** | [+0.2300, +0.2547] | 9.999e-05 |  | yes |
| exploratory | `graph_correct` > `bm25` | recall@50 | 909 | 0.4006 | 0.1913 | **+0.2093** | [+0.1959, +0.2226] | 9.999e-05 |  | yes |
| exploratory | `graph_correct` > `bm25` | recall@20 | 909 | 0.2833 | 0.1288 | **+0.1546** | [+0.1427, +0.1667] | 9.999e-05 |  | yes |
| exploratory | `graph_correct` > `bm25` | ndcg@50 | 909 | 0.3663 | 0.1772 | **+0.1892** | [+0.1760, +0.2022] | 9.999e-05 |  | yes |
| exploratory | `hybrid_rewired` > `bm25` | recall@50 | 909 | 0.2516 | 0.1913 | **+0.0602** | [+0.0519, +0.0687] | 9.999e-05 |  | yes |
| exploratory | `hybrid_rewired` > `bm25` | recall@20 | 909 | 0.1579 | 0.1288 | **+0.0291** | [+0.0224, +0.0357] | 9.999e-05 |  | yes |
| exploratory | `hybrid_rewired` > `bm25` | ndcg@50 | 909 | 0.2033 | 0.1772 | **+0.0262** | [+0.0188, +0.0333] | 9.999e-05 |  | yes |
| exploratory | `hybrid_correct` > `hybrid_correct_citation_only` | recall@50 | 909 | 0.4266 | 0.4265 | **+0.0000** | [-0.0001, +0.0002] | 1 |  | no |
| exploratory | `hybrid_correct` > `hybrid_correct_citation_only` | recall@20 | 909 | 0.2876 | 0.2876 | **+0.0001** | [+0.0000, +0.0002] | 1 |  | no |
| exploratory | `hybrid_correct` > `hybrid_correct_citation_only` | ndcg@50 | 909 | 0.3618 | 0.3617 | **+0.0001** | [-0.0004, +0.0005] | 0.684 |  | no |

### frozen_all

| family | comparison | metric | n | mean hi | mean lo | diff | 95% CI | p (2-sided) | p Holm | CI > 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| confirmatory | `hybrid_correct` > `bm25` | recall@50 | 1200 | 0.4208 | 0.1824 | **+0.2384** | [+0.2290, +0.2477] | 9.999e-05 |  | yes |
| confirmatory | `hybrid_correct` > `bm25` | recall@20 | 1200 | 0.2845 | 0.1205 | **+0.1640** | [+0.1562, +0.1720] | 9.999e-05 |  | yes |
| confirmatory | `hybrid_correct` > `bm25` | ndcg@50 | 1200 | 0.3601 | 0.1704 | **+0.1898** | [+0.1814, +0.1980] | 9.999e-05 |  | yes |
| confirmatory | `hybrid_correct` > `hybrid_rewired` | recall@50 | 1200 | 0.4208 | 0.2417 | **+0.1790** | [+0.1697, +0.1882] | 9.999e-05 |  | yes |
| confirmatory | `hybrid_correct` > `hybrid_rewired` | recall@20 | 1200 | 0.2845 | 0.1509 | **+0.1336** | [+0.1257, +0.1418] | 9.999e-05 |  | yes |
| confirmatory | `hybrid_correct` > `hybrid_rewired` | ndcg@50 | 1200 | 0.3601 | 0.1972 | **+0.1629** | [+0.1541, +0.1713] | 9.999e-05 |  | yes |
| exploratory | `hybrid_correct` > `graph_correct` | recall@50 | 1200 | 0.4208 | 0.3999 | **+0.0209** | [+0.0141, +0.0276] | 9.999e-05 |  | yes |
| exploratory | `hybrid_correct` > `graph_correct` | recall@20 | 1200 | 0.2845 | 0.2834 | **+0.0011** | [-0.0057, +0.0078] | 0.7386 |  | no |
| exploratory | `hybrid_correct` > `graph_correct` | ndcg@50 | 1200 | 0.3601 | 0.3692 | **-0.0091** | [-0.0151, -0.0030] | 0.0042 |  | no |
| exploratory | `graph_correct` > `graph_rewired` | recall@50 | 1200 | 0.3999 | 0.1502 | **+0.2497** | [+0.2379, +0.2611] | 9.999e-05 |  | yes |
| exploratory | `graph_correct` > `graph_rewired` | recall@20 | 1200 | 0.2834 | 0.0950 | **+0.1884** | [+0.1786, +0.1979] | 9.999e-05 |  | yes |
| exploratory | `graph_correct` > `graph_rewired` | ndcg@50 | 1200 | 0.3692 | 0.1221 | **+0.2472** | [+0.2363, +0.2578] | 9.999e-05 |  | yes |
| exploratory | `graph_correct` > `bm25` | recall@50 | 1200 | 0.3999 | 0.1824 | **+0.2175** | [+0.2058, +0.2292] | 9.999e-05 |  | yes |
| exploratory | `graph_correct` > `bm25` | recall@20 | 1200 | 0.2834 | 0.1205 | **+0.1628** | [+0.1527, +0.1735] | 9.999e-05 |  | yes |
| exploratory | `graph_correct` > `bm25` | ndcg@50 | 1200 | 0.3692 | 0.1704 | **+0.1989** | [+0.1874, +0.2105] | 9.999e-05 |  | yes |
| exploratory | `hybrid_rewired` > `bm25` | recall@50 | 1200 | 0.2417 | 0.1824 | **+0.0593** | [+0.0521, +0.0665] | 9.999e-05 |  | yes |
| exploratory | `hybrid_rewired` > `bm25` | recall@20 | 1200 | 0.1509 | 0.1205 | **+0.0303** | [+0.0248, +0.0362] | 9.999e-05 |  | yes |
| exploratory | `hybrid_rewired` > `bm25` | ndcg@50 | 1200 | 0.1972 | 0.1704 | **+0.0269** | [+0.0208, +0.0331] | 9.999e-05 |  | yes |
| exploratory | `hybrid_correct` > `hybrid_correct_citation_only` | recall@50 | 1200 | 0.4208 | 0.4208 | **+0.0000** | [-0.0001, +0.0001] | 1 |  | no |
| exploratory | `hybrid_correct` > `hybrid_correct_citation_only` | recall@20 | 1200 | 0.2845 | 0.2845 | **+0.0000** | [+0.0000, +0.0001] | 1 |  | no |
| exploratory | `hybrid_correct` > `hybrid_correct_citation_only` | ndcg@50 | 1200 | 0.3601 | 0.3601 | **+0.0000** | [-0.0003, +0.0004] | 0.7324 |  | no |

A non-significant difference is not evidence of equality. Directional hypotheses are judged on the sign of the observed difference together with the CI, never on the two-sided p alone.

## 5. Literature-agent launch gate (§4.3 / stopping rule 9.5)

> launch literature agents only if correct hybrid exceeds BM25 by >= 0.05 absolute Recall@50 AND exceeds the rewired hybrid by >= 0.05 absolute Recall@50, with positive paired 95% CIs for BOTH differences, and zero temporal/source-edge leakage (runbook 4.3 / stopping rule 9.5)

| gate criterion | required | observed | pass |
|---|---|---|---|
| correct hybrid − BM25, Recall@50 | ≥ +0.05 | +0.2353 | YES |
| 95% paired CI of that difference | strictly positive | [+0.2245, +0.2460] | YES |
| correct hybrid − rewired hybrid, Recall@50 | ≥ +0.05 | +0.1750 | YES |
| 95% paired CI of that difference | strictly positive | [+0.1644, +0.1858] | YES |
| temporal / source-edge leakage | zero | future=0, source-in-results=0 | YES |

**GATE DECISION (screen split, n=909): PASS**

Action: PROCEED to 4.4 (GPU stage, owned by another process). `agents_launched = False` — the §4.4 snowballing-agent experiment is GPU work and is owned by another process; this stage only records the mechanical gate decision and the numbers behind it.

Secondary (non-decisive) subsets, for transparency:

| subset | hybrid−BM25 R@50 | hybrid−rewired R@50 | decision |
|---|---|---|---|
| frozen_all | +0.2384 | +0.1790 | PASS |

## 6. Reproduction

```bash
cd $PROJECT_ROOT
PY=python
OUT=results/graph_autoresearch/graphauto_20260925_035919_027d9a/literature_set_expansion
$PY -m tasks.agentic_autoresearch.literature_set_expansion --stage build     --out-dir $OUT
$PY -m tasks.agentic_autoresearch.literature_set_expansion --stage screen    --out-dir $OUT
$PY -m tasks.agentic_autoresearch.literature_set_expansion --stage aggregate --out-dir $OUT
$PY -m tasks.agentic_autoresearch.literature_set_expansion --stage report    --out-dir $OUT
```
`--stage screen` is append-only and resumable: rows already present in `retrieval_per_instance.jsonl` are skipped, so re-running it resumes rather than recomputes. Re-running `--stage build` regenerates the identical frozen set (seed 20260925); the frozen IDs were hashed before any arm was compared and were not changed afterwards.

## 7. Files

- `construction_manifest.json` — present, 5,215 bytes
- `temporal_audit.csv` — present, 3,415,234 bytes
- `frozen_ids.jsonl` — present, 2,325,307 bytes
- `frozen_ids_only.txt` — present, 13,200 bytes
- `retrieval_per_instance.jsonl` — present, 6,276,110 bytes
- `retrieval_results.csv` — present, 4,218 bytes
- `retrieval_comparisons.csv` — present, 16,859 bytes
- `gate_decision.json` — present, 2,209 bytes
- `aggregate_summary.json` — present, 16,991 bytes
- `bm25_equivalence_check.json` — present, 255 bytes
- `graph_identity_audit.json` — present, 723 bytes
- `independent_leakage_verification.json` — present, 1,242 bytes
- `report.md` — present, 24,983 bytes
- `superseded_v1/WHY_SUPERSEDED.md` — present, 2,919 bytes

## 8. §4.4 agent stage — evaluation set frozen, no inference run here

The gate passed, so §4.4 is cleared. The agent runner (`litexp_agent.py`) and its aggregator (`litexp_aggregate.py`) are built and dry-run verified offline; **no inference was started by this stage and no GPU or model server was contacted.**

| item | value |
|---|---|
| agent evaluation sources frozen | **150** |
| drawn from | the **screen** split of `frozen_ids.jsonl` (the split the gate was decided on) |
| selection | seeded shuffle (seed 20260925) of the frozen SCREEN-split source ids, first 150 |
| agent ids SHA-256 | `b309861eee606288d77163c059fcefc95c628fdb2e06d3727a3dc02faedcea10` |
| agent ids file SHA-256 | `23429f6159c50b82f170b32f87a089e304085b4a5c7c8b0a078f384fc3de7f15` |
| frozen before any inference | True |
| declared primary comparison | `correct_snowballing_agent > rewired_snowballing_agent on recall@50` |
| smoke slice | the DEV split of the same frozen file; whole (year, primary category) clusters are assigned to dev or screen, so the smoke slice is structurally disjoint from these ids |

Confirmatory family (Holm-adjusted together, and only these):

- `correct_snowballing_agent > rewired_snowballing_agent on recall@50`
- `compact_hybrid_literature_agent > correct_hybrid_fixed_list on recall@50`
- `flat_literature_agent > matched_pool_replay::flat_literature_agent on recall@50`
- `correct_snowballing_agent > matched_pool_replay::correct_snowballing_agent on recall@50`
- `rewired_snowballing_agent > matched_pool_replay::rewired_snowballing_agent on recall@50`
- `compact_hybrid_literature_agent > matched_pool_replay::compact_hybrid_literature_agent on recall@50`

Everything else is reported `secondary_unadjusted`. Ids are scored independently of prose quality; rationale text never enters an effectiveness metric.

No agent episodes exist yet: `agent_results/` currently holds only the frozen ids and their hashes. Agent inference is GPU work owned by another process.

---

# Part II — Snowballing agent experiment (runbook §4.4)

Protocol `litexp_controlled_v1`, the repaired controlled loop. Frozen **150** source papers
(`b309861eee606288…`). Primary declared in code before inference:
`correct_snowballing_agent > rewired_snowballing_agent` on **Recall@50**. Tool uptake **100%**
in every agent condition, so the correct-vs-rewired contrast is informative.

**Weak-label reminder:** scored against the source's own resolved earlier references — an
incomplete, selective label, not exhaustive ground truth (mean unresolved-reference rate 0.68).
Absolute recall is a floor; only paired between-condition differences are interpretable. Ids are
scored independently of prose quality; rationale text never enters an effectiveness metric.

## Confirmatory family (Holm within model)

| model | comparison | diff | 95% CI | p_Holm | sig |
|---|---|---|---|---|---|
| qwen2.5-7b | **correct > rewired snowballing** | **+0.0660** | [+0.0387, +0.0934] | **0.0001** | **yes** |
| qwen3-8b | **correct > rewired snowballing** | **+0.0644** | [+0.0462, +0.0843] | **0.0006** | **yes** |
| qwen3-8b | compact hybrid agent > correct hybrid fixed list | **−0.1421** | [−0.1713, −0.1123] | 0.0006 | yes, **negative** |
| qwen3-8b | flat agent > its matched-pool replay | −0.0002 | [−0.0007, +0.0000] | 1.000 | no |
| qwen3-8b | correct snowballing > its matched-pool replay | −0.0046 | [−0.0119, −0.0002] | 0.494 | no |
| qwen3-8b | rewired snowballing > its matched-pool replay | −0.0022 | [−0.0048, −0.0002] | 0.494 | no |
| qwen3-8b | compact hybrid > its matched-pool replay | −0.0079 | [−0.0201, +0.0000] | 1.000 | no |

## Effectiveness (n=150)

| model | condition | Recall@10 | Recall@20 | validity |
|---|---|---|---|---|
| qwen3-8b | bm25_fixed_list | 0.078 | 0.108 | 1.00 |
| qwen3-8b | correct_hybrid_fixed_list | 0.151 | 0.223 | 1.00 |
| qwen3-8b | flat_literature_agent | 0.070 | 0.100 | 0.99 |
| qwen3-8b | rewired_snowballing_agent | 0.067 | 0.091 | 0.99 |
| qwen3-8b | correct_snowballing_agent | 0.074 | 0.121 | 0.99 |
| qwen3-8b | compact_hybrid_literature_agent | 0.170 | 0.239 | 0.99 |

## Reading

**Correct connectivity is supported, and replicates.** On both models the snowballing agent
recovers significantly more of the source's earlier bibliography with the correct citation graph
than with a degree-preserving rewired one, at 100% tool uptake. Effect sizes agree closely
(+0.066 and +0.064), and the same contrast is significant on nDCG@20/@50 and MAP@50 as secondary
metrics.

**Two negative results, stated directly.**

1. *Sequential interaction buys nothing.* No agent beat its own matched-pool replay on any
   condition; three of four point estimates are slightly negative. Whatever the agent gains comes
   from the candidate pool its tools assemble, not from planning over turns.
2. *The agent is worse than the compiled list it was given.* `compact_hybrid_literature_agent`
   scores **−0.142 Recall@50 below** the fixed correct-hybrid list — a large, significant deficit.
   Handing these models a compiled hybrid ranking beats letting them retrieve for themselves.

Together with §3 (EBC) this gives one consistent picture: **the graph carries real, causally
demonstrable signal, and 7-8B agents under-exploit it.** The connectivity intervention moves what
the agent *sees*; it does not move what the agent *does* with it.

**Caveat on qwen2.5-7b.** Answer validity in its snowballing conditions is low (0.54 correct /
0.64 rewired) — many episodes failed to produce a well-formed ranked set. Validity is *lower* in
the correct condition, so the significant recall advantage is achieved despite that handicap, not
because of it. The qwen3-8b run (validity 0.99) is the cleaner evidence.

## Incomplete work, preserved

An optional third replication on Qwen2.5-32B reached 91/150 on `correct_snowballing_agent` and
0/150 on `rewired_snowballing_agent`. Runbook §7.2 forbids reducing only one side of a paired
comparison, so it was stopped at that boundary and its rows quarantined in
`agent_results/incomplete_32b/` — excluded from every reported number, preserved for a later
resume. Labelled **stopped**, not failed.
