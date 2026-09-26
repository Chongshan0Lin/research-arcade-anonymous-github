# Evidence Bundle Completion (EBC) — construction, leakage audit, retrieval screening, agent-launch gate

Runbook: `xjp/cc_graph_autoresearch_full_experiment_runbook.md`, section 3.
Master run: `results/graph_autoresearch/graphauto_20260925_035919_027d9a/`
Commit at execution: `2631ae6d320009d828da4d451556241f7359fde2`
Executed: 2026-09-25. **CPU only. No GPU, no LLM, no agent was run.** Ports 8011/8021/8022/8031/8032 untouched.

**Gate decision: PASS.** All four launch criteria are met with large margins. The EBC agent
experiment (runbook §3.5) is cleared to proceed; that GPU work is explicitly *not* part of this
report.

---

## 1. Task construction (§3.1)

An EBC instance is a real paragraph that resolves to 2–5 distinct arXiv citations:

* **input** — the source paragraph with *every* citation marker normalised to the neutral token
  `[CITATION]` (no bibliography key survives, anchor or target);
* **revealed evidence** — exactly one anchor citation (id + title), drawn uniformly at random under
  a fixed per-paragraph seed;
* **targets** — the remaining 1–4 citations of the same paragraph, i.e. the complementary evidence
  bundle the researcher must recover.

Source of truth: `data/provenance/paragraph_citation_edges.csv` (resolved paragraph→arXiv citation
edges) joined to the `papers` table on PostgreSQL `localhost:5433`.

### Construction funnel

| stage | count |
|---|---|
| paragraph–citation edges | 1,200,010 |
| resolved arXiv target | 957,831 |
| source date known | 957,831 |
| distinct paragraphs | 165,555 |
| dropped: any target with unknown date | 0 |
| dropped: any target published after the source cutoff | 2,358 |
| paragraphs with a 2–5 citation bundle | 60,298 |
| dropped: masked context under 40 alphabetic tokens | 2,351 |
| **instances built** | **57,947** |
| distinct source papers | 11,807 |

Strict rule: a paragraph is dropped *entirely* if **any** of its resolved citations is undated or
post-cutoff, so every evaluated bundle is the complete real bundle rather than a truncated one.

Artefacts: `data/agentic_autoresearch/evidence_bundle_completion/ebc_instances.jsonl`
(sha256 `9d3875fe252161c721c9c5549a4a4df6020688aac7a8a217bfd5dcf849a3e966`),
`construction_manifest.json`.

---

## 2. Leakage prevention and audit (§3.2) — **zero violations**

For source paper `s` with cutoff `t_s` (= `papers.submit_date` of `s`, day granularity):

1. **Availability.** Every node carries an availability date; a node is traversable or returnable
   only if `avail(v) <= t_s`. Paragraph nodes inherit their owning paper's date; paper nodes with
   unknown dates are never traversable.
2. **Source-edge removal.** A single node mask `group(v) != group(s)` removes `paper:s` and every
   paragraph node owned by `s`, hence *every citation edge and every paragraph edge created by `s`*
   — in **both** graph variants, using the identical node set.
3. **Evaluation paragraph.** The instance's own paragraph node is owned by `s`, so it is masked by
   the same rule and can never create an anchor→target relation.
4. **Candidate pools.** `s` (under any arXiv-version spelling) and the revealed anchor are excluded
   from every arm's output.

### Leakage bug found and fixed by the audit

The first construction pass reported non-zero audit counters. Root cause: the citation graph mixes
arXiv id spellings — `cites` edges use version-stripped ids, while ~20% of `has_paragraph` /
`para_cites` edges carry an explicit version (`paper:2002.08165v2`). Masking on the literal source
id therefore left the source paper's own paragraphs reachable under its versioned twin, i.e. the
evaluation paragraph itself could supply the anchor→target link.

Scale of the exposure on the frozen set: **2,585 / 5,000** instances have a source paper that exists
under a versioned node spelling, and **2,270 / 5,000 (45.4%)** evaluation paragraphs are owned by a
versioned source node. 6,198 base papers appear as both `paper:X` and `paper:Xv*`.

Fix: all paper identity — masking, candidate ids, reachability and score accumulation — now operates
on the **version-stripped base id**; `paper:X` and `paper:Xv2` are one paper whose complementary
edge sets are merged. Construction was rebuilt and re-audited from scratch after the fix. The
superseded first-pass artefacts are preserved under `superseded_v1/` and were never used for any
comparative result.

The 60 instances used to debug the audit are recorded in
`data/agentic_autoresearch/evidence_bundle_completion/dev_ids_debug.txt` and were **held out** of
the refrozen evaluation set (runbook §0.2 disjoint-development rule).

### Audit result

`temporal_audit.csv` — 299,776 rows: 5,000 source rows, 5,000 anchor rows, 8,731 target rows and
281,045 de-duplicated per-candidate rows (top-20 of the five primary arms, with per-candidate date,
legality, ownership, source-edge provenance, hop and — for targets — the 2-hop witness intermediate
and that witness's own availability). Exhaustive checking additionally covered **2,204,525**
candidates (all nine arms at full depth K=50).

| audit counter | value |
|---|---|
| future candidate returned | 0 |
| source paper returned as candidate | 0 |
| revealed anchor returned as candidate | 0 |
| candidate on an edge created by the source | 0 |
| illegal 2-hop witness intermediate | 0 |
| instances with an unmasked source node | 0 |
| instances with an unmasked evaluation paragraph | 0 |
| instances with an illegal anchor / target | 0 |
| **total violations** | **0** |

Row-level recheck of `temporal_audit.csv`: 0 of 299,776 rows flagged `violation=True`.

---

## 3. Frozen slices (§3.3)

Split by **source paper** (70/15/15) with near-duplicate paragraph groups kept inside one split, so
no source paper spans splits: 40,733 train / 8,750 validation / 8,464 test instances.

The screening set is a seeded shuffle of the test split (minus the 60 development ids), first 5,000.

| slice | n instances | n targets | definition |
|---|---|---|---|
| **General EBC** | 5,000 | 8,731 | representative frozen test sample |
| **EBC-GraphHard** | 3,494 | 6,755 | ≥1 target outside BM25 top-20 **and** reachable from the anchor within 1–2 hops in the correct pre-cutoff graph, with a legal witness (path uses neither `s` nor a future node) |

Bundle sizes (targets per instance): 1 → 2,695; 2 → 1,252; 3 → 680; 4 → 373 (mean 1.75).
Anchor degree in the correct graph: median 259, mean 997, p90 3,281. Anchor in graph: 100%.

GraphHard is 69.9% of the frozen set. It is an **intervention-focused challenge slice, not a
representative sample of citation search**: BM25 over a masked paragraph rarely places all cited
papers in its top 20, and 2-hop co-citation reachability is broad, so the criterion is satisfied
often. It is reported as such and never as a general-population estimate.

### Frozen hashes (recorded before any comparative arm was executed)

```
general    n=5000  sha256 89942a2ae3449deabb291f4de105752562f8cd49dbc8d0d5af006c81a6623606
graphhard  n=3494  sha256 4de71d9d85625aa0d405f1479e6ba155a52021220ac4dcc4e53c4de435b72ab5
frozen_ids.jsonl   sha256 063b66b0ca51897f9b02c8447b1698c2b01c564d355787e68b1b89a79f4b3548
ebc_instances      sha256 9d3875fe252161c721c9c5549a4a4df6020688aac7a8a217bfd5dcf849a3e966
citation_graph_full   sha256 c2af169a5ca669af6b9f362cd1fef8a13151a4f28ec8206905f6be324a37c83b
citation_graph_rewire sha256 372a7c56907cf4cd2bf17d715f683a3116af6656766f6a79f0979863692dd33a
paragraph_citation_edges.csv sha256 172dabbae79ab864fd744510edc9fdc5499307ee763cbb7715a22d17b76afe80
```

GraphHard labelling uses only (a) the BM25 ranking and (b) correct-graph reachability. No rewired
arm and no comparative outcome exists at freeze time. The frozen ids were not touched afterwards;
the hashes in `gate_decision.json` were verified to match `construction_manifest.json`.

---

## 4. Retrieval screening (§3.4)

n = 5,000 frozen test instances (well above the ≥500 target). All arms share identical instance
ids, cutoffs, masking and a candidate budget of K = 50. **Correct and rewired differ only in the
backing adjacency**: node ids, node availability dates, ownership metadata, the masked node set,
the ranking rule and the budget are byte-identical.

* **Connectivity is untyped** (relation labels ignored), per runbook §0.2. The typed-edge claim is
  not reopened.
* **Rewired control** is the pre-existing degree-preserving `citation_graph_rewire.jsonl`
  (per-relation double-edge swaps; per-type in/out degree sequences preserved — verified, and the
  aggregate undirected degree sequences of the two variants are identical element-for-element).
* **Preregistered ranking rule** (declared in `ebc_common.PARAMS` before any arm ran):
  `score(p) = 2.0*1[p in N(anchor)] + sum over legal m in N(anchor)&N(p) of 1/log(1+deg m)` —
  Adamic–Adar proximity from the single revealed anchor, ≤2 hops, label-free, identical in every
  variant. Hybrid = reciprocal rank fusion (k=60) of BM25 and the graph list.
* BM25 replicates `tasks/agentic_autoresearch/environment.py:BM25` exactly (verified numerically:
  max absolute score difference 5.7e-14) on a scipy CSR backend for throughput.

### Macro target recall and set metrics

`recall@k` = mean over instances of |top-k ∩ targets| / |targets|.
`set_recall@20` = fraction of instances recovering the **complete** bundle in the top 20.
`set_f1@10` = macro F1 of the top-10 prediction against the target set.
`gonly` = mean number of targets found in the arm's top-20 that BM25's top-20 missed.

**General EBC (n = 5,000)**

| arm | R@5 | R@10 | R@20 | setR@20 | setF1@10 | MRR | gonly |
|---|---|---|---|---|---|---|---|
| bm25 | 0.2124 | 0.2761 | 0.3516 | 0.2514 | 0.0759 | 0.2234 | 0.0000 |
| **graph_correct** | 0.3504 | 0.4506 | 0.5475 | 0.4322 | 0.1252 | 0.3390 | 0.6074 |
| **hybrid_correct** | **0.3886** | **0.5110** | **0.6303** | **0.5090** | **0.1409** | **0.3753** | 0.5172 |
| graph_rewired | 0.0313 | 0.0562 | 0.0907 | 0.0576 | 0.0162 | 0.0346 | 0.1418 |
| hybrid_rewired | 0.1853 | 0.2580 | 0.3396 | 0.2366 | 0.0702 | 0.1670 | 0.0990 |
| graph_random_anchor | 0.0092 | 0.0151 | 0.0228 | 0.0150 | 0.0040 | 0.0088 | 0.0314 |
| hybrid_random_anchor | 0.1606 | 0.2275 | 0.2970 | 0.2078 | 0.0619 | 0.1440 | 0.0210 |
| graph_degree_matched_anchor | 0.0163 | 0.0286 | 0.0471 | 0.0316 | 0.0075 | 0.0168 | 0.0664 |
| anchor_text_only | 0.1321 | 0.1844 | 0.2470 | 0.1736 | 0.0498 | 0.1340 | 0.2014 |

**EBC-GraphHard (n = 3,494)**

| arm | R@5 | R@10 | R@20 | setR@20 | setF1@10 | MRR | gonly |
|---|---|---|---|---|---|---|---|
| bm25 | 0.0723 | 0.0980 | 0.1330 | 0.0000 | 0.0422 | 0.1294 | 0.0000 |
| **graph_correct** | **0.3675** | **0.4700** | **0.5728** | **0.4244** | **0.1404** | **0.3758** | 0.8692 |
| hybrid_correct | 0.2800 | 0.4161 | 0.5545 | 0.3950 | 0.1312 | 0.3033 | 0.7401 |
| graph_rewired | 0.0426 | 0.0760 | 0.1219 | 0.0758 | 0.0222 | 0.0472 | 0.2023 |
| hybrid_rewired | 0.0914 | 0.1302 | 0.1831 | 0.0581 | 0.0475 | 0.1258 | 0.1411 |
| graph_random_anchor | 0.0114 | 0.0186 | 0.0285 | 0.0175 | 0.0051 | 0.0116 | 0.0449 |
| hybrid_random_anchor | 0.0586 | 0.0869 | 0.1191 | 0.0137 | 0.0355 | 0.0882 | 0.0301 |
| graph_degree_matched_anchor | 0.0204 | 0.0366 | 0.0596 | 0.0378 | 0.0100 | 0.0225 | 0.0944 |
| anchor_text_only | 0.0981 | 0.1490 | 0.2070 | 0.1191 | 0.0457 | 0.1152 | 0.2808 |

`setR@20 = 0.0000` for BM25 on GraphHard is true by construction (the slice requires at least one
target outside BM25's top 20). On GraphHard, RRF with a near-useless BM25 list *dilutes* the graph
ranking, so `graph_correct` beats `hybrid_correct` there — the compiled hybrid is the better general
configuration, raw traversal the better hard-case configuration.

### Graph-only discovery

42.2% of instances (hybrid_correct) and 47.4% (graph_correct) recover at least one target in their
top-20 that BM25's top-20 missed. On GraphHard the mean graph-only discovery count is 0.87 targets
per instance for `graph_correct`.

### Target reachability

| | General | GraphHard |
|---|---|---|
| targets | 8,731 | 6,755 |
| reachable ≤1 hop, correct graph | 0.4352 | 0.4422 |
| reachable ≤2 hops, correct graph | 0.9104 | 0.9538 |
| reachable ≤2 hops, **rewired** graph | 0.5186 | 0.5753 |
| **reachable under the declared tool budget** (8 calls x 10 candidates = 80 observable) | **0.7039** | **0.7298** |

Reachability is exact and uncapped (neighbour-set intersection), not a truncated BFS. The rewired
graph retains 52–58% nominal 2-hop reachability because degree-preserving rewiring keeps hub
structure — which is exactly why *ranked* recall, not reachability, is the discriminating metric.

### Paired comparisons

Paired bootstrap, 10,000 resamples, fixed seed 20260925; paired sign-flip permutation test, 10,000
permutations; Holm adjustment inside the declared confirmatory family only.
Full table: `retrieval_comparisons.csv` (both slices, 30 comparisons).

**Declared primary comparison** (fixed before any arm was executed):
`hybrid_correct - hybrid_rewired` on macro target Recall@20.
Confirmatory family: that plus `graph_correct - graph_rewired` at Recall@20.

General EBC, n = 5,000 paired:

| comparison | metric | diff | 95% CI | p (raw) | p (Holm) |
|---|---|---|---|---|---|
| **hybrid_correct - hybrid_rewired** *(primary)* | recall@20 | **+0.2907** | **[+0.2789, +0.3026]** | 1.0e-4 | 2.0e-4 |
| graph_correct - graph_rewired *(family)* | recall@20 | +0.4568 | [+0.4437, +0.4695] | 1.0e-4 | 2.0e-4 |
| hybrid_correct - bm25 | recall@20 | +0.2787 | [+0.2665, +0.2905] | 1.0e-4 | — |
| graph_correct - bm25 | recall@20 | +0.1959 | [+0.1794, +0.2122] | 1.0e-4 | — |
| graph_correct - graph_random_anchor | recall@20 | +0.5247 | [+0.5123, +0.5370] | 1.0e-4 | — |
| graph_correct - graph_degree_matched_anchor | recall@20 | +0.5004 | [+0.4878, +0.5131] | 1.0e-4 | — |
| hybrid_correct - anchor_text_only | recall@20 | +0.3833 | [+0.3701, +0.3964] | 1.0e-4 | — |
| hybrid_correct - hybrid_random_anchor | recall@20 | +0.3333 | [+0.3213, +0.3450] | 1.0e-4 | — |
| hybrid_correct - hybrid_rewired | recall@10 | +0.2530 | [+0.2410, +0.2648] | 1.0e-4 | — |
| hybrid_correct - hybrid_rewired | recall@5 | +0.2032 | [+0.1919, +0.2143] | 1.0e-4 | — |
| hybrid_correct - hybrid_rewired | set_f1@10 | +0.0708 | [+0.0676, +0.0738] | 1.0e-4 | — |
| hybrid_correct - hybrid_rewired | mrr | +0.2082 | [+0.1985, +0.2180] | 1.0e-4 | — |
| graph_correct - graph_rewired | recall@10 | +0.3944 | [+0.3821, +0.4069] | 1.0e-4 | — |
| graph_correct - graph_rewired | recall@5 | +0.3191 | [+0.3071, +0.3312] | 1.0e-4 | — |
| graph_correct - graph_rewired | mrr | +0.3044 | [+0.2938, +0.3152] | 1.0e-4 | — |
| graph_correct - graph_rewired | set_f1@10 | +0.1090 | [+0.1056, +0.1126] | 1.0e-4 | — |

EBC-GraphHard, n = 3,494 paired: `hybrid_correct - hybrid_rewired` recall@20 = **+0.3714**
[+0.3567, +0.3860]; `graph_correct - graph_rewired` recall@20 = **+0.4508** [+0.4354, +0.4662].

p = 9.999e-05 is the floor of a 10,000-permutation test, i.e. no permutation reached the observed
effect; it is reported as `< 1e-4` rather than as an exact value.

The degree-matched anchor control is the decisive negative control: a *different* paper with the
same degree recovers 0.0471 R@20 against the true anchor's 0.5475. The effect is anchor identity
plus correct topology, not degree or hub proximity.

---

## 5. Agent-launch gate (§3.4) — **PASS**

Evaluated mechanically in `ebc_screen.py`; written to `gate_decision.json`.

| # | criterion | threshold | observed | result |
|---|---|---|---|---|
| 1 | correct graph **or** correct hybrid beats its rewired counterpart on absolute Recall@20 | ≥ 0.05 | graph **+0.4568**, hybrid **+0.2907** (met by both) | **PASS** |
| 2 | paired 95% CI of the declared primary correct-minus-rewired comparison strictly positive | lo > 0 | [+0.2789, +0.3026], n=5,000, 10,000 resamples, seed 20260925, p_Holm = 2.0e-4 | **PASS** |
| 3 | GraphHard targets reachable under the declared tool budget (8 calls x 10 candidates) | ≥ 0.30 | **0.7298** (4,930 / 6,755 targets) | **PASS** |
| 4 | zero temporal / source-edge leakage | 0 | 0 violations over 2,204,525 audited candidates and 299,776 audit rows | **PASS** |

**Decision: PASS.** The EBC agent experiment (runbook §3.5) is cleared. No agent was run here —
GPU work is out of scope for this task and was not started.

---

## 6. Scope, caveats and honest limits

* This is a **retrieval screening** result. It shows the correct citation graph carries
  evidence-bundle signal that lexical retrieval and a degree-preserving rewired control do not. It
  says nothing about whether an agent will *use* that signal; the previous natural-action run found
  zero tool calls, and that remains a separate, unresolved interface question.
* A 2-hop witness path is **retrieval provenance**, not semantic proof that the recovered paper
  supports the paragraph. Witness intermediates are stored per target in
  `retrieval_per_instance.jsonl` so any downstream faithfulness claim can be joined and checked.
* Weak supervision: an author's citation bundle is selective, not exhaustive. Papers that *should*
  have been cited count as misses for every arm equally.
* GraphHard is 69.9% of the frozen sample and is an intervention slice, not a representative
  estimate of citation search difficulty.
* The correct/rewired manipulation removes an *identical* node set (the source paper and its
  paragraphs) from both graphs. Degree preservation is a whole-graph property, so the two induced
  subgraphs after that removal are not degree-identical down to the last node; the removed set is
  ~1 paper plus its paragraphs out of 232,247 nodes, and the removal rule is byte-identical across
  variants.
* Availability is compared at day granularity because `papers.submit_date` is a `DATE`; a paper
  submitted on the same day as the source counts as available. `strictly_before_cutoff` is recorded
  per audited object in `temporal_audit.csv` for anyone wanting the stricter reading.
* 6,211 of 52,307 paper nodes exist under versioned spellings; these are merged into their base
  paper rather than being dropped or double-counted.

---

## 7. Files

```
results/graph_autoresearch/graphauto_20260925_035919_027d9a/evidence_bundle_completion/
  construction_manifest.json      parameters, funnel, leakage policy, splits, all SHA-256 hashes
  frozen_ids.jsonl                5,000 frozen ids with slice membership, anchor and targets
  temporal_audit.csv              299,776 per-object / per-candidate / per-witness audit rows
  temporal_audit_summary.json     roll-up, zero_violations = true
  retrieval_results.csv           9 arms x 2 slices x 18 metrics
  retrieval_comparisons.csv       30 paired comparisons with CIs, raw p and Holm-adjusted p
  retrieval_per_instance.jsonl    raw per-instance rows (all arms, top-20 lists, reachability)
  reachability.json               reachability + tool-budget summary per slice
  gate_decision.json              mechanical gate evaluation, PASS
  report.md                       this file
  superseded_v1/                  first-pass construction artefacts, superseded by the leakage fix

data/agentic_autoresearch/evidence_bundle_completion/
  ebc_instances.jsonl             57,947 instances with splits
  ebc_structural_labels.jsonl     BM25 top-20 + reachability labels for the frozen set
  dev_ids_debug.txt               60 development ids held out of the evaluation set

tasks/agentic_autoresearch/
  ebc_common.py                   preregistered parameters, corpus, CSR graphs, sparse BM25
  ebc_retrieval.py                anchor expansion, exact <=2-hop reachability, arms, metrics
  ebc_build.py                    construction, splits, GraphHard labelling, freeze, audit
  ebc_screen.py                   screening, paired statistics, gate
```

## 8. Reproduction

From the repository root, with `python`:

```bash
# 1. construct, split, label, freeze, construction audit  (~25 min, CPU only)
python -m tasks.agentic_autoresearch.ebc_build \
  --rebuild \
  --exclude-ids data/agentic_autoresearch/evidence_bundle_completion/dev_ids_debug.txt \
  --out results/graph_autoresearch/graphauto_20260925_035919_027d9a/evidence_bundle_completion

# 2. retrieval screening, paired statistics, gate         (~4 min, CPU only; resumable)
python -m tasks.agentic_autoresearch.ebc_screen \
  --out results/graph_autoresearch/graphauto_20260925_035919_027d9a/evidence_bundle_completion
```

`ebc_screen` is append-only and resumable by `instance_id`: rerunning it reuses completed rows and
only recomputes aggregates, statistics and the gate. `--force` discards the per-instance rows and
rescreens from scratch.

## 9. Verification performed

* aggregates in `retrieval_results.csv` recomputed from `retrieval_per_instance.jsonl` — exact match
  for every arm, slice and metric;
* the primary bootstrap CI recomputed independently — exact match (+0.29065, [+0.2789, +0.3026]);
* frozen id set hash recomputed from `frozen_ids.jsonl` — matches both `construction_manifest.json`
  and `gate_decision.json`;
* one row per frozen instance, no duplicate keys, screened set == frozen set;
* `temporal_audit.csv` re-read row by row — 0 of 299,776 rows flagged as violations;
* correct and rewired graphs: identical node sets, identical edge counts (989,123), element-wise
  identical aggregate degree sequences, different topology;
* the previous canonical run `results/agentic_autoresearch/20260924_182602_a26e2e` has no file
  modified after 2026-09-25 03:00 — untouched;
* `git diff --check` passes; no unrelated worktree change was modified; no destructive git command
  was used.

---

# Part III — EBC agent experiment (runbook §3.5)

Protocol `ebc_controlled_v1`, the repaired controlled loop adapted to a set-valued answer.
Model **qwen3-8b** (cleanest controlled smoke). Frozen **150** instances
(`agent_ids_sha256 32e72938d23e8b96…`), 110 of them GraphHard, labelled per row and never merged
into the general slice. Declared **before** inference: primary causal comparison
`correct_traversal_agent > rewired_traversal_agent` on macro target **Recall@10**.

**Protocol disclosure:** each condition enforces a mandatory minimum research phase; the anchor
citation is revealed in the opening prompt (that is the task definition, not leakage); agents get
no prepopulated candidate list.

## Effectiveness (n=150 paired)

| condition | Recall@10 | Recall@5 | set F1 | set recall | MRR |
|---|---|---|---|---|---|
| bm25_one_shot | 0.236 | 0.229 | 0.103 | 0.147 | 0.275 |
| flat_controlled_agent | 0.254 | 0.249 | 0.105 | 0.180 | 0.280 |
| **rewired_traversal_agent** | 0.205 | 0.202 | 0.085 | 0.147 | 0.202 |
| **correct_traversal_agent** | **0.307** | 0.302 | 0.130 | 0.213 | 0.326 |
| correct_hybrid_one_shot | 0.427 | 0.410 | 0.186 | 0.293 | 0.375 |
| compact_hybrid_agent | 0.426 | 0.399 | 0.183 | 0.300 | 0.386 |

## Primary family (Holm-adjusted)

| comparison | diff | 95% CI | p_Holm |
|---|---|---|---|
| **correct_traversal > rewired_traversal** | **+0.1017** | **[+0.0528, +0.1522]** | **0.0006** |
| compact_hybrid > correct_hybrid_one_shot | −0.0017 | [−0.0611, +0.0561] | 1.000 |
| compact_hybrid > its matched-pool replay | +0.0428 | [+0.0067, +0.0817] | 0.139 |
| correct_traversal > its matched-pool replay | +0.0072 | [−0.0317, +0.0456] | 1.000 |
| flat_controlled > its matched-pool replay | +0.0383 | [+0.0017, +0.0778] | 0.196 |
| rewired_traversal > its matched-pool replay | −0.0033 | [−0.0333, +0.0267] | 1.000 |

Traversal uptake **98%**, so the primary comparison is informative rather than a failed-uptake
check. Invariants recomputed from raw rows: prompt/schema hashes identical across correct and
rewired, graph hashes differ, degree hashes match, observations not universally identical.

## Graph-only target discovery

Targets found that the flat agent never observed:

| condition | targets | graph-only targets | rate | episodes with ≥1 | graph-only **selected** |
|---|---|---|---|---|---|
| correct_traversal | 260 | **81** | **31.2%** | 43.3% | 30 (11.5%) |
| rewired_traversal | 260 | 21 | 8.1% | 12.7% | 9 (3.5%) |
| compact_hybrid | 260 | 65 | 25.0% | 36.0% | 54 (20.8%) |

Correct connectivity surfaces ~4× as many otherwise-unreachable targets as the degree-preserving
rewired control.

## Faithfulness and provenance

Evidence faithfulness **1.00** in every condition (every cited id was actually observed). For
graph-derived selected targets the anchor→target witness path was re-verified in that condition's
own pre-cutoff graph: **105/105 verified** for compact hybrid, path-faithful rate 1.00.
Action-parser validity 0.976-0.990.

**A verified path is retrieval provenance only.** It shows the target was reachable from the
anchor through legal pre-cutoff edges. It is *not* evidence that the paper supports the paragraph.

## Reading

On evidence-bundle completion the graph effect **survives end to end**: an agent with correct
connectivity recovers significantly more missing bundle members than the same agent with a
degree-preserving rewired graph, and finds four times as many targets lexical search never
surfaces.

Two honest qualifications. First, raw traversal (0.307) remains well below the *compiled* hybrid
one-shot list (0.427) — the compiled retrieval interface beats what this agent does with raw
traversal. Second, no agent beat its own matched-pool replay after correction, so the gain comes
from the candidate pool the graph assembles, not from sequential planning.

Contrast with controlled Task-A, where the same intervention moved discovery (+0.125/+0.080,
p≤0.002) but not the answer: EBC's metric rewards *surfacing* targets, which is exactly the stage
where connectivity helps. Task-A's Top-1 demands picking one correct paper out of candidates
already seen, which is where these 7-8B agents fail.

## Part IV — three-model replication of the primary causal comparison

Added after the qwen3-8b run, using GPUs freed by controlled Task-A. Same 150 frozen instances,
same protocol, same declared primary comparison. Nothing about the design was changed; these are
additive replications of an already-decided contrast.

`correct_traversal_agent > rewired_traversal_agent`, macro target **Recall@10**, paired n=150,
10,000 resamples, Holm within the confirmatory family:

| model | traversal uptake | diff | 95% CI | p_Holm | decision |
|---|---|---|---|---|---|
| qwen2.5-7b-instruct | 97.3% | **+0.0889** | [+0.0428, +0.1378] | 0.0012 | **Supported** |
| qwen3-8b | 98.0% | **+0.1017** | [+0.0528, +0.1522] | 0.0006 | **Supported** |
| qwen2.5-32b-instruct (Int4) | — | **+0.1028** | [+0.0550, +0.1533] | 0.0001 | **Supported** |

The effect is stable across a 4.5× parameter range (+0.089 to +0.103) and significant after
correction on every model. Secondary metrics agree in sign and significance on all three
(Recall@5, set F1, set recall, MRR).

**Caveat worth recording:** on qwen2.5-7b the *validity* rate is significantly **lower** in the
correct condition than the rewired one (−0.107 [−0.180, −0.040], p_Holm 0.004). Correct traversal
returns more usable candidates, which lengthens the transcript and appears to cost that weaker model
some well-formed answers. The recall gain therefore understates the retrieval effect for that model
and is not an artefact inflating it. qwen3-8b shows no validity difference at all (0.000).
