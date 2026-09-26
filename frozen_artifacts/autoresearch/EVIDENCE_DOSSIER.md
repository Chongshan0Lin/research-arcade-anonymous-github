# Graph × AutoResearch — complete evidence dossier

**Run** `graphauto_20260925_035919_027d9a` · commit `2631ae6d3200` · executed 2026-09-25,
audited 2026-09-26 · 10,402 episodes · §12 verification **14/14 PASS**

Everything below is traceable to a frozen artifact under
`results/graph_autoresearch/graphauto_20260925_035919_027d9a/`. Nothing is inferred or
back-filled. Where a quantity could not be recovered it is marked **missing** with the artifact
that would be needed. Where a result is negative it is stated as negative.

Naming: the paper's **Citation Set Completion (CSC)** is `evidence_bundle_completion` on disk;
**Literature Set Expansion (LSE)** is `literature_set_expansion`.

---

# Part 0 — What was asked, and what the evidence supports

| Question | Answer | Strength |
|---|---|---|
| Does correct citation connectivity improve candidate discovery? | **Yes** | fixed retrieval, n=5000 / n=909, CIs exclude 0 at the permutation floor |
| Does it still help when an agent must traverse for itself? | **Yes, on set-recovery tasks** | replicated on 3 models (CSC) and 2 models (LSE), Holm-significant |
| Does it help end-to-end on single-target Top-1? | **No** | Task-A: nothing survives Holm; discovery moves, selection does not |
| Do small agents use graph tools spontaneously? | **No** | 0 tool calls in 950 natural-interface episodes across 3 scales |
| Does sequential agentic interaction beat a matched candidate pool? | **No** | no agent beats its own replay after correction, anywhere |
| Is raw traversal the best way to use the graph? | **No** | a compiled hybrid list beats the agent by 0.142 R@50 on LSE |

**One-sentence summary.** The citation graph carries a large, causally demonstrated retrieval
signal; 7–8B agents convert that signal into *discovery* reliably and into *answers* only when the
metric rewards surfacing a set rather than picking one item — and none of the gain comes from
sequential planning.

---

# Part 1 — The zero-tool-call diagnosis (prerequisite to everything else)

The prior canonical run `20260924_182602_a26e2e` found correct and rewired graphs performing
*identically*. That was not a null result about connectivity; it was a failed manipulation.

## 1.1 Direct evidence

All **2,614 / 2,614** cached generations across both models parsed as a well-formed immediate final
answer:

```
$ grep -o '"tool"[[:space:]]*:[[:space:]]*"[a-z_]*"' cache_*.jsonl | sort | uniq -c
   1307 cache_qwen2.5-7b-instruct.jsonl:"tool": "submit"
   1307 cache_qwen3-8b.jsonl:"tool": "submit"
```

Zero malformed outputs, zero parser failures, zero blocked calls. Every episode terminated on its
first generation.

## 1.2 Cause

Two compounding interface faults, both in `canonical_agent.py`:

1. **Prepopulation** — `opening = render(bm_list[:ANSWER_K*2], papers)` put 10 BM25 candidates into
   the first user turn of every agent condition. There was nothing to discover.
2. **Terminal instruction** — `task_prompt()` appended the answer spec unconditionally, as the
   *last* line: `Reply with ONE JSON object: {"tool": "submit", ...} using EXACTLY 5 distinct ids`.
   Submitting on turn 1 was the compliant reading.

A latent third fault did not fire but was fixed: the single `parse()` fell through any unrecognised
object to a `submit`, so a malformed action would have silently ended the episode as an answer.

## 1.3 Why the equality was uninformative

With zero tool calls, `flat`, `correct_connectivity` and `compact_hybrid` all degenerate to the
same one-shot selection over the same pool. Their equality carries no information about topology.
Confirmed at a third scale: Qwen2.5-32B returned **numerically identical 0.240 Top-1** in all three
natural conditions.

Artifacts: `tool_loop_root_cause.md`, `results/agentic_autoresearch/20260924_182602_a26e2e/cache_*.jsonl`

---

# Part 2 — The repair, and proof that it worked

New protocol `controlled_interaction_v1` (`tasks/agentic_autoresearch/controlled_agent.py`).
`canonical_agent.py` was left untouched so the natural-action run stays reproducible.

Changes: separate action/final parsers (an unrecognised object is malformed, never a silent
submit); no prepopulated candidate list; a **disclosed mandatory minimum research phase** per
condition; one deterministic format-repair turn; one forced-answer turn on budget exhaustion,
identical in every condition; anytime predictions never back-filled; six call counters kept
separately; correct and rewired share byte-identical prompts and tool schemas, asserted at run time
by `prompt_schema_hash`.

## 2.1 Smoke gate — 41/41 checks passed

| Quantity | Natural interface | Controlled interface |
|---|---|---|
| episodes with ≥1 successful tool call | **0%** | **95–100%** |
| mean successful tool calls | 0.00 | 3.5–5.9 |
| traverse uptake (graph conditions) | 0% | 95–100% |
| action-parser validity | — | 96.9% |
| correct vs rewired observed pools differ | n/a | every instance |

Artifacts: `controlled_task_a/smoke_report.md`, `smoke_gate.json`

## 2.2 Full-run uptake (n=200 per cell)

| Model | Condition | Mean calls | Traverse | Hybrid |
|---|---|---|---|---|
| Qwen2.5-7B | flat | 5.01 | — | — |
| Qwen2.5-7B | correct graph | 5.42 | **0.975** | — |
| Qwen2.5-7B | rewired graph | 5.90 | **0.975** | — |
| Qwen2.5-7B | compact hybrid | 5.27 | — | **0.990** |
| Qwen3-8B | flat | 5.15 | — | — |
| Qwen3-8B | correct graph | 3.86 | **0.980** | — |
| Qwen3-8B | rewired graph | 4.09 | **0.980** | — |
| Qwen3-8B | compact hybrid | 5.55 | — | **0.995** |

The uptake gate is satisfied, so every correct-vs-rewired comparison below is *informative* rather
than a failed-uptake check.

---

# Part 3 — Task-A: the intervention works at discovery and dies at selection

Same frozen 200 instances as the natural run (`ec357bef29eece81`), so the two interfaces are
directly comparable.

## 3.1 The decomposition (the central finding of the round)

Correct vs degree-preserving-rewired connectivity, paired n=200, 10,000 resamples:

| Funnel stage | Qwen2.5-7B | p | Qwen3-8B | p |
|---|---|---|---|---|
| **gold ever observed** (discovery) | 0.480 vs 0.355 → **+0.125** [+0.070, +0.185] | **0.0001** | 0.425 vs 0.345 → **+0.080** [+0.035, +0.125] | **0.0017** |
| **gold selected** (in answer) | 0.260 vs 0.235 → +0.025 [−0.015, +0.065] | 0.329 | 0.300 vs 0.280 → +0.020 [−0.020, +0.060] | 0.454 |
| **Top-1** | 0.155 vs 0.155 → +0.000 | 1.000 | 0.220 vs 0.185 → +0.035 | 0.038 |

At 98% traversal uptake, correct connectivity significantly surfaces the gold paper on **both**
models. The advantage then does not propagate to the answer.

## 3.2 Absolute Top-1 (n=200)

| Condition | Qwen2.5-7B | Qwen3-8B |
|---|---|---|
| bm25 one-shot (ctrl) | 0.155 | 0.175 |
| correct hybrid one-shot (ctrl) | 0.185 | 0.205 |
| flat controlled agent | 0.185 | 0.210 |
| rewired graph agent | 0.155 | 0.185 |
| correct graph agent | 0.155 | 0.220 |
| compact hybrid agent | **0.240** | **0.230** |
| replay :: compact hybrid | 0.215 | 0.235 |
| replay :: correct graph | 0.140 | 0.180 |

One-shot controls were **re-run** under the ctrl-v1 prompt/parser, not reused: reuse is permitted
only when prompt, parser and answer semantics are identical, and ctrl-v1 answers are
`{"final":[...]}` rather than `{"tool":"submit",...}`.

## 3.3 Preregistered family — nothing survives Holm

| Model | Comparison | diff | 95% CI | p_raw | p_Holm |
|---|---|---|---|---|---|
| Qwen2.5-7B | correct > rewired | +0.000 | [−0.035, +0.035] | 1.000 | 1.000 |
| Qwen3-8B | correct > rewired | +0.035 | [+0.010, +0.065] | 0.038 | 0.368 |
| Qwen2.5-7B | compact hybrid > hybrid one-shot | +0.055 | [−0.005, +0.115] | 0.123 | 1.000 |
| Qwen3-8B | compact hybrid > hybrid one-shot | +0.025 | [−0.040, +0.085] | 0.550 | 1.000 |
| Qwen2.5-7B | compact hybrid > its replay | +0.025 | [−0.025, +0.075] | 0.420 | 1.000 |
| Qwen3-8B | compact hybrid > its replay | −0.005 | [−0.050, +0.040] | 1.000 | 1.000 |
| Qwen2.5-7B | controlled > natural (compact hybrid) | +0.070 | [+0.010, +0.130] | 0.033 | 0.333 |
| Qwen3-8B | controlled > natural (compact hybrid) | +0.070 | [+0.010, +0.130] | 0.037 | 0.368 |
| Qwen3-8B | controlled > natural (correct graph) | +0.060 | [+0.010, +0.110] | 0.037 | 0.368 |

**Task-A verdict:** connectivity-helps-under-controlled-traversal is **Not supported** on Top-1 for
both models. This is a genuine negative at 98% uptake, not a failed-uptake check. Scope it as
*end-to-end candidate-to-answer utility on the canonical n=200 subset*; it does not contradict the
full-scale structural retrieval evidence, and the discovery-stage result above is positive.

Artifacts: `controlled_task_a/{metrics_by_condition,paired_comparisons,candidate_funnel,tool_utilization,faithfulness}.csv`,
`claim_decisions.json`, `report.md`

---

# Part 4 — Citation Set Completion (CSC)

## 4.1 Fixed retrieval, general slice, n=5000

| Arm | R@5 | R@10 | R@20 | MRR | set F1 |
|---|---|---|---|---|---|
| BM25 | 0.212 | 0.276 | 0.352 | 0.223 | 0.076 |
| graph, rewired | 0.031 | 0.056 | 0.091 | 0.035 | 0.016 |
| graph, correct | 0.350 | 0.451 | 0.547 | 0.339 | 0.125 |
| hybrid, rewired | 0.185 | 0.258 | 0.340 | 0.167 | 0.070 |
| **hybrid, correct** | **0.389** | **0.511** | **0.630** | **0.375** | **0.141** |
| graph, random anchor | 0.009 | 0.015 | 0.023 | 0.009 | 0.004 |
| graph, degree-matched anchor | 0.016 | 0.029 | 0.047 | 0.017 | 0.008 |
| anchor text only | 0.132 | 0.184 | 0.247 | 0.134 | 0.050 |

Contrasts (paired, 10,000 resamples; p at the permutation floor <1e-4, Holm 2e-4):

| Contrast | diff | 95% CI |
|---|---|---|
| hybrid correct − hybrid rewired (**primary**) | **+0.291** | [+0.279, +0.303] |
| graph correct − graph rewired | +0.457 | [+0.444, +0.470] |
| hybrid correct − BM25 | +0.279 | [+0.266, +0.290] |
| graph correct − degree-matched anchor | +0.500 | [+0.488, +0.513] |

**The decisive control:** a *different* paper with the *same degree* reaches R@20 = 0.047 against
0.547 for the true anchor. The effect is anchor identity combined with correct topology — not
degree, not hub proximity.

## 4.2 Agents, n=150 frozen, primary = correct − rewired on R@10

| Model | Traverse uptake | diff | 95% CI | p_Holm |
|---|---|---|---|---|
| Qwen2.5-7B | 0.973 | **+0.0889** | [+0.0428, +0.1378] | 0.0012 |
| Qwen3-8B | 0.980 | **+0.1017** | [+0.0528, +0.1522] | 0.0006 |
| Qwen2.5-32B (Int4) | 0.940 | **+0.1028** | [+0.0550, +0.1533] | 0.0001 |

Stable across a 4.5× parameter range. Secondary metrics (R@5, set F1, set recall, MRR) agree in
sign and significance on all three.

Absolute R@10 (Qwen3-8B): BM25 one-shot 0.236 · flat agent 0.254 · rewired traversal 0.205 ·
correct traversal 0.307 · compact hybrid 0.426 · **fixed correct hybrid 0.427**.

## 4.3 Mechanism

| Model | Condition | Target seen | Target selected | Graph-only targets | Episodes w/ ≥1 graph-only |
|---|---|---|---|---|---|
| Qwen3-8B | flat | 0.311 | 0.231 | — | — |
| Qwen3-8B | rewired | 0.265 | 0.185 | **0.081** | **0.127** |
| Qwen3-8B | correct | **0.542** | 0.288 | **0.311** | **0.433** |
| Qwen2.5-7B | rewired | 0.285 | 0.127 | 0.119 | 0.187 |
| Qwen2.5-7B | correct | 0.615 | 0.215 | 0.392 | 0.480 |
| Qwen2.5-32B | rewired | 0.365 | 0.231 | 0.365 | 0.533 |
| Qwen2.5-32B | correct | 0.589 | 0.327 | 0.589 | 0.720 |

"Graph-only" = target observed here but never observed by the matched flat-search agent. Correct
connectivity surfaces ~4× as many otherwise-unreachable targets.

Failure taxonomy (Qwen3-8B, n=150): correct connectivity halves the dominant failure —
*target never retrieved* drops **61.3% → 31.3%** — while *observed but not selected* rises
**10.0% → 26.7%**. The residual bottleneck is selection.

## 4.4 The illustrative trace — `EBC-3452627`

2 targets, anchor `2009.02252`, byte-identical prompts in all three conditions.

| Condition | Actions | Outcome | R@10 |
|---|---|---|---|
| correct traversal | `search("retrieval augmented generation benchmarks")`, `traverse(2009.02252)` | both targets recovered; `1705.03551` first seen at call 2, 1 hop from anchor, path verified | **1.000** |
| rewired traversal | **identical two actions, identical query** | neither target returned | 0.000 |
| flat search | `search` ×8 — same query repeated 7× after the first | neither target returned | 0.000 |

The correct and rewired agents behave *identically*; only the adjacency behind `traverse` differs.
This is the cleanest available demonstration that the gain is topological, not policy-driven. The
flat agent's degenerate repetition also explains its higher token cost (2883 vs 1822).

Artifact: `appendix/trace_pair.json`

---

# Part 5 — Literature Set Expansion (LSE)

## 5.0 Why n=909 and not n=1200 — a split, not an exclusion

All 1,200 frozen LSE sources are retained; **nothing is discarded after freezing**. The frozen set
is partitioned by a *predeclared* grouped dev/screen split on (year, primary arXiv category)
clusters: 13 of 116 clusters (291 sources) → dev, the remaining 103 clusters (909) → screen, and
291 + 909 = 1200 exactly. Whole clusters go to one side, so the realised 24.3% dev share does not
land exactly on the nominal 20%. The `split` label is stored per record inside `frozen_ids.jsonl`.

Predeclaration is verifiable from timestamps, not just asserted: frozen ids + split labels written
**04:45**, first retrieval numbers **05:27**, gate decision **05:33**. No split assignment could
have been informed by an observed score. The rule itself is in
`construction_manifest.json → predeclared_rules.splits`.

**The screen split (n=909) is the single canonical LSE evaluation set** and is what the main text
reports. Metrics over all 1,200 were also computed and kept for completeness; that superset *also*
passes the gate (+0.2384 vs BM25, +0.1790 vs rewired), so screen is not a winner chosen among a
passing and a failing variant. The two must never be mixed in one table. The 150 agent instances
come entirely from screen → strict nesting **1200 ⊃ 909 ⊃ 150**. Dev was used only for interface
debugging and the agent smoke test.

## 5.1 Fixed retrieval, screen split, n=909

| Arm | R@20 | R@50 | nDCG@50 |
|---|---|---|---|
| BM25 | 0.129 | 0.191 | 0.177 |
| graph, rewired | 0.096 | 0.153 | 0.124 |
| graph, correct | 0.283 | 0.401 | 0.366 |
| hybrid, rewired | 0.158 | 0.252 | 0.203 |
| **hybrid, correct** | **0.288** | **0.427** | 0.362 |
| graph, citation edges only | 0.283 | 0.401 | 0.366 |
| graph, paragraph edges only | 0.152 | 0.180 | 0.176 |

| Contrast | diff | 95% CI |
|---|---|---|
| hybrid correct − hybrid rewired (**primary**) | **+0.175** | [+0.164, +0.186] |
| hybrid correct − BM25 | **+0.235** | [+0.224, +0.246] |

Note: hybrid correct beats graph correct by only +0.026 on R@50 and is **−0.005 on nDCG@50**
(CI includes 0) — fusion buys coverage, not ranking quality.

## 5.2 Agents, n=150 frozen, primary = correct − rewired on R@50

| Model | Tool uptake | diff | 95% CI | p_Holm |
|---|---|---|---|---|
| Qwen2.5-7B | 1.000 | **+0.0660** | [+0.0387, +0.0934] | 0.0001 |
| Qwen3-8B | 1.000 | **+0.0644** | [+0.0462, +0.0843] | 0.0006 |

Absolute R@50 (Qwen3-8B): flat 0.123 · rewired snowballing 0.111 · correct snowballing 0.175 ·
compact hybrid agent 0.281 · **fixed correct hybrid 0.423** · fixed BM25 0.176.

## 5.3 Weak-supervision caveat

Gold = the source's own resolved earlier references. Mean unresolved-reference rate on the frozen
set is **0.68** — roughly two-thirds of bibliography entries never resolve to an in-corpus paper.
Absolute LSE recall is therefore a **floor**, not a measurement of topical recall; only paired
between-condition differences are interpretable. Gold set size: mean 16.6, median 14, range 5–88.

---

# Part 6 — The three negative results

These are as load-bearing as the positives and are stated at full strength.

## 6.1 Sequential interaction adds nothing detectable

No agent beats its own matched-pool replay after Holm correction, in either workflow:

| Workflow / Model | Comparison | diff | 95% CI | p_Holm |
|---|---|---|---|---|
| CSC, Qwen3-8B | correct traversal − its replay | +0.0072 | [−0.0317, +0.0456] | 1.000 |
| CSC, Qwen3-8B | compact hybrid − its replay | +0.0428 | [+0.0067, +0.0817] | 0.139 |
| LSE, Qwen3-8B | correct snowballing − its replay | −0.0046 | [−0.0119, −0.0002] | 0.494 |
| LSE, Qwen3-8B | flat − its replay | −0.0002 | [−0.0007, +0.0000] | 1.000 |
| Task-A, Qwen3-8B | compact hybrid − its replay | −0.005 | [−0.050, +0.040] | 1.000 |

**We do not claim agents outperform matched replays.** Whatever an agent gains comes from the
candidate pool its tools assemble, not from planning across turns.

## 6.2 A compiled retrieval interface beats agentic traversal

| Workflow | Fixed correct hybrid | Best agent | Gap |
|---|---|---|---|
| LSE (R@50) | **0.423** | 0.281 | agent is **−0.142** [−0.171, −0.112], p_Holm 0.0006 |
| CSC (R@10) | 0.427 | 0.426 | parity |

Handing these models a compiled hybrid ranking is at least as good as letting them retrieve for
themselves, and on LSE it is substantially better.

## 6.3 Scale does not buy spontaneous tool use

Qwen2.5-32B-Instruct (Int4), n=50 of the frozen 200:

| Interface | Condition | Mean calls | Uptake | Top-1 |
|---|---|---|---|---|
| natural | flat | 0.00 | **0%** | 0.240 |
| natural | correct graph | 0.00 | **0%** | 0.240 |
| natural | compact hybrid | 0.00 | **0%** | 0.240 |
| controlled | flat | 3.62 | **100%** | 0.300 |
| controlled | correct graph | 4.40 | **100%** | 0.320 |
| controlled | compact hybrid | 3.38 | **100%** | **0.400** |

All three natural conditions return **identical 0.240** — the collapse made visible. Repairing the
interface restores both uptake and separation. Exploratory (n=50, one seed, quantised checkpoint);
the binary fact of zero tool calls is not exploratory.

---

# Part 7 — Integrity: two real bugs caught

## 7.1 arXiv version-spelling leakage (found in construction, affected both workflows)

`cites` edges are version-stripped, but **73,178 of 165,555** `has_paragraph` edges carry an
explicit version (e.g. `paper:2002.08165v2`). **6,198** base identifiers existed as two unconnected
nodes. Masking on the literal source identifier therefore left the source reachable through its
versioned twin — exactly the forbidden source-derived channel.

| Workflow | Exposure |
|---|---|
| CSC | 45.4% of frozen instances had a source owning a versioned node |
| LSE | 223/1200 sources (18.6%), **3,070** source-owned paragraph nodes left unmasked |

Fix: all paper identity resolved on the version-stripped base id in every graph variant; degree
preservation re-verified after the merge. Both workflows rebuilt and re-audited.

**Post-fix:** CSC 0 violations over **2,204,525** checked candidates; LSE 0 future-dated, 0
source-in-results, 44,346 blocked source-edge traversal attempts, plus an independent re-audit of
15,000 returned identifiers against a freshly queried date table.

Superseded artifacts preserved in `*/superseded_v1/` with `WHY_SUPERSEDED.md` and used for nothing.
Honest sizing: two deliberately-broken controls measured the bug at only **−0.0032 R@50** under this
expansion policy — real and structural, but small here. It was still required to fix, because
exposure was non-uniform and its size under an agent with a direct co-citation tool is not bounded
by what it was worth in fixed retrieval.

## 7.2 Dispatch-before-permission-check (in my own agent loop)

`_dispatch` ran *before* the "is this tool allowed?" check and its results entered `observed`
unconditionally. A graph-free condition emitting `hybrid_search` would have silently received
graph-derived candidates.

**Verified zero contamination:** across 2,338 rows there were **0 blocked calls** and **0
out-of-interface emissions**, so no reported number is affected. Patched to refuse before dispatch;
`traverse` additionally guarded against a missing-adjacency `KeyError`. Two permanent verification
checks added — *"blocked calls returned no candidates"* and *"no out-of-interface tool executed"* —
so this cannot pass silently again.

## 7.3 §12 verification — 14/14

previous canonical run preserved · frozen split hashes recorded · identical IDs across all 12 cells
of 200 · 0 duplicate episode keys · one terminal row per expected episode · graph hashes differ ·
degree preservation holds · correct/rewired prompts and schemas match · all cited evidence joins to
observed evidence · blocked calls returned no candidates · no out-of-interface tool executed · all
2,400 matched-pool hashes reproduce · uptake gate passed · `git diff --check` clean.

One process error worth recording: two orchestrators were briefly alive at once and double-wrote
episode files (1,093 rows → 548 unique). Caught, de-duplicated by `(model, condition, instance_id)`,
and prevented thereafter by a `flock` guard plus a duplicate-key check that fails the build.

---

# Part 8 — A2 reviewer-request evidence: descriptive only

Audited all 64 instances **before** any inference. **10 passed → `descriptive_appendix_only`**
(gate is ≥40). No confirmatory statistical claims are made.

| Check | Failures |
|---|---|
| genuine reviewer request / rebuttal cue | 34 |
| rebuttal phase correctly identified | 28 |
| input does not reveal the target | 12 |
| indexes can resolve the target | 11 |
| target responsive to the cue | 4 |
| target existed by cutoff | 2 |

Robustness: waiving any single check still leaves the gate unmet (best single-waiver 24); only
discarding cue + responsiveness + phase together reaches 41.

Descriptive retrieval on the 10 valid instances supports **no ordering of arms** — every Wilson
interval is 28–55 points wide and all arms overlap; correct-vs-rewired does not even keep a
consistent sign across k. The 11 OpenReview-namespace targets are excluded from all graph arms
(they have no nodes in the citation graph) rather than scored as misses.

Two defects recorded, not fixed: the submission's own arXiv preprint is never excluded (BM25 rank-1
in 16/53), and no graph-side leakage audit is possible for OpenReview sources.

---

# Part 9 — Protocol constants (for reproduction)

Tool budget 8 calls · observation budget 6,000 tokens · 10 candidates per call · 1 format-repair
turn · 1 forced-answer turn on budget exhaustion · context guard 13,000 tokens.
Graph expansion: Adamic–Adar from anchor, ≤2 hops (CSC) / ≤3 hops with per-node cap 25 (LSE),
one-hop bonus 2.0. Hybrid fusion: RRF k=60. K=50 retrieved candidates.
Seeds 20260925 throughout. Bootstrap 10,000 resamples, unit = instance; paired sign-flip
permutation 10,000. Holm within declared confirmatory families only, per model, never pooled.
Checkpoints: Qwen2.5-7B-Instruct, Qwen3-8B, Qwen2.5-32B-Instruct **GPTQ Int4** (only quantised one).
Greedy decoding with a transcript-keyed response cache.

Frozen IDs: CSC fixed `063b66b0ca51897f` (n=5000) · CSC agent `2849d46ced72ed93` (n=150) ·
LSE fixed `5b7eadacd40f892a` (n=1200) · LSE agent `23429f6159c50b82` (n=150) ·
Task-A `ec357bef29eece81` (n=200) · scaling `0d6aeb4568262ea4` (n=50).
Graphs: correct `c33f0afd190858f3` · rewired `0232ea9a6a878502` · shared degree histogram
`1ee5994e832b330b`.

```bash
cd $PROJECT_ROOT
RUN=results/graph_autoresearch/graphauto_20260925_035919_027d9a
python -m tasks.agentic_autoresearch.controlled_aggregate --root $RUN/controlled_task_a
python -m tasks.agentic_autoresearch.ebc_aggregate       --root $RUN/evidence_bundle_completion/agent_results
python -m tasks.agentic_autoresearch.litexp_aggregate    --root $RUN/literature_set_expansion/agent_results
python -m tasks.agentic_autoresearch.appendix_tables     --root $RUN
python -m tasks.agentic_autoresearch.graph_autoresearch_report --root $RUN
```

---

# Part 10 — Missing and incomplete, explicitly

| Item | Status | What would be needed |
|---|---|---|
| **Anytime quality curves (Task-A)** | **failed to collect** | models emitted `current_top5` in only **87 of 4,655** snapshots (1.9%). Reported missing, never imputed. Fix: make the running top-5 a parser-enforced required field. The anytime *discovery* curve is valid (computed from observations, not model output). |
| MAP, nDCG for CSC | not computed | cheap CPU re-run of `ebc_screen`; the set-valued screen used set recall / set F1 |
| Semantic-retrieval arm for LSE | **does not exist** | a paper-level dense index; none in the repository |
| Per-episode latency for LSE | present, not aggregated | `latency_ms` is in the episode rows; `litexp_aggregate` emits no mean |
| Identical-answer rate across perturbations | not logged | recomputable CPU-only: join correct/rewired on `instance_id`, compare `ranked` |
| Venue diversity for LSE | **not computable** | corpus has no venue field; (year, category) clusters used instead |
| Qwen2.5-32B on LSE | **stopped, not failed** | 91/150 on `correct`, 0/150 on `rewired` — one side of a pair. Quarantined in `literature_set_expansion/agent_results/incomplete_32b/`, excluded from every number here |
| Canonical paper LaTeX source | **not in this repository** | only `ResearchArcade.pdf` (2026-09-22) exists, containing Tables 1–3 and no mention of these workflows. Appendix delivered standalone; no compile possible (`pdflatex` absent) |

---

# Part 11 — Claim table, scoped

| Claim | Scope | Label |
|---|---|---|
| Correct connectivity improves candidate discovery | fixed retrieval, n=5000 (CSC) / n=909 (LSE) | **Supported** |
| Correct connectivity helps agents on set recovery | CSC n=150 ×3 models; LSE n=150 ×2 models | **Supported** |
| Correct connectivity helps end-to-end Top-1 | Task-A n=200, 2 models | **Not supported** (genuine negative at 98% uptake) |
| Small agents spontaneously use graph tools | natural interface, 3 scales | **Not supported** (0 calls / 950 episodes) |
| Sequential interaction beats a matched pool | all workflows | **Not supported** |
| Compiled hybrid beats raw traversal | LSE | **Supported** (agent −0.142 below fixed list) |
| Relation types add candidate-discovery value | prior round | not reopened |
| A2 supports confirmatory claims | 10/64 valid | **No** — appendix/external-validity only |

Full report: `graph_autoresearch_canonical_report.md` · appendix package: `appendix/`
