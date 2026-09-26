# AutoResearch appendix package — audit

Prepared 2026-09-26 from the frozen artifacts of run `graphauto_20260925_035919_027d9a`
(commit `2631ae6d3200`). **No new inference, agent evaluation, training, or GPU job was run.**
All aggregation was CPU-only and read existing episode rows.

Workflow naming: the paper's **Citation Set Completion (CSC)** is `evidence_bundle_completion`
on disk; **Literature Set Expansion (LSE)** is `literature_set_expansion`.

---

## 1. Main-text reproduction: 31/31 values match exactly

Machine-readable: `appendix/maintext_verification.json`. Zero discrepancies; no diagnosis needed.

| Group | Checked | Result |
|---|---|---|
| CSC fixed retrieval R@20 (BM25, rewired hybrid, correct hybrid, diff, CI) | 6 | all match |
| LSE fixed retrieval R@50 (BM25, rewired hybrid, correct hybrid, diff, CI) | 6 | all match |
| CSC agent correct−rewired R@10, 3 models (diff + CI) | 9 | all match |
| LSE agent correct−rewired R@50, 2 models (diff + CI) | 6 | all match |
| CSC mechanism (graph-only target rate, episode rate; correct + rewired) | 4 | all match |

Two points on provenance of the main-text numbers:

- The **authoritative LSE subset is `screen`, n=909**, not `frozen_all` (n=1200). Both are in the
  artifact; the main-text values correspond to `screen`. The `frozen_all` values differ slightly
  (+0.2384 vs BM25, +0.1790 vs rewired) and should not be mixed in.
- The **CSC values come from the `general` slice, n=5000**, not from the exploratory **GraphHard**
  slice (n=3494). GraphHard is kept out of the canonical table entirely, as required.

Reproduce:

```bash
cd $PROJECT_ROOT
RUN=results/graph_autoresearch/graphauto_20260925_035919_027d9a
python -m tasks.agentic_autoresearch.appendix_tables --root $RUN
```

---

## 2. Artifact manifest

Paths relative to `results/graph_autoresearch/graphauto_20260925_035919_027d9a/`.

| Appendix table | Source artifact | Generating script | CSV behind it |
|---|---|---|---|
| Construction funnel | `*/construction_manifest.json` | `ebc_build`, `literature_set_expansion` | `appendix/tables/t1_data_funnel.csv`, `t1b_construction_detail.csv` |
| Complete fixed retrieval | `*/retrieval_results.csv` | `ebc_screen`, `literature_set_expansion` | `t2_fixed_retrieval.csv` |
| Fixed-retrieval contrasts | `*/retrieval_comparisons.csv` | same | `t2b_fixed_retrieval_comparisons.csv` |
| Absolute agent results | `*/agent_results/metrics_by_condition.csv` | `ebc_aggregate`, `litexp_aggregate` | `t3_agent_absolute.csv` |
| Agent contrasts | `*/agent_results/paired_comparisons.csv` | same | `t3b_agent_comparisons.csv` |
| Access + tool use | `candidate_funnel.csv`, `tool_utilization.csv`, `graph_only_discovery.csv` | same | `t4_access_and_tools.csv` |
| Efficiency + faithfulness | `faithfulness.csv`, `tool_utilization.csv` | same | `t5_efficiency_faithfulness.csv` |
| Failure taxonomy | `agent_results/failure_taxonomy.json` | same | `t6_failure_taxonomy.csv` |
| Matched traces | `agent_results/episodes/qwen3-8b/*.jsonl` | `appendix_tables` | `appendix/trace_pair.json` |

### Frozen IDs and hashes

| Split | n | SHA-256 (prefix) |
|---|---|---|
| CSC fixed retrieval (general) | 5000 | `063b66b0ca51897f` |
| CSC agent | 150 | `2849d46ced72ed93` |
| LSE fixed retrieval | 1200 | `5b7eadacd40f892a` |
| LSE agent | 150 | `23429f6159c50b82` |
| Task-A controlled (prior round) | 200 | `ec357bef29eece81` |

Graph variants: correct `c33f0afd190858f3`, rewired `0232ea9a6a878502`, shared degree histogram
`1ee5994e832b330b` (degree preservation verified after the identity merge).

### Protocol constants recovered

Tool budget 8 calls; observation budget 6000 tokens; 10 candidates per call; 1 format-repair turn;
1 forced-answer turn on budget exhaustion; context guard 13000 tokens. Graph expansion:
Adamic–Adar from anchor, ≤2 hops (CSC) / ≤3 hops with per-node cap 25 (LSE), one-hop bonus 2.0.
Hybrid fusion: RRF k=60. K=50 retrieved candidates. Seeds: 20260925 (construction, bootstrap,
permutation). Bootstrap 10,000 resamples, unit = instance; paired sign-flip permutation 10,000.
Holm applied within declared confirmatory families only, per model, never pooled.
Checkpoints: Qwen2.5-7B-Instruct, Qwen3-8B, Qwen2.5-32B-Instruct **GPTQ Int4** (only quantised one).

---

## 3. Discrepancies found

**None between the artifacts and the main text.** One error was introduced and caught during *my
own* drafting, recorded for transparency:

| Where | Error | Resolution |
|---|---|---|
| `appendix_autoresearch.tex`, efficiency table | I first typed observation tokens 2731/1614/1572/2434 and flat-agent tool errors 12 / parser validity 0.985 | Corrected to the CSV values 2883/1856/1822/2257 and 7 / 0.965 before delivery |

Every other hand-transcribed value in the LaTeX was re-checked cell-by-cell against
`appendix/tables/*.csv`.

---

## 4. Missing quantities and exactly what would be needed

Machine-readable: `appendix/missing.json`.

| Quantity | Status | Artifact that would be required |
|---|---|---|
| MAP for CSC | not computed | re-run of `ebc_screen` aggregation (CPU, cheap); the set-valued screen used set recall / set F1 instead |
| nDCG for CSC | not computed | same |
| Semantic-retrieval arm for LSE | **does not exist** | a paper-level title+abstract dense index; none exists in the repository (consistent with the README's "no dense index" deviation) |
| Per-episode latency for LSE | present but not aggregated | `literature_set_expansion/agent_results/episodes/*.jsonl` carries `latency_ms`; `litexp_aggregate` does not emit a mean |
| Identical-answer rate across graph perturbations | not logged | recomputable CPU-only from episode rows: join correct and rewired on `instance_id`, compare `ranked` |
| Per-stage temporal/context drop counts for LSE | not recorded as stages | LSE records exclusions as a combined rule histogram (`funnel.exclusion_histogram`), reported there instead |
| Venue diversity for LSE | **not computable** | the corpus has no venue field; (year, primary category) clusters were used instead |
| Qwen2.5-32B on LSE | **stopped, not failed** | 91/150 on `correct` and 0/150 on `rewired` — one side of a paired comparison. Quarantined in `literature_set_expansion/agent_results/incomplete_32b/`, excluded from every reported number, per the rule against reducing one side of a pair |

---

## 5. Claims: fully supported vs requiring qualification

### Fully supported

1. **Correct citation connectivity improves candidate discovery** (fixed retrieval). CSC +0.291 R@20
   [+0.279,+0.303]; LSE +0.175 R@50 [+0.164,+0.186]; both at the permutation floor. The
   degree-matched-anchor control (0.047 vs 0.547 R@20) rules out degree and hub-proximity artifacts.
2. **Correct connectivity helps interactive agents**, replicated: CSC on 3 models
   (+0.0889 / +0.1017 / +0.1028 R@10, Holm p ≤ 0.0012) and LSE on 2 models (+0.0660 / +0.0644 R@50,
   Holm p ≤ 0.0006), all at 94–98% traversal uptake.
3. **The graph acts on discovery.** CSC Qwen3-8B: target ever observed 0.542 (correct) vs 0.265
   (rewired); "target never retrieved" failures drop 61.3% → 31.3%; graph-only targets 31.2% vs 8.1%.
4. **Evidence faithfulness is perfect** (1.000 every condition) and every stored graph path
   re-verified in its own pre-cutoff graph.
5. **Zero temporal / source-edge leakage** after the identity fix, on full-coverage audits.

### Requiring qualification

1. **Agents do not beat matched replays.** No agent beats its own matched-pool replay after Holm
   correction; several estimates are negative. The appendix states this explicitly and makes no
   agentic-interaction claim. Any main-text sentence implying agentic interaction *per se* adds value
   must be softened.
2. **A compiled retrieval interface beats agentic traversal on LSE.** Fixed correct hybrid R@50
   = 0.423 vs best agent 0.281; the compact-hybrid agent is 0.142 *below* the fixed list
   (Holm p = 0.0006). Raw traversal is not the best way to use the graph.
3. **Spontaneous graph-tool use is not supported.** Under a natural action interface, 0 tool calls in
   800 episodes (7B/8B) and 0 in 150 (32B); all natural conditions returned identical scores because
   they degenerate to one-shot. The mandatory minimum research phase is a **disclosed intervention**
   and must be described as such wherever these numbers appear.
4. **LSE absolute recall is a floor, not a measurement.** Mean unresolved-reference rate 0.68 on the
   frozen set; a bibliography is selective weak supervision. Only paired differences are interpretable.
5. **Qwen2.5-7B on LSE answered only 54–64% of episodes.** Absolute values are depressed. The paired
   contrast survives because validity is *lower* in the correct condition, so the effect is not an
   artifact of differential answering — but Qwen3-8B (98.7% answered) is the cleaner evidence.
6. **CSC GraphHard is an intervention slice, not representative** (69.9% of the construction pool by
   design). Kept out of the canonical table.

---

## 6. Integration status — BLOCKED, with reason

**The canonical ResearchArcade LaTeX source is not present in this repository.** Searched
`$PROJECT_ROOT`, `<withheld: unrelated local directory>`, `<withheld: unrelated local directory>` and
`$DATA_ROOT/openreview-papers`: no `.tex` file for this paper exists. The only
artifact is `ResearchArcade.pdf` (2026-09-22), whose text contains **Tables 1–3 only** and zero
mentions of "Citation Set Completion", "Literature Set Expansion" or "rewired".

Consequences:

- **Step 4 (insert into the canonical paper) could not be performed.** The appendix is delivered as
  a standalone, drop-in `\section{}` file.
- **Main-text Tables 4 and 5 could not be inspected.** They are not in the available PDF, so they
  live in a newer source revision that is not on this machine. I verified their *numbers* against
  the artifacts (all 31 match) but could not check their *formatting* or cross-references.
- **No compilation was attempted**, so "no undefined references / no overflow" is unverified.

Style was matched to the available PDF: `booktabs`, no vertical rules, **3-decimal precision**
(as in Table 2), concise captions, task-grouped subtables. To integrate: `\input` the file inside the
appendix, ensure `\usepackage{booktabs}`, and compile. The only external dependency is `booktabs`.

---

## 7. Files created

| Path (under `results/graph_autoresearch/graphauto_20260925_035919_027d9a/`) | Content |
|---|---|
| `appendix/appendix_autoresearch.tex` | the appendix section, 10 subsections, 9 tables |
| `appendix/AUTORESEARCH_APPENDIX_AUDIT.md` | this file |
| `appendix/maintext_verification.json` | 31 main-text checks, machine-readable |
| `appendix/missing.json` | unrecoverable quantities + what each needs |
| `appendix/trace_pair.json` | matched correct / rewired / flat traces for `EBC-3452627` |
| `appendix/tables/t1_data_funnel.csv` … `t6_failure_taxonomy.csv` | 9 CSVs behind every table |

New code: `tasks/agentic_autoresearch/appendix_tables.py` (CPU-only aggregation). No existing
module was modified; the previous canonical run and the six-task benchmark results were not touched.

---

## 8. The illustrative trace

Instance `EBC-3452627`, 2 targets, anchor `2009.02252`. All conditions get byte-identical prompts.

| Condition | Actions emitted | Outcome | R@10 |
|---|---|---|---|
| Correct traversal | `search("retrieval augmented generation benchmarks")`, `traverse(2009.02252)` | both targets recovered; `1705.03551` first seen at call 2, 1 hop from anchor, path verified | **1.000** |
| Rewired traversal | **identical two actions, identical query** | neither target returned | 0.000 |
| Flat search | `search` ×8 — the same query repeated 7× after the first | neither target returned | 0.000 |

The correct and rewired agents behave *identically*; only the adjacency behind `traverse` differs.
This is the cleanest available demonstration that the gain comes from topology rather than policy.
The flat agent's degenerate query repetition also explains its higher observation-token cost
(2883 vs 1822).
