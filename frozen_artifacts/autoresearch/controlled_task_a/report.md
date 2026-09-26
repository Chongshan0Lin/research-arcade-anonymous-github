# Controlled-interaction Task-A (`controlled_interaction_v1`)

**Explicitly post-hoc.** This protocol was designed after observing the natural-action zero-tool
result. It is not preregistered, and every comparison against the natural run is labelled as such.

**Protocol disclosure (runbook §1.3).** Each condition enforces a *mandatory minimum research
phase* before an answer is accepted: flat → ≥1 `search`; correct/rewired graph → ≥1 `search` and
≥1 `traverse`; compact hybrid → ≥1 `hybrid_search`. A premature `final` is refused with a
deterministic message and costs no tool call. Agents receive **no** prepopulated candidate list.
This is part of the protocol, not a model behaviour, and must be stated wherever these numbers are.

Frozen set: the **same 200 ids** as the natural-action run (`ec357bef29eece81`), so the two
interfaces are directly comparable. 4,800 rows, 0 duplicate keys, 0 temporal-leakage violations,
0 blocked calls, 0 out-of-interface executions.

## Tool uptake — the repair worked

| model | condition | mean calls | traverse uptake | hybrid uptake | forced-final |
|---|---|---|---|---|---|
| qwen2.5-7b | flat_controlled | 5.01 | – | – | 60/200 |
| qwen2.5-7b | correct_graph | 5.42 | **98%** | – | 51/200 |
| qwen2.5-7b | rewired_graph | 5.89 | **98%** | – | 84/200 |
| qwen2.5-7b | compact_hybrid | 5.26 | – | **99%** | 71/200 |
| qwen3-8b | flat_controlled | 5.14 | – | – | 75/200 |
| qwen3-8b | correct_graph | 3.86 | **98%** | – | 15/200 |
| qwen3-8b | rewired_graph | 4.08 | **98%** | – | 19/200 |
| qwen3-8b | compact_hybrid | 5.54 | – | **100%** | 84/200 |

Natural interface: **0.00 calls, 0% uptake**. The uptake gate (§9.1) is satisfied, so the
correct-vs-rewired comparison is *informative* rather than a failed-uptake check.

## Effectiveness (Top-1, n=200 per cell)

| condition | qwen2.5-7b | qwen3-8b |
|---|---|---|
| bm25_one_shot_ctrl | 0.150 | 0.175 |
| correct_hybrid_one_shot_ctrl | 0.185 | 0.205 |
| flat_controlled_agent | 0.185 | 0.210 |
| rewired_graph_controlled_agent | 0.155 | 0.185 |
| correct_graph_controlled_agent | 0.155 | 0.220 |
| compact_hybrid_controlled_agent | 0.240 | 0.230 |

One-shot controls were **re-run** under the ctrl-v1 prompt and parser rather than reused from the
natural run: §2.1 permits reuse only when prompt, parser and answer semantics are identical, and
ctrl-v1 answers are `{"final":[...]}` not `{"tool":"submit",...}`.

## Primary family (Holm-adjusted within model, Top-1)

**No comparison survives Holm correction on either model.** Raw effects and CIs:

| model | comparison | diff | 95% CI | p_raw | p_Holm |
|---|---|---|---|---|---|
| qwen2.5-7b | correct > rewired | **+0.000** | [−0.035, +0.035] | 1.000 | 1.000 |
| qwen3-8b | correct > rewired | +0.035 | [+0.010, +0.065] | 0.038 | 0.368 |
| qwen2.5-7b | compact_hybrid > hybrid_one_shot | +0.055 | [−0.005, +0.115] | 0.123 | 1.000 |
| qwen3-8b | compact_hybrid > hybrid_one_shot | +0.025 | [−0.040, +0.085] | 0.550 | 1.000 |
| qwen2.5-7b | compact_hybrid > its replay | +0.025 | [−0.025, +0.075] | 0.420 | 1.000 |
| qwen3-8b | compact_hybrid > its replay | −0.005 | [−0.050, +0.040] | 1.000 | 1.000 |
| qwen2.5-7b | controlled > natural (compact_hybrid) | +0.070 | [+0.010, +0.130] | 0.033 | 0.333 |
| qwen3-8b | controlled > natural (compact_hybrid) | +0.070 | [+0.010, +0.130] | 0.037 | 0.368 |
| qwen3-8b | controlled > natural (correct_graph) | +0.060 | [+0.010, +0.110] | 0.037 | 0.368 |

## The central finding: the intervention works at discovery and dies at selection

Decomposing correct-vs-rewired by funnel stage (paired, n=200, 10,000 resamples):

| stage | qwen2.5-7b | p | qwen3-8b | p |
|---|---|---|---|---|
| **gold ever observed** (discovery) | 0.480 vs 0.355, **+0.125** [+0.070, +0.185] | **0.0001** | 0.425 vs 0.345, **+0.080** [+0.035, +0.125] | **0.0017** |
| **gold selected** (in answer) | 0.260 vs 0.235, +0.025 [−0.015, +0.065] | 0.329 | 0.300 vs 0.280, +0.020 [−0.020, +0.060] | 0.454 |
| **Top-1** | 0.155 vs 0.155, +0.000 | 1.000 | 0.220 vs 0.185, +0.035 | 0.038 |

With traversal uptake at 98%, correct connectivity surfaces the gold paper **significantly more
often than a degree-preserving rewired graph on both models** — a large, replicated, causally
controlled effect. That advantage then **does not propagate to the answer**: the selection and
Top-1 differences are null.

This reconciles the two halves of this round. The EBC retrieval screen found a very large graph
effect (+0.291 R@20 correct vs rewired, n=5,000). Here the same kind of effect appears in the
agent's *observation stream* and is then lost at the selection step. The bottleneck for these
7-8B agents is not candidate discovery; it is choosing correctly from candidates they have already
seen.

## Claim decisions (runbook §2.5)

| claim | qwen2.5-7b | qwen3-8b |
|---|---|---|
| Correct connectivity helps when traversal is controlled (Top-1) | **Not supported** | **Not supported** |
| Sequential interaction beats a matched observed pool | **Not supported** | **Not supported** |
| Compiled hybrid beats raw traversal | **Supported** (+0.085 [+0.025,+0.145] p=0.009, secondary/unadjusted) | **Not supported** (+0.010, p=0.87) |

Traversal uptake was 97.5% / 98.0%, so these are genuine negative results under the controlled
protocol, **not** failed-uptake checks. Scoped: *end-to-end candidate-to-answer utility on the
canonical n=200 subset.* They do not contradict the full-scale structural retrieval evidence, and
the discovery-stage result above is positive and significant.

## Limitations, stated plainly

- **Anytime quality curves failed to collect.** §1.4 required a parseable running top-5 each turn.
  Models emitted `current_top5` in only **87 of 4,655** snapshots (1.9%), so anytime Top-1/MRR are
  not measurable and are reported as missing rather than imputed. The anytime *discovery* curve
  (gold-observed rate by call count) is computed from observations, not model output, and is valid.
  Fix for a future round: make the running top-5 a required field enforced by the action parser.
- The mandatory research phase is an intervention. These agents did not choose to use tools.
- 15-42% of episodes ended via the forced-final turn (budget exhausted without a voluntary answer),
  applied identically in every condition.
- Two models, one seed, one task. No pooling across models for any primary claim.
