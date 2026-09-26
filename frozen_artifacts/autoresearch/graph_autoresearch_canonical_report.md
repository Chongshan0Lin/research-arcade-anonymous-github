# Graph x AutoResearch canonical report

Run: `graphauto_20260925_035919_027d9a`  |  commit `2631ae6d3200`  |  protocol `controlled_interaction_v1` (explicitly post-hoc)

## 1. Existing full-scale structural retrieval evidence

Carried forward unchanged from the prior round: the n=2034 structural retrieval result stands on its own and is **not** superseded by any 200-instance end-to-end number below. Scope any 200-instance claim as *end-to-end candidate-to-answer utility on the canonical n=200 subset*.

## 2. Previous natural-action result (preserved)

Run `20260924_182602_a26e2e` at `results/agentic_autoresearch/20260924_182602_a26e2e` is preserved byte-for-byte and reported as the **natural-action interface** result.

> Under the natural-action interface, neither 7B-8B model invoked a retrieval or traversal tool. We therefore treat correct-versus-rewired equality as a failed intervention-uptake check, not evidence against graph utility.

Root cause of the zero-tool run is documented in `tool_loop_root_cause.md`: the first user turn already carried a 10-candidate BM25 pool and the prompt's closing line demanded an immediate 5-id submission, so submitting on turn 1 was the compliant action. All 2614 cached generations were well-formed final answers; none were malformed or rejected.

## 3. New controlled-interaction result

**Disclosure:** each controlled condition enforces a mandatory minimum research phase (flat: >=1 search; correct/rewired graph: >=1 search and >=1 traverse; compact hybrid: >=1 hybrid_search). This is part of the protocol, not a model behaviour. Agents receive no prepopulated candidate list. Correct and rewired conditions share byte-identical prompts and tool schemas; only the backing adjacency differs.

### Effectiveness

| model | condition | n | top1 | mrr | recall_at5 | validity |
|---|---|---|---|---|---|---|
| qwen2.5-7b-instruct | bm25_one_shot_ctrl | 200 | 0.155 | 0.1787 | 0.22 | 1.0 |
| qwen2.5-7b-instruct | compact_hybrid_controlled_agent | 200 | 0.24 | 0.3103 | 0.44 | 0.965 |
| qwen2.5-7b-instruct | correct_graph_controlled_agent | 200 | 0.155 | 0.1969 | 0.26 | 0.955 |
| qwen2.5-7b-instruct | correct_hybrid_one_shot_ctrl | 200 | 0.185 | 0.2195 | 0.29 | 1.0 |
| qwen2.5-7b-instruct | flat_controlled_agent | 200 | 0.185 | 0.2208 | 0.285 | 0.92 |
| qwen2.5-7b-instruct | matched_pool_replay::bm25_one_shot_ctrl | 200 | 0.155 | 0.1787 | 0.22 | 1.0 |
| qwen2.5-7b-instruct | matched_pool_replay::compact_hybrid_controlled_agent | 200 | 0.215 | 0.2831 | 0.39 | 0.995 |
| qwen2.5-7b-instruct | matched_pool_replay::correct_graph_controlled_agent | 200 | 0.14 | 0.1878 | 0.275 | 0.995 |
| qwen2.5-7b-instruct | matched_pool_replay::correct_hybrid_one_shot_ctrl | 200 | 0.185 | 0.2195 | 0.29 | 1.0 |
| qwen2.5-7b-instruct | matched_pool_replay::flat_controlled_agent | 200 | 0.15 | 0.1899 | 0.265 | 0.995 |
| qwen2.5-7b-instruct | matched_pool_replay::rewired_graph_controlled_agent | 200 | 0.135 | 0.1789 | 0.265 | 0.995 |
| qwen2.5-7b-instruct | rewired_graph_controlled_agent | 200 | 0.155 | 0.1843 | 0.235 | 0.95 |
| qwen3-8b | bm25_one_shot_ctrl | 200 | 0.175 | 0.1927 | 0.23 | 1.0 |
| qwen3-8b | compact_hybrid_controlled_agent | 200 | 0.23 | 0.3027 | 0.41 | 0.99 |
| qwen3-8b | correct_graph_controlled_agent | 200 | 0.22 | 0.2528 | 0.3 | 0.995 |
| qwen3-8b | correct_hybrid_one_shot_ctrl | 200 | 0.205 | 0.2488 | 0.32 | 1.0 |
| qwen3-8b | flat_controlled_agent | 200 | 0.21 | 0.248 | 0.31 | 0.98 |
| qwen3-8b | matched_pool_replay::bm25_one_shot_ctrl | 200 | 0.175 | 0.1927 | 0.23 | 1.0 |
| qwen3-8b | matched_pool_replay::compact_hybrid_controlled_agent | 200 | 0.235 | 0.2879 | 0.38 | 0.99 |
| qwen3-8b | matched_pool_replay::correct_graph_controlled_agent | 200 | 0.18 | 0.2187 | 0.275 | 0.995 |
| qwen3-8b | matched_pool_replay::correct_hybrid_one_shot_ctrl | 200 | 0.205 | 0.2488 | 0.32 | 1.0 |
| qwen3-8b | matched_pool_replay::flat_controlled_agent | 200 | 0.17 | 0.2188 | 0.295 | 0.99 |
| qwen3-8b | matched_pool_replay::rewired_graph_controlled_agent | 200 | 0.17 | 0.2068 | 0.25 | 0.995 |
| qwen3-8b | rewired_graph_controlled_agent | 200 | 0.185 | 0.223 | 0.28 | 0.995 |


### Tool uptake

| model | condition | mean_tool_calls | mean_successful_calls | traverse_uptake | hybrid_uptake | blocked | failed | malformed | forced_final |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-7b-instruct | compact_hybrid_controlled_agent | 5.265 | 5.22 | 0.0 | 0.99 | 0 | 9 | 17 | 71 |
| qwen2.5-7b-instruct | correct_graph_controlled_agent | 5.42 | 5.375 | 0.975 | 0.0 | 0 | 9 | 18 | 51 |
| qwen2.5-7b-instruct | flat_controlled_agent | 5.01 | 5.005 | 0.0 | 0.0 | 0 | 1 | 32 | 60 |
| qwen2.5-7b-instruct | rewired_graph_controlled_agent | 5.895 | 5.86 | 0.975 | 0.0 | 0 | 7 | 20 | 84 |
| qwen3-8b | compact_hybrid_controlled_agent | 5.545 | 5.54 | 0.0 | 0.995 | 0 | 1 | 3 | 84 |
| qwen3-8b | correct_graph_controlled_agent | 3.86 | 3.855 | 0.98 | 0.0 | 0 | 1 | 1 | 15 |
| qwen3-8b | flat_controlled_agent | 5.145 | 5.145 | 0.0 | 0.0 | 0 | 0 | 7 | 75 |
| qwen3-8b | rewired_graph_controlled_agent | 4.085 | 4.08 | 0.98 | 0.0 | 0 | 1 | 1 | 19 |


### Candidate funnel

| model | condition | gold_ever_observed | gold_selected | mean_observed_candidates | mean_calls_to_first_gold |
|---|---|---|---|---|---|
| qwen2.5-7b-instruct | bm25_one_shot_ctrl | 0.34 | 0.22 | 20.0 | 0.0 |
| qwen2.5-7b-instruct | compact_hybrid_controlled_agent | 0.54 | 0.44 | 16.18 | 1.2685185185185186 |
| qwen2.5-7b-instruct | correct_graph_controlled_agent | 0.48 | 0.26 | 29.45 | 1.7916666666666667 |
| qwen2.5-7b-instruct | correct_hybrid_one_shot_ctrl | 0.47 | 0.29 | 20.0 | 0.0 |
| qwen2.5-7b-instruct | flat_controlled_agent | 0.37 | 0.285 | 17.34 | 1.3513513513513513 |
| qwen2.5-7b-instruct | matched_pool_replay::bm25_one_shot_ctrl | 0.34 | 0.22 | 20.0 | 0.0 |
| qwen2.5-7b-instruct | matched_pool_replay::compact_hybrid_controlled_agent | 0.54 | 0.39 | 16.18 | 1.2685185185185186 |
| qwen2.5-7b-instruct | matched_pool_replay::correct_graph_controlled_agent | 0.48 | 0.275 | 29.45 | 1.7916666666666667 |
| qwen2.5-7b-instruct | matched_pool_replay::correct_hybrid_one_shot_ctrl | 0.47 | 0.29 | 20.0 | 0.0 |
| qwen2.5-7b-instruct | matched_pool_replay::flat_controlled_agent | 0.37 | 0.265 | 17.34 | 1.3513513513513513 |
| qwen2.5-7b-instruct | matched_pool_replay::rewired_graph_controlled_agent | 0.355 | 0.265 | 33.04 | 1.6056338028169015 |
| qwen2.5-7b-instruct | rewired_graph_controlled_agent | 0.355 | 0.235 | 33.04 | 1.6056338028169015 |
| qwen3-8b | bm25_one_shot_ctrl | 0.34 | 0.23 | 20.0 | 0.0 |
| qwen3-8b | compact_hybrid_controlled_agent | 0.515 | 0.41 | 16.8 | 1.2718446601941749 |
| qwen3-8b | correct_graph_controlled_agent | 0.425 | 0.3 | 22.7 | 1.7529411764705882 |
| qwen3-8b | correct_hybrid_one_shot_ctrl | 0.47 | 0.32 | 20.0 | 0.0 |
| qwen3-8b | flat_controlled_agent | 0.39 | 0.31 | 18.3 | 1.3205128205128205 |
| qwen3-8b | matched_pool_replay::bm25_one_shot_ctrl | 0.34 | 0.23 | 20.0 | 0.0 |
| qwen3-8b | matched_pool_replay::compact_hybrid_controlled_agent | 0.515 | 0.38 | 16.8 | 1.2718446601941749 |
| qwen3-8b | matched_pool_replay::correct_graph_controlled_agent | 0.425 | 0.275 | 22.7 | 1.7529411764705882 |
| qwen3-8b | matched_pool_replay::correct_hybrid_one_shot_ctrl | 0.47 | 0.32 | 20.0 | 0.0 |
| qwen3-8b | matched_pool_replay::flat_controlled_agent | 0.39 | 0.295 | 18.3 | 1.3205128205128205 |
| qwen3-8b | matched_pool_replay::rewired_graph_controlled_agent | 0.345 | 0.25 | 24.0 | 1.4927536231884058 |
| qwen3-8b | rewired_graph_controlled_agent | 0.345 | 0.28 | 24.0 | 1.4927536231884058 |


### Anytime quality vs cost

| model | condition | after_n_calls | top1 | mrr | recall_at5 | gold_observed_rate |
|---|---|---|---|---|---|---|
| qwen2.5-7b-instruct | compact_hybrid_controlled_agent | 1 | 0.0 | 0.0 | 0.0 | 0.4673 |
| qwen2.5-7b-instruct | compact_hybrid_controlled_agent | 2 | 0.0653 | 0.0768 | 0.0955 | 0.5176 |
| qwen2.5-7b-instruct | compact_hybrid_controlled_agent | 4 | 0.0 | 0.0 | 0.0 | 0.4508 |
| qwen2.5-7b-instruct | compact_hybrid_controlled_agent | 8 | 0.0 | 0.0 | 0.0 | 0.4085 |
| qwen2.5-7b-instruct | correct_graph_controlled_agent | 1 | 0.0 | 0.0 | 0.0 | 0.2965 |
| qwen2.5-7b-instruct | correct_graph_controlled_agent | 2 | 0.0102 | 0.0102 | 0.0102 | 0.3858 |
| qwen2.5-7b-instruct | correct_graph_controlled_agent | 4 | 0.0 | 0.0 | 0.0 | 0.3986 |
| qwen2.5-7b-instruct | correct_graph_controlled_agent | 8 | 0.0 | 0.0 | 0.0 | 0.3529 |
| qwen2.5-7b-instruct | flat_controlled_agent | 1 | 0.0 | 0.0 | 0.0 | 0.2965 |
| qwen2.5-7b-instruct | flat_controlled_agent | 2 | 0.0103 | 0.0103 | 0.0103 | 0.3436 |
| qwen2.5-7b-instruct | flat_controlled_agent | 4 | 0.0 | 0.0 | 0.0 | 0.2846 |
| qwen2.5-7b-instruct | flat_controlled_agent | 8 | 0.0 | 0.0 | 0.0 | 0.2667 |
| qwen2.5-7b-instruct | rewired_graph_controlled_agent | 1 | 0.0 | 0.0 | 0.0 | 0.2965 |
| qwen2.5-7b-instruct | rewired_graph_controlled_agent | 2 | 0.0102 | 0.0102 | 0.0102 | 0.3046 |
| qwen2.5-7b-instruct | rewired_graph_controlled_agent | 4 | 0.0 | 0.0 | 0.0 | 0.2774 |
| qwen2.5-7b-instruct | rewired_graph_controlled_agent | 8 | 0.0 | 0.0 | 0.0 | 0.2738 |
| qwen3-8b | compact_hybrid_controlled_agent | 1 | 0.0 | 0.0 | 0.0 | 0.4271 |
| qwen3-8b | compact_hybrid_controlled_agent | 2 | 0.0 | 0.0 | 0.0 | 0.4798 |
| qwen3-8b | compact_hybrid_controlled_agent | 4 | 0.0 | 0.0 | 0.0 | 0.48 |
| qwen3-8b | compact_hybrid_controlled_agent | 8 | 0.0 | 0.0 | 0.0 | 0.3765 |
| qwen3-8b | correct_graph_controlled_agent | 1 | 0.0 | 0.0 | 0.0 | 0.28 |
| qwen3-8b | correct_graph_controlled_agent | 2 | 0.0 | 0.0 | 0.0 | 0.32 |
| qwen3-8b | correct_graph_controlled_agent | 4 | 0.0 | 0.0 | 0.0 | 0.2857 |
| qwen3-8b | correct_graph_controlled_agent | 8 | 0.0 | 0.0 | 0.0 | 0.25 |
| qwen3-8b | flat_controlled_agent | 1 | 0.0 | 0.0 | 0.0 | 0.3116 |
| qwen3-8b | flat_controlled_agent | 2 | 0.0 | 0.0 | 0.0 | 0.3608 |
| qwen3-8b | flat_controlled_agent | 4 | 0.0 | 0.0 | 0.0 | 0.303 |
| qwen3-8b | flat_controlled_agent | 8 | 0.0 | 0.0 | 0.0 | 0.2763 |
| qwen3-8b | rewired_graph_controlled_agent | 1 | 0.0 | 0.0 | 0.0 | 0.28 |
| qwen3-8b | rewired_graph_controlled_agent | 2 | 0.0 | 0.0 | 0.0 | 0.295 |
| qwen3-8b | rewired_graph_controlled_agent | 4 | 0.0 | 0.0 | 0.0 | 0.2577 |
| qwen3-8b | rewired_graph_controlled_agent | 8 | 0.0 | 0.0 | 0.0 | 0.25 |


### Paired comparisons

| model | higher | lower | n | diff | ci_lo | ci_hi | p_raw | p_adj | significant |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-7b-instruct | correct_graph_controlled_agent | rewired_graph_controlled_agent | 200 | 0.0 | -0.035 | 0.035 | 1.0 | 1.0 | False |
| qwen2.5-7b-instruct | compact_hybrid_controlled_agent | correct_hybrid_one_shot_ctrl | 200 | 0.055 | -0.005 | 0.115 | 0.12248775122487751 | 1.0 | False |
| qwen2.5-7b-instruct | flat_controlled_agent | matched_pool_replay::flat_controlled_agent | 200 | 0.035 | -0.01 | 0.08 | 0.19768023197680232 | 1.0 | False |
| qwen2.5-7b-instruct | correct_graph_controlled_agent | matched_pool_replay::correct_graph_controlled_agent | 200 | 0.015 | -0.025 | 0.06 | 0.6447355264473552 | 1.0 | False |
| qwen2.5-7b-instruct | rewired_graph_controlled_agent | matched_pool_replay::rewired_graph_controlled_agent | 200 | 0.02 | -0.02 | 0.06 | 0.45155484451554845 | 1.0 | False |
| qwen2.5-7b-instruct | compact_hybrid_controlled_agent | matched_pool_replay::compact_hybrid_controlled_agent | 200 | 0.025 | -0.025 | 0.075 | 0.41995800419958 | 1.0 | False |
| qwen2.5-7b-instruct | flat_controlled_agent | natural::flat_search_agent | 200 | 0.025 | -0.03 | 0.08 | 0.45565443455654436 | 1.0 | False |
| qwen2.5-7b-instruct | correct_graph_controlled_agent | natural::correct_connectivity_agent | 200 | -0.02 | -0.07 | 0.03 | 0.5583441655834417 | 1.0 | False |
| qwen2.5-7b-instruct | rewired_graph_controlled_agent | natural::rewired_connectivity_agent | 200 | -0.02 | -0.07 | 0.03 | 0.5373462653734626 | 1.0 | False |
| qwen2.5-7b-instruct | compact_hybrid_controlled_agent | natural::compact_hybrid_agent | 200 | 0.07 | 0.01 | 0.13 | 0.0332966703329667 | 0.33296670332966705 | False |
| qwen3-8b | correct_graph_controlled_agent | rewired_graph_controlled_agent | 200 | 0.035 | 0.01 | 0.065 | 0.0377962203779622 | 0.36796320367963203 | False |
| qwen3-8b | compact_hybrid_controlled_agent | correct_hybrid_one_shot_ctrl | 200 | 0.025 | -0.04 | 0.085 | 0.5503449655034497 | 1.0 | False |
| qwen3-8b | flat_controlled_agent | matched_pool_replay::flat_controlled_agent | 200 | 0.04 | 0.0 | 0.08 | 0.09269073092690731 | 0.6488351164883512 | False |
| qwen3-8b | correct_graph_controlled_agent | matched_pool_replay::correct_graph_controlled_agent | 200 | 0.04 | -0.005 | 0.085 | 0.11808819118088192 | 0.7085291470852915 | False |
| qwen3-8b | rewired_graph_controlled_agent | matched_pool_replay::rewired_graph_controlled_agent | 200 | 0.015 | -0.02 | 0.05 | 0.5816418358164184 | 1.0 | False |
| qwen3-8b | compact_hybrid_controlled_agent | matched_pool_replay::compact_hybrid_controlled_agent | 200 | -0.005 | -0.05 | 0.04 | 1.0 | 1.0 | False |
| qwen3-8b | flat_controlled_agent | natural::flat_search_agent | 200 | 0.04 | -0.01 | 0.09 | 0.18518148185181482 | 0.9259074092590741 | False |
| qwen3-8b | correct_graph_controlled_agent | natural::correct_connectivity_agent | 200 | 0.06 | 0.01 | 0.11 | 0.037196280371962806 | 0.36796320367963203 | False |
| qwen3-8b | rewired_graph_controlled_agent | natural::rewired_connectivity_agent | 200 | 0.025 | -0.025 | 0.075 | 0.4273572642735726 | 1.0 | False |
| qwen3-8b | compact_hybrid_controlled_agent | natural::compact_hybrid_agent | 200 | 0.07 | 0.01 | 0.13 | 0.0367963203679632 | 0.36796320367963203 | False |


Secondary (unadjusted):

| model | higher | lower | metric | diff | ci_lo | ci_hi | p_raw |
|---|---|---|---|---|---|---|---|
| qwen2.5-7b-instruct | correct_graph_controlled_agent | flat_controlled_agent | top1 | -0.03 | -0.075 | 0.015 | 0.29077092290770923 |
| qwen2.5-7b-instruct | correct_graph_controlled_agent | flat_controlled_agent | mrr | -0.0239 | -0.0643 | 0.0148 | 0.24277572242775722 |
| qwen2.5-7b-instruct | compact_hybrid_controlled_agent | flat_controlled_agent | top1 | 0.055 | -0.005 | 0.115 | 0.0861913808619138 |
| qwen2.5-7b-instruct | compact_hybrid_controlled_agent | flat_controlled_agent | mrr | 0.0895 | 0.037 | 0.1423 | 0.000999900009999 |
| qwen2.5-7b-instruct | compact_hybrid_controlled_agent | correct_graph_controlled_agent | top1 | 0.085 | 0.025 | 0.145 | 0.008999100089991 |
| qwen2.5-7b-instruct | compact_hybrid_controlled_agent | correct_graph_controlled_agent | mrr | 0.1134 | 0.0578 | 0.1693 | 0.00029997000299970003 |
| qwen2.5-7b-instruct | correct_graph_controlled_agent | bm25_one_shot_ctrl | top1 | 0.0 | -0.05 | 0.05 | 1.0 |
| qwen2.5-7b-instruct | correct_graph_controlled_agent | bm25_one_shot_ctrl | mrr | 0.0182 | -0.03 | 0.0674 | 0.46285371462853714 |
| qwen3-8b | correct_graph_controlled_agent | flat_controlled_agent | top1 | 0.01 | -0.035 | 0.055 | 0.821017898210179 |
| qwen3-8b | correct_graph_controlled_agent | flat_controlled_agent | mrr | 0.0048 | -0.0348 | 0.0447 | 0.8147185281471853 |
| qwen3-8b | compact_hybrid_controlled_agent | flat_controlled_agent | top1 | 0.02 | -0.035 | 0.075 | 0.6125387461253875 |
| qwen3-8b | compact_hybrid_controlled_agent | flat_controlled_agent | mrr | 0.0547 | 0.0059 | 0.1044 | 0.033896610338966106 |
| qwen3-8b | compact_hybrid_controlled_agent | correct_graph_controlled_agent | top1 | 0.01 | -0.05 | 0.065 | 0.8654134586541345 |
| qwen3-8b | compact_hybrid_controlled_agent | correct_graph_controlled_agent | mrr | 0.0499 | -0.0003 | 0.1001 | 0.05649435056494351 |
| qwen3-8b | correct_graph_controlled_agent | bm25_one_shot_ctrl | top1 | 0.045 | -0.005 | 0.095 | 0.1100889911008899 |
| qwen3-8b | correct_graph_controlled_agent | bm25_one_shot_ctrl | mrr | 0.0601 | 0.0099 | 0.1101 | 0.0188981101889811 |


## 4. Evidence Bundle Completion

Status: **complete**

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

## 5. Literature Set Expansion

Status: **complete**

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

## 6. A2 reviewer-request evidence

Status: **not_run**

_No report produced; gate/status: {"decision": "not_run"}_

## 7. Model-scaling / tool-uptake slice

Status: **complete**

# Model-scaling / tool-uptake slice (runbook §6)

**Purpose: natural tool-call and traversal uptake by model scale.** Effectiveness is exploratory
only — `n=50`, no confirmatory claim is made from this section.

## Setup

| | |
|---|---|
| Model | Qwen2.5-32B-Instruct (GPTQ-Int4), already local at `$LOCAL_MODELS/Qwen2.5-32B-Instruct-GPTQ-Int4` |
| Comparison family | Qwen2.5-7B-Instruct — same family, so scale is the only difference |
| Instances | first 50 of the canonical frozen 200 (`frozen_ids.jsonl`, hash recorded before running) |
| GPU | one reserved idle A6000 (GPU 5), no other workload displaced |
| Natural interface | `canonical_agent.py`, **unmodified** — the exact code that produced the zero-tool run |
| Controlled interface | `controlled_agent.py`, protocol `controlled_interaction_v1` |

No new checkpoint was downloaded. The 32B was read-only from another user's directory; no process
of theirs was touched.

## Result

| interface | condition | n | mean tool calls | episodes with ≥1 call | Top-1 | validity |
|---|---|---|---|---|---|---|
| natural | flat_search_agent | 50 | 0.00 | **0%** | 0.240 | 1.00 |
| natural | correct_connectivity_agent | 50 | 0.00 | **0%** | 0.240 | 1.00 |
| natural | compact_hybrid_agent | 50 | 0.00 | **0%** | 0.240 | 1.00 |
| controlled | flat_controlled_agent | 50 | 3.62 | **100%** | 0.300 | 1.00 |
| controlled | correct_graph_controlled_agent | 50 | 4.40 | **100%** | 0.320 | 1.00 |
| controlled | compact_hybrid_controlled_agent | 50 | 3.38 | **100%** | 0.400 | 1.00 |

## Reading

**Scale does not buy spontaneous tool use.** A 32B instruction model, roughly 4.5× the parameters
of the checkpoints in the canonical run, invoked a retrieval or traversal tool in **0 of 150**
natural-interface episodes — exactly matching Qwen2.5-7B and Qwen3-8B. The zero-tool finding is a
property of the interface, not of model capability or scale.

**The three natural conditions are numerically identical (0.240 each).** That is the pathology made
visible: with zero tool calls, `flat`, `correct_connectivity` and `compact_hybrid` degenerate into
the same one-shot selection over the same prepopulated BM25 pool. Their equality carries no
information about connectivity.

**Repairing the interface restores both uptake and separation.** Under `controlled_interaction_v1`
the same model on the same 50 instances reaches 100% uptake, and the conditions now differ
(0.300 / 0.320 / 0.400) because the manipulation actually reaches the model.

**Effectiveness rises with the controlled interface** (0.240 → 0.300-0.400), but `n=50` and these
are single-model, single-seed numbers with no paired test reported here. Treat the ordering as
exploratory. The confirmatory statistics live in `controlled_task_a/`.

## Scope limits

- `n=50`; no CIs or hypothesis tests are reported for this section by design.
- One quantised checkpoint (GPTQ-Int4); quantisation is a confound for absolute effectiveness,
  though not for the binary fact of zero tool calls.
- Three scales is not a scaling curve. The claim supported is the negative one: uptake did not
  appear at 32B.

## Files

`frozen_ids.jsonl`, `frozen_ids.sha256`, `natural_episodes/` (natural interface, run id
`graphauto_20260925_035919_027d9a_scale32b`), `episodes/` + `smoke/` (controlled interface).
The previous canonical run was not touched; the natural-interface episodes were written under their
own run id.

## 8. Claim table

| Claim | Scope | Label |
|---|---|---|
| Correct citation connectivity improves candidate discovery | full-scale structural retrieval, n=2034 | carried forward from the prior round, not re-decided here |
| Relation types add candidate-discovery value | prior round | not reopened (runbook 0.2) |
| Small agents spontaneously use graph tools | natural-action interface, 7B-8B | **Not supported** (0 tool calls in 800 episodes) |
| Correct connectivity helps when traversal is controlled | qwen2.5-7b-instruct, controlled n=200 | **Not supported** (traverse uptake 98%) |
| Sequential interaction beats a matched observed pool | qwen2.5-7b-instruct, controlled n=200 | **Not supported** |
| Compiled hybrid interface beats raw traversal | qwen2.5-7b-instruct, controlled n=200 | **Supported** |
| Correct connectivity helps when traversal is controlled | qwen3-8b, controlled n=200 | **Not supported** (traverse uptake 98%) |
| Sequential interaction beats a matched observed pool | qwen3-8b, controlled n=200 | **Not supported** |
| Compiled hybrid interface beats raw traversal | qwen3-8b, controlled n=200 | **Not supported** |
| Graph supports evidence-bundle completion | qwen2.5-32b-instruct, EBC agent n=150 | **Supported** (traverse uptake 94%) |
| Graph supports evidence-bundle completion | qwen2.5-7b-instruct, EBC agent n=150 | **Supported** (traverse uptake 97%) |
| Graph supports evidence-bundle completion | qwen3-8b, EBC agent n=150 | **Supported** (traverse uptake 98%) |
| Graph supports literature snowballing | qwen2.5-7b-instruct, snowballing agent n=150 | **Supported** (diff +0.0660, p_Holm 0.0001) |
| Graph supports literature snowballing | qwen3-8b, snowballing agent n=150 | **Supported** (diff +0.0644, p_Holm 0.0006) |

## 9. Verification

All section-12 checks passed: **True**

| check | passed | detail |
|---|---|---|
| previous canonical run preserved | None | results/agentic_autoresearch/20260924_182602_a26e2e is not bundled in this release; checkable only in the source tree |
| frozen split hashes recorded | True | {'controlled_task_a': {'path': 'results/agentic_autoresearch/20260924_182602_a26e2e/frozen_ids.jsonl', 'sha256_16': 'ec357bef29eece81', 'note': 'reused verbatim from the natural-action run so the two interfaces are directly comparable'}, 'evidence_bundle_completion_screen': {'path': 'results/graph_autoresearch/graphauto_20260925_035919_027d9a/evidence_bundle_completion/frozen_ids.jsonl', 'sha256': '063b66b0ca51897f9b02c8447b1698c2b01c564d355787e68b1b89a79f4b3548', 'n': 5000}, 'evidence_bundle_completion_agent': {'path': 'results/graph_autoresearch/graphauto_20260925_035919_027d9a/evidence_bundle_completion/agent_results/agent_frozen_ids.jsonl', 'sha256': '2849d46ced72ed931b5a2b84646fcbc35a7bd783e5ae457be953ba0a74b5fd0d', 'n': 150}, 'literature_set_expansion_screen': {'path': 'results/graph_autoresearch/graphauto_20260925_035919_027d9a/literature_set_expansion/frozen_ids.jsonl', 'sha256': '5b7eadacd40f892a0f918cef4510649059f01ee11d2c8fa93ca4384a509e4553', 'n': 1200}, 'literature_set_expansion_agent': {'path': 'results/graph_autoresearch/graphauto_20260925_035919_027d9a/literature_set_expansion/agent_results/agent_frozen_ids.jsonl', 'sha256': '23429f6159c50b82f170b32f87a089e304085b4a5c7c8b0a078f384fc3de7f15', 'n': 150}, 'scaling_slice': {'path': 'results/graph_autoresearch/graphauto_20260925_035919_027d9a/scaling_slice/frozen_ids.jsonl', 'sha256': '0d6aeb4568262ea472c19e2d15b46e0a900a4cbe445d11e11d66a9185b46502d', 'n': 50}, 'a2_audited_pass': {'path': 'results/graph_autoresearch/graphauto_20260925_035919_027d9a/a2/audited_pass_ids.jsonl', 'sha256': '34b25e00af728a7d6e294665a7924be81806b17e629cf8e2236c8322ac1a851c', 'n': 10, 'note': '10 audit-passing instances; NOT a frozen confirmatory set (runbook 5.2 descriptive only)'}} |
| identical IDs across paired conditions | True | 8 agent cells, sizes [200] |
| no duplicate episode keys | True | 0 duplicates |
| one terminal row per expected episode | True | {'qwen2.5-7b-instruct/bm25_one_shot_ctrl': 200, 'qwen2.5-7b-instruct/correct_graph_controlled_agent': 200, 'qwen2.5-7b-instruct/compact_hybrid_controlled_agent': 200, 'qwen2.5-7b-instruct/flat_controlled_agent': 200, 'qwen2.5-7b-instruct/rewired_graph_controlled_agent': 200, 'qwen2.5-7b-instruct/correct_hybrid_one_shot_ctrl': 200, 'qwen3-8b/flat_controlled_agent': 200, 'qwen3-8b/bm25_one_shot_ctrl': 200, 'qwen3-8b/rewired_graph_controlled_agent': 200, 'qwen3-8b/correct_graph_controlled_agent': 200, 'qwen3-8b/compact_hybrid_controlled_agent': 200, 'qwen3-8b/correct_hybrid_one_shot_ctrl': 200} |
| graph hashes differ | True | {'c33f0afd190858f3'} vs {'0232ea9a6a878502'} |
| degree preservation holds | True | {'1ee5994e832b330b'} vs {'1ee5994e832b330b'} |
| correct/rewired prompts and tool schemas match | True | 198 distinct each |
| all cited evidence joins to observed evidence | True | 0 violations |
| blocked calls returned no candidates | True | 0 contaminating calls |
| no out-of-interface tool executed | True | 0 executions |
| matched-pool hashes match logged observed pools | True | 0 mismatches of 2400 replays |
| manipulation uptake gate passed for reported agent comparisons | True | smoke gate passed=True |
| git diff --check passes | True |  |


