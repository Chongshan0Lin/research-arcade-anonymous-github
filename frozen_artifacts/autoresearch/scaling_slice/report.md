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
