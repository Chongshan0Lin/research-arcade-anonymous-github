# Root cause: zero tool calls in run `20260924_182602_a26e2e`

Audited 2026-09-25. Evidence base: the four agent conditions' episode logs
(`results/agentic_autoresearch/20260924_182602_a26e2e/episodes/{qwen3-8b,qwen2.5-7b-instruct}/*.jsonl`),
the transcript-hash response caches (`cache_qwen3-8b.jsonl`, `cache_qwen2.5-7b-instruct.jsonl`,
1307 raw generations each), and `tasks/agentic_autoresearch/canonical_agent.py`.

No private chain-of-thought is reproduced here; `<think>` spans were already stripped by
`strip_think()` before caching. Only emitted actions and final answers appear below.

## Finding

**The models never failed to call tools. They were instructed to submit immediately, and complied.**

All 2614 cached generations across both models parse as a well-formed final answer:

```
$ grep -o '"tool"[[:space:]]*:[[:space:]]*"[a-z_]*"' cache_*.jsonl | sort | uniq -c
   1307 cache_qwen2.5-7b-instruct.jsonl:"tool": "submit"
   1307 cache_qwen3-8b.jsonl:"tool": "submit"
```

Representative raw outputs (turn 1 of an agent episode, verbatim):

```json
{"tool": "submit", "ranked_paper_ids": ["2111.15141", "2302.13834", "2408.16249", "2409.09787", "2403.12063"]}
```
```json
{"tool": "submit", "ranked_paper_ids": ["2111.15141","2302.13834","2409.09614","2409.08311","2307.02037"]}
```

Zero malformed outputs, zero parser failures, zero blocked calls
(`blocked_calls == 0` and `tool_errors == 0` in all 800 agent episodes). Every episode
terminated on its first generation via `if tool == "submit": break`.

## The eight required checks

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Initial prompt already contained ten candidates and allowed immediate finalization | **CAUSE** | `canonical_agent.py:180-182` — `opening = render(bm_list[:ANSWER_K * 2], papers)` renders 10 BM25 candidates into the first user turn of every agent condition |
| 2 | `answer length exactly 5` applied to intermediate actions | **CAUSE** | `task_prompt()` appends `ANSWER_SPEC` unconditionally, including agent conditions, and it is the **last** line of the prompt: `Reply with ONE JSON object: {"tool": "submit", "ranked_paper_ids": [...]} using EXACTLY 5 distinct ids you have seen.` |
| 3 | Action and final-answer parsing used the same parser | **TRUE, latent risk** | one `parse()` (`canonical_agent.py:76-90`) handles both |
| 4 | Qwen tool-call syntax generated but rejected or stripped | Ruled out | every cached generation is already clean `{"tool": ...}` JSON; no `<tool_call>` XML, no rejects |
| 5 | Tool calls parsed but not dispatched | Ruled out | dispatch block at `canonical_agent.py:191-221` is reachable and correct; it simply never executed |
| 6 | Observations not returned to the next turn | Ruled out | `msgs += [... "OBSERVATION:\n"+payload]` at `canonical_agent.py:229-230` |
| 7 | Chat template / function-calling format mismatch | Not a cause | plain chat completions, no `tools=` parameter; both checkpoints emitted the requested bare-JSON format correctly |
| 8 | Default fallback converted malformed actions into final answers | **TRUE, latent bug — did not fire** | in `parse()`, any object whose `tool` value is not one of the four action verbs falls through to `ids = obj.get("ranked_paper_ids") or []` and returns `("submit", ...)`. A malformed action would have silently ended the episode as a final answer. No malformed actions occurred, so this did not contribute to the observed result — but §1.2 forbids it and it is fixed regardless. |

## Diagnosis

The interface made research unnecessary and then asked for the answer. Two design faults compounded:

1. **Prepopulation.** The agent's first turn already carried a 10-candidate BM25 pool — the same
   pool the one-shot conditions received. There was nothing to discover.
2. **Terminal instruction.** The prompt's closing line demanded a 5-id submission. Tool
   descriptions appeared *above* it and were optional. Submitting on turn 1 was the compliant
   reading, not a failure of instruction-following.

This is an interface/elicitation result, not a capability result, and not evidence about graph
utility. Because no traversal ever occurred, the correct and rewired adjacencies never reached
either model, which is why C4 and C5 differ by exactly 0.0.

## Code changes

Implemented in the new module `tasks/agentic_autoresearch/controlled_agent.py`
(protocol `controlled_interaction_v1`, explicitly post-hoc). `canonical_agent.py` is left
untouched so run `20260924_182602_a26e2e` stays reproducible as the `natural_action` result.

1. **Separate parsers.** `parse_action()` accepts only the four action verbs plus `finalize`;
   `parse_final()` accepts only `{"final": [...]}`. Neither can produce the other's output.
   An unrecognized object is `malformed`, never a submit.
2. **Length-5 applies only to `final`.** Intermediate turns carry no answer-format instruction.
3. **No prepopulated candidate pool.** Controlled agents open with task context and tool
   descriptions only.
4. **Mandatory minimum research phase** per condition (§1.3), disclosed in every report:
   flat → ≥1 `search`; correct/rewired graph → ≥1 `search` and ≥1 `traverse`; compact hybrid →
   ≥1 `hybrid_search`. `finalize` before the minimum is refused with a deterministic message and
   does not consume a tool call.
5. **One deterministic format-repair turn** for a malformed action; a second failure is logged as
   `parse_failed` and the episode ends as invalid, never as a silent answer.
6. **Anytime outputs** (§1.4): each turn also emits `current_top5`, stored after calls 1, 2, 4 and
   8. Never back-filled from the final answer.
7. **Separate counters**: attempted / parsed / dispatched / succeeded / blocked / failed.
8. Correct and rewired conditions share byte-identical prompts and tool schemas; only the backing
   adjacency differs. Asserted at runtime by hashing the rendered prompt and the tool doc.
