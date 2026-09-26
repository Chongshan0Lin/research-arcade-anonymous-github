"""EBC agent experiment (runbook section 3.5), protocol `ebc_controlled_v1`.

Cleared by the EBC agent-launch gate (`../evidence_bundle_completion/gate_decision.json`, PASS).
This module only DEFINES and RUNS episodes; it starts no inference on import and freezes its
evaluation ids before any model is contacted.

Protocol reused wholesale from `controlled_agent.py` (the repaired loop that took Task-A tool
uptake from 0% to 100%):
  * separate `parse_action` / `parse_final`; an unrecognised object is malformed, NEVER a silent
    final answer;
  * no prepopulated candidate list - agents open with task context and tool docs only;
  * a mandatory minimum research phase per condition, refused deterministically at no tool cost;
  * one format-repair turn, then the episode ends invalid;
  * one forced-final turn when the tool budget is exhausted, identical in every condition;
  * a context-token guard, because the server rejects over-long transcripts;
  * budgets 8 tool calls / 6000 observation tokens / 10 candidates per call;
  * anytime predictions after calls 1/2/4/8, never back-filled from the final answer;
  * attempted / parsed / dispatched / succeeded / blocked / failed counted separately;
  * byte-identical prompts and tool schemas for correct vs rewired, asserted via
    `prompt_schema_hash`; only the backing adjacency differs.

EBC-specific:
  * the anchor citation is REVEALED in the opening prompt. That is the task definition (the
    researcher already holds one seed paper), not leakage;
  * the answer is a RANKED SET of the missing targets. Its length is derived from the instance's
    own bundle, `answer_k = min(ANSWER_CAP, n_targets + ANSWER_SLACK)`, and is stated in the
    prompt together with how many of those ids are actually missing papers. The rule depends only
    on the instance, so it is identical across every condition for a given instance;
  * ALL identity, masking and temporal-legality logic goes through the version-stripped base-id
    resolution in `ebc_common` / `ebc_retrieval`. A paper's versioned and unversioned node
    spellings are one paper; the source paper and every paragraph it emits are masked by
    `group(v) != group(s)` in BOTH graph variants.

Provenance, not proof (runbook 3.5, final paragraph): for every graph-derived selected target the
episode stores the edge the tool actually traversed and the anchor->target witness path, recomputed
at scoring time in the condition's own pre-cutoff graph. This is RETRIEVAL PROVENANCE only. It
records how the candidate was reachable; it does not and must not be read as semantic evidence
that the paper supports the paragraph.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from .common import git_commit, now_iso, read_jsonl, write_json
from .ebc_build import INSTANCES, sha256_list
from .ebc_common import EBCCorpus, RRF_K, TASK_NAME, base_id, date_int, rrf
from .ebc_retrieval import graph_expand, reach_2hop
from .policies import ResponseCache, VLLMPolicy, strip_think

PROTOCOL = "ebc_controlled_v1"
PROMPT_VERSION = "ebc-v1"
SEED = 20260925

POOL_K = 20                     # candidates shown to the one-shot controls / replay
ANSWER_CAP = 10                 # ranked answers never exceed 10 ids (primary metric is Recall@10)
ANSWER_SLACK = 4                # ... and are n_targets + 4 long, so ranking has room to be wrong
BUDGET = dict(max_tool_calls=8, max_observation_tokens=6000, per_call_candidates=10,
              snippet_chars=220, max_repairs=1, max_refusals=3, max_turns=16,
              max_context_tokens=13000)
ANYTIME_AT = (1, 2, 4, 8)

AGENTS = ["flat_controlled_agent", "correct_traversal_agent", "rewired_traversal_agent",
          "compact_hybrid_agent"]
ONESHOT = ["bm25_one_shot", "correct_hybrid_one_shot"]
CONDITIONS = ONESHOT + AGENTS

# DECLARED PRIMARY CAUSAL COMPARISON, fixed before any inference:
#   correct_traversal_agent > rewired_traversal_agent on macro target Recall@10.
PRIMARY_COMPARISON = ("correct_traversal_agent", "rewired_traversal_agent", "recall@10")

# the only thing that differs between the correct and rewired conditions
GRAPH_FOR = {"correct_traversal_agent": "full", "rewired_traversal_agent": "rewire"}
# which graph a condition's graph-derived candidates must be traced back to. compact_hybrid's
# `hybrid_search` is backed by the CORRECT graph, so its provenance is checked there too.
PROVENANCE_GRAPH_FOR = dict(GRAPH_FOR, compact_hybrid_agent="full")
TOOLS_FOR = {
    "flat_controlled_agent": ["search", "open"],
    "correct_traversal_agent": ["search", "open", "traverse"],
    "rewired_traversal_agent": ["search", "open", "traverse"],
    "compact_hybrid_agent": ["search", "open", "hybrid_search"],
}
# mandatory minimum research phase (runbook 1.3), disclosed in every report
REQUIRED_FOR = {
    "flat_controlled_agent": {"search": 1},
    "correct_traversal_agent": {"search": 1, "traverse": 1},
    "rewired_traversal_agent": {"search": 1, "traverse": 1},
    "compact_hybrid_agent": {"hybrid_search": 1},
}
ACTION_VERBS = ("search", "open", "traverse", "hybrid_search")

SYSTEM = ("You are a research assistant completing a citation bundle: a paragraph cites several "
          "papers, one of which you already know, and you must recover the others. "
          "You must investigate using the tools before answering. "
          "Reply with exactly one JSON object per turn and no other text.")

TOOLS_DOC = {
    "search": '{"action":"search","query":"<text>"} - lexical search over the literature',
    "open": '{"action":"open","paper_id":"<id>"} - read a paper\'s full record',
    "traverse": '{"action":"traverse","paper_id":"<id>"} - papers structurally connected to this '
                'one (citation neighbourhood, up to two hops), most strongly connected first',
    "hybrid_search": '{"action":"hybrid_search","query":"<text>"} - combined lexical and structural '
                     'search, seeded from the paper you already know',
}

PROVENANCE_NOTE = ("A stored graph path is retrieval provenance: it records how a candidate was "
                   "reachable in the pre-cutoff graph. It is not evidence that the paper supports "
                   "the paragraph.")


_COMMIT = []


def commit():
    """git commit, resolved once per process (never per episode: this shells out)."""
    if not _COMMIT:
        _COMMIT.append(git_commit())
    return _COMMIT[0]


def ntok(s):
    return max(1, len(str(s)) // 4)


def h16(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:16]


def answer_k(inst):
    """Answer length derived from the instance's own bundle; identical across all conditions."""
    return min(ANSWER_CAP, len(inst["target_paper_ids"]) + ANSWER_SLACK)


def render(ids, corpus, n=POOL_K, snippet=BUDGET["snippet_chars"]):
    return "\n".join(f"id={p} | title={corpus.title.get(p)} | "
                     f"abstract={str(corpus.abstract.get(p))[:snippet]}" for p in ids[:n])


def turn_spec(k):
    return ('Each turn reply with ONE JSON object, either an action:\n'
            '{"action":"<verb>", ...args..., "current_ranking":["<id>",...]}\n'
            'or, once you have investigated enough, your final answer:\n'
            '{"final":["<id>", ...]}\n'
            '"current_ranking" is your best ranking so far from ids you have actually seen; it may '
            f'be empty on the first turn. Only "final" requires exactly {k} distinct ids.')


def opening_prompt(inst, tools, required):
    """No candidate pool: the agent must research. Byte-identical for correct vs rewired."""
    k = answer_k(inst)
    n_missing = len(inst["target_paper_ids"])
    need = ", ".join(f"at least {v} {t}" for t, v in sorted(required.items()))
    return "\n\n".join([
        f"Cutoff: {inst['as_of']} (only papers available by this date may be used)",
        f"Paragraph, with every citation marker replaced by [CITATION]:\n{inst['masked_text'][:2200]}",
        f"One of the cited papers is already known to you:\n"
        f"  id={inst['anchor_paper_id']} | title={inst['anchor_title']}",
        f"Beyond that known paper, this paragraph cites {n_missing} further paper(s). "
        f"Recover them and rank them, most likely first.",
        "Available tools (one action per turn):\n" + "\n".join(f"- {TOOLS_DOC[t]}" for t in tools),
        f"Required minimum investigation before answering: {need}.",
        turn_spec(k),
    ])


# ---------------------------------------------------------------- parsers

def parse_action(raw):
    """Actions only. Never returns a final answer."""
    raw = strip_think(raw or "")
    try:
        obj = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except Exception:
        return None, None, "unparseable_json"
    if not isinstance(obj, dict):
        return None, None, "not_an_object"
    if "final" in obj:
        return "finalize", obj, None
    verb = obj.get("action")
    if verb in ACTION_VERBS:
        return verb, obj, None
    if verb == "finalize":
        return "finalize", obj, None
    return None, obj, "unrecognized_action"


def parse_final(obj, allowed, k):
    """Final answers only. Requires k distinct ids the agent actually observed."""
    ids = obj.get("final") if isinstance(obj, dict) else None
    if not isinstance(ids, list):
        return [], "final_not_a_list"
    seen = [base_id(str(x)) for x in ids if base_id(str(x)) in allowed]
    seen = list(dict.fromkeys(seen))
    if not seen:
        return [], "no_valid_ids"
    return seen[:k], (None if len(seen) >= k else "short_answer")


def parse_running(obj, allowed, k):
    v = obj.get("current_ranking") if isinstance(obj, dict) else None
    if not isinstance(v, list):
        return []
    out = [base_id(str(x)) for x in v if base_id(str(x)) in allowed]
    return list(dict.fromkeys(out))[:k]


# ---------------------------------------------------------------- metrics

def met(ranked, targets, ok, k):
    """EBC metrics. Primary = macro target recall@10; set F1 is over the returned set of size k."""
    t = set(targets)
    nt = len(t)
    if not ok or not nt:
        return dict(recall_at5=0.0, recall_at10=0.0, set_f1=0.0, set_recall=0.0,
                    r_precision=0.0, mrr=0.0, hit=0.0, valid=float(ok))
    h5 = len(set(ranked[:5]) & t)
    h10 = len(set(ranked[:ANSWER_CAP]) & t)
    hk = len(set(ranked[:k]) & t)
    prec, rec = (hk / k if k else 0.0), hk / nt
    mrr = 0.0
    for i, p in enumerate(ranked, 1):
        if p in t:
            mrr = 1.0 / i
            break
    return dict(recall_at5=h5 / nt, recall_at10=h10 / nt,
                set_f1=(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0,
                set_recall=float(hk == nt), r_precision=len(set(ranked[:nt]) & t) / nt,
                mrr=mrr, hit=float(h10 > 0), valid=1.0)


# ---------------------------------------------------------------- episode state

class Ep:
    def __init__(self, inst, cond, model, rid, ghash, dhash, schema_hash):
        self.inst, self.cond, self.model, self.rid = inst, cond, model, rid
        self.ghash, self.dhash, self.schema_hash = ghash, dhash, schema_hash
        self.calls, self.obs_tokens, self.truncations = [], 0, 0
        self.observed, self.opened = [], []
        self.first_seen = {}                  # base id -> (call_index, tool, from_paper_id)
        self.attempted = self.parsed = self.dispatched = 0
        self.succeeded = self.blocked = self.failed = self.malformed = 0
        self.repairs = self.refusals = 0
        self.forced_final = False
        self.uptake = {v: 0 for v in ACTION_VERBS}
        self.anytime = {}
        self.calls_to_first_target = None
        self.t0 = time.time()

    def add_obs(self, ids, payload, tool=None, from_paper=None):
        t = ntok(payload)
        if self.obs_tokens + t > BUDGET["max_observation_tokens"]:
            self.truncations += 1
            return False
        self.obs_tokens += t
        for i in ids:
            if i not in self.first_seen:
                self.first_seen[i] = (len(self.calls) + 1, tool, from_paper)
            if i not in self.observed:
                self.observed.append(i)
        return True

    def note_targets(self, targets):
        if self.calls_to_first_target is None and any(t in self.observed for t in targets):
            self.calls_to_first_target = len(self.calls)

    def pad(self, ranked, pool, k):
        out = list(dict.fromkeys(ranked))
        n_before = len(out)
        for c in pool:
            if len(out) >= k:
                break
            if c not in out:
                out.append(c)
        return out[:k], len(out) - n_before

    def row(self, ranked, problem, targets, faithful, cited, prov, path_faithful, padded=0):
        ok = problem is None
        k = answer_k(self.inst)
        inst = self.inst
        return dict(
            run_id=self.rid, commit=commit(), protocol=PROTOCOL, task=TASK_NAME,
            condition=self.cond, model=self.model, prompt_version=PROMPT_VERSION, seed=SEED,
            resume_key=f"{self.model}|{self.cond}|{inst['instance_id']}",
            instance_id=inst["instance_id"], as_of=inst["as_of"],
            graphhard=inst.get("graphhard"),
            source_paper_id=inst["source_paper_id"], anchor_paper_id=inst["anchor_paper_id"],
            n_targets=len(targets), answer_k=k,
            graph_variant_hash=self.ghash, graph_degree_hash=self.dhash,
            prompt_schema_hash=self.schema_hash,
            tool_calls=self.calls, n_tool_calls=len(self.calls),
            calls_attempted=self.attempted, calls_parsed=self.parsed,
            calls_dispatched=self.dispatched, calls_succeeded=self.succeeded,
            blocked_calls=self.blocked, failed_calls=self.failed,
            malformed_actions=self.malformed, repairs=self.repairs,
            premature_finalize_refusals=self.refusals, forced_final=self.forced_final,
            uptake=dict(self.uptake),
            observed_candidates=self.observed, opened=self.opened,
            observation_tokens=self.obs_tokens, truncation_events=self.truncations,
            matched_pool_hash=h16("|".join(self.observed)),
            anytime=self.anytime, calls_to_first_target=self.calls_to_first_target,
            ranked=ranked, cited_evidence=cited, targets=targets,
            targets_observed=[t for t in targets if t in self.observed],
            targets_opened=[t for t in targets if t in self.opened],
            targets_selected=[t for t in targets if t in ranked],
            n_targets_observed=sum(t in self.observed for t in targets),
            n_targets_selected=sum(t in ranked for t in targets),
            retrieval_provenance_path=prov, provenance_note=PROVENANCE_NOTE,
            path_faithful=path_faithful,
            parser_problem=problem, valid=ok, padded_ids=padded, evidence_faithful=faithful,
            latency_ms=int((time.time() - self.t0) * 1000),
            metrics=met(ranked, targets, ok, k))


# ---------------------------------------------------------------- tools

def _legal_start(corpus, pid, legal):
    nodes = corpus.nodes_for_paper(pid)
    if nodes.size == 0:
        return None
    ok = nodes[legal[nodes]]
    return ok if ok.size else None


def _dispatch(verb, arg, inst, ctx, cond):
    """Execute one action. Returns (base ids, payload, error).

    Every path resolves ids through `base_id` / `nodes_for_paper` and filters through the
    per-instance `legal` mask, so the source paper, its paragraphs and every future node are
    unreachable regardless of which spelling the model supplies.
    """
    corpus, legal = ctx["corpus"], ctx["legal"]
    n = BUDGET["per_call_candidates"]
    s, as_i, anchor = inst["source_paper_id"], inst["as_of_int"], inst["anchor_paper_id"]
    drop = {base_id(s), base_id(anchor)}

    if verb not in TOOLS_FOR.get(cond, ()):
        # a tool the model asked for that this condition does not expose: blocked BEFORE execution,
        # so no graph is touched and no condition can leak a tool it is not supposed to have
        return [], {"status": "BLOCKED",
                    "reason": f"tool {verb!r} unavailable in this condition"}, None

    if verb == "search":
        q = str(arg.get("query", ""))[:400]
        if not q.strip():
            return [], {"status": "ERROR", "reason": "empty_query"}, "empty_query"
        ids = [p for p in corpus.bm25_rank(q, as_i, s, exclude=[anchor], k=n + 5)
               if p not in drop][:n]
        return ids, render(ids, corpus, n=n), None

    if verb == "hybrid_search":
        q = str(arg.get("query", ""))[:400]
        if not q.strip():
            return [], {"status": "ERROR", "reason": "empty_query"}, "empty_query"
        bl = corpus.bm25_rank(q, as_i, s, exclude=[anchor], k=POOL_K)
        gl = ctx["anchor_expansion"]["full"]
        ids = [p for p in rrf([bl, gl[:POOL_K]], k=RRF_K, budget=n + 5) if p not in drop][:n]
        return ids, render(ids, corpus, n=n), None

    if verb == "traverse":
        pid = base_id(str(arg.get("paper_id", "")).replace("paper:", "").strip())
        start = _legal_start(corpus, pid, legal)
        if start is None:
            return [], {"status": "ERROR", "reason": "unknown_or_future_id"}, "unknown_or_future_id"
        g = corpus.graphs[GRAPH_FOR[cond]]
        ranked = graph_expand(corpus, g, start, legal)[0]
        ids = [p for p in ranked if p not in drop][:n]
        if not ids:
            return [], {"status": "EMPTY",
                        "reason": "no connected papers available by the cutoff"}, None
        return ids, render(ids, corpus, n=n), None

    if verb == "open":
        pid = base_id(str(arg.get("paper_id", "")).replace("paper:", "").strip())
        if pid in drop or pid not in corpus.bm25.pos or date_int(corpus.date.get(pid)) > as_i:
            return [], {"status": "ERROR", "reason": "unknown_or_future_id"}, "unknown_or_future_id"
        return [pid], render([pid], corpus, n=1, snippet=800), None

    return [], {"status": "BLOCKED",
                "reason": f"tool {verb!r} unavailable in this condition"}, None


# ---------------------------------------------------------------- provenance

def supporting_paths(corpus, inst, ctx, cond, ep, selected_targets):
    """Recompute, at scoring time, the graph support for every graph-derived selected target.

    Records (a) the edge the tool actually traversed (from_paper -> target) and (b) the
    anchor -> target witness path, both in the CONDITION'S OWN pre-cutoff graph under the same
    legality mask. Retrieval provenance only; see PROVENANCE_NOTE.
    """
    variant = PROVENANCE_GRAPH_FOR.get(cond)
    out, all_ok = [], True
    if variant is None:
        return out, None
    g, legal = corpus.graphs[variant], ctx["legal"]
    a_nodes = ctx["anchor_nodes"]
    for t in selected_targets:
        call_idx, tool, from_paper = ep.first_seen.get(t, (None, None, None))
        if tool not in ("traverse", "hybrid_search"):
            continue                                   # not graph-derived
        t_nodes = corpus.nodes_for_paper(t)
        a_hops, a_wit = reach_2hop(corpus, g, a_nodes, t_nodes, legal)
        e_hops, e_wit = (None, None)
        if from_paper:
            f_nodes = _legal_start(corpus, from_paper, legal)
            if f_nodes is not None:
                e_hops, e_wit = reach_2hop(corpus, g, f_nodes, t_nodes, legal)
        ok = e_hops is not None if from_paper else a_hops is not None
        all_ok = all_ok and bool(ok)
        out.append(dict(
            target=t, graph_variant=variant, first_seen_at_call=call_idx, via_tool=tool,
            traversed_from=from_paper, hops_from_traversed_paper=e_hops,
            witness_from_traversed_paper=e_wit,
            hops_from_anchor=a_hops, witness_from_anchor=a_wit,
            witness_available_at=(int(corpus.avail[corpus.node_index[a_wit]])
                                  if a_wit in corpus.node_index else None),
            path_verified_in_condition_graph=bool(ok), note=PROVENANCE_NOTE))
    return out, (all_ok if out else None)


# ---------------------------------------------------------------- episode

def _instance_ctx(corpus, inst):
    s = inst["source_paper_id"]
    legal = corpus.legal_nodes(inst["as_of_int"], s)
    a_nodes = corpus.nodes_for_paper(inst["anchor_paper_id"])
    exp = {}
    for v in corpus.graphs:
        exp[v] = [] if not a_nodes.size else graph_expand(corpus, corpus.graphs[v], a_nodes, legal)[0]
    return dict(legal=legal, anchor_nodes=a_nodes, anchor_expansion=exp)


def run_oneshot(inst, cond, pol, ctx):
    """Fixed-list control: no tools, same answer format, so it is comparable to the agents."""
    corpus = ctx["corpus"]
    s, as_i, anchor = inst["source_paper_id"], inst["as_of_int"], inst["anchor_paper_id"]
    drop = {base_id(s), base_id(anchor)}
    targets = list(inst["target_paper_ids"])
    bl = [p for p in corpus.bm25_rank(inst["masked_text"], as_i, s, exclude=[anchor], k=POOL_K + 5)
          if p not in drop][:POOL_K]
    if cond == "bm25_one_shot":
        pool = bl
    else:
        gl = ctx["anchor_expansion"]["full"]
        pool = [p for p in rrf([bl, gl[:POOL_K]], k=RRF_K, budget=POOL_K + 5) if p not in drop][:POOL_K]
    ep = Ep(inst, cond, pol.model, ctx["rid"], "none", "none", h16(cond))
    ep.add_obs(pool, render(pool, corpus), tool="one_shot_pool")
    ep.note_targets(targets)
    row = ep.row([], "placeholder", targets, None, [], [], None)
    out = _select_from_pool(row, inst, pol, corpus, pool, cond)
    return dict(out, condition=cond), ep


def run_episode(inst, cond, pol, ctx):
    corpus, rid = ctx["corpus"], ctx["rid"]
    ctx = dict(ctx, **_instance_ctx(corpus, inst))
    if cond in ONESHOT:
        return run_oneshot(inst, cond, pol, ctx)

    targets = list(inst["target_paper_ids"])
    k = answer_k(inst)
    tools, required = TOOLS_FOR[cond], REQUIRED_FOR[cond]
    gv = GRAPH_FOR.get(cond, "none")
    prompt = opening_prompt(inst, tools, required)
    # correct vs rewired must be byte-identical here; only the adjacency behind `traverse` differs
    schema_hash = h16(SYSTEM + "||" + prompt + "||" + "|".join(TOOLS_DOC[t] for t in tools))
    ep = Ep(inst, cond, pol.model, rid, ctx["ghash"].get(gv, "none"),
            ctx["dhash"].get(gv, "none"), schema_hash)

    msgs = [dict(role="system", content=SYSTEM), dict(role="user", content=prompt)]
    ranked, problem, final_obj = [], "budget_exhausted", None

    for _turn in range(BUDGET["max_turns"]):
        if len(ep.calls) >= BUDGET["max_tool_calls"]:
            problem = "budget_exhausted"
            break
        if sum(ntok(m["content"]) for m in msgs) > BUDGET["max_context_tokens"]:
            # deterministic and condition-independent: stop acting and take the forced-answer turn
            # rather than letting the server reject an over-long request
            problem = "budget_exhausted"
            break
        ep.attempted += 1
        raw = pol._chat(msgs)
        verb, obj, perr = parse_action(raw)

        if perr:
            ep.malformed += 1
            if ep.repairs < BUDGET["max_repairs"]:
                ep.repairs += 1
                msgs += [dict(role="assistant", content=str(raw)[:400]),
                         dict(role="user", content=("FORMAT ERROR: reply with exactly one JSON "
                                                    "object and nothing else. " + turn_spec(k)))]
                continue
            problem = "parse_failed"            # never silently becomes a final answer
            break
        ep.parsed += 1

        if verb == "finalize":
            missing = {t: v for t, v in required.items() if ep.uptake.get(t, 0) < v}
            if missing and ep.refusals < BUDGET["max_refusals"]:
                ep.refusals += 1                # does not consume a tool call
                need = ", ".join(f"at least {v} {t}" for t, v in sorted(missing.items()))
                msgs += [dict(role="assistant", content=str(raw)[:400]),
                         dict(role="user", content=(
                             f"You have not completed the required investigation: {need}. "
                             "Issue that action now."))]
                continue
            final_obj = obj
            break

        ids, payload, err = _dispatch(verb, obj, inst, ctx, cond)
        ep.dispatched += 1
        if verb not in tools:
            ep.blocked += 1
        elif err:
            ep.failed += 1
        else:
            ep.succeeded += 1
            ep.uptake[verb] = ep.uptake.get(verb, 0) + 1
        from_paper = (base_id(str(obj.get("paper_id", "")).replace("paper:", "").strip())
                      if verb in ("traverse", "open") else None)
        ok = ep.add_obs(ids, payload, tool=verb, from_paper=from_paper)
        if verb == "open" and ids:
            for i in ids:
                if i not in ep.opened:
                    ep.opened.append(i)
        ep.calls.append(dict(tool=verb, arguments=obj if isinstance(obj, dict) else {},
                             returned_ids=ids, truncated=not ok, error=err,
                             blocked=verb not in tools))
        ep.note_targets(targets)
        running = parse_running(obj, set(ep.observed), k)
        if len(ep.calls) in ANYTIME_AT:
            ep.anytime[str(len(ep.calls))] = dict(
                ranking=running, n_targets_observed=sum(t in ep.observed for t in targets),
                metrics=met(running, targets, bool(running), k))
        msgs += [dict(role="assistant", content=json.dumps(
                     {kk: vv for kk, vv in obj.items() if kk != "current_ranking"})),
                 dict(role="user", content="OBSERVATION:\n"
                      + (str(payload) if ok else "[TRUNCATED: observation budget exhausted]"))]

    if final_obj is None and problem == "budget_exhausted" and ep.observed:
        # The agent spent its whole tool budget and was never given a turn to answer. Give exactly
        # one forced-answer turn, identically in every condition, so a researching agent is not
        # scored zero purely for running out of calls. Costs no tool call, adds no candidates.
        ep.attempted += 1
        raw = pol._chat(msgs + [dict(role="user", content=(
            "Your tool budget is exhausted. Reply now with ONE JSON object and nothing else: "
            f'{{"final":["<id>", ...]}} using EXACTLY {k} distinct ids you have already seen, '
            "most likely first."))])
        _v, obj, perr = parse_action(raw)
        if not perr and isinstance(obj, dict) and "final" in obj:
            final_obj, problem = obj, None
            ep.forced_final = True

    padded, cited, faithful, prov, pfaith = 0, [], None, [], None
    if final_obj is not None:
        ranked, problem = parse_final(final_obj, set(ep.observed), k)
        cited = list(ranked)
        faithful = all(x in ep.observed for x in cited) if cited else None
        if ranked and problem == "short_answer":
            ranked, padded = ep.pad(ranked, ep.observed, k)
            problem = None
    if ranked:
        prov, pfaith = supporting_paths(corpus, inst, ctx, cond, ep,
                                        [t for t in targets if t in ranked])
    return ep.row(ranked, problem, targets, faithful, cited, prov, pfaith, padded), ep


# ---------------------------------------------------------------- matched-pool replay

def _select_from_pool(row, inst, pol, corpus, pool, cond_label):
    """One-shot selector over exactly `pool`, under the same answer format and token ceiling."""
    k = answer_k(inst)
    targets = list(inst["target_paper_ids"])
    base = dict(row, condition=cond_label, matched_pool_hash=h16("|".join(pool)),
                tool_calls=[], n_tool_calls=0, anytime={},
                observed_candidates=list(pool), opened=[], targets_opened=[],
                targets_observed=[t for t in targets if t in pool],
                n_targets_observed=sum(t in pool for t in targets),
                uptake={v: 0 for v in ACTION_VERBS}, calls_attempted=0, calls_parsed=0,
                calls_dispatched=0, calls_succeeded=0, blocked_calls=0, failed_calls=0,
                malformed_actions=0, repairs=0, premature_finalize_refusals=0,
                forced_final=False, truncation_events=0, calls_to_first_target=None,
                resume_key=f"{row['model']}|{cond_label}|{row['instance_id']}")
    if not pool:
        return dict(base, ranked=[], valid=False, parser_problem="empty_pool",
                    metrics=met([], targets, False, k))
    n_missing = len(targets)
    p = "\n\n".join([
        f"Cutoff: {inst['as_of']}",
        f"Paragraph, with every citation marker replaced by [CITATION]:\n{inst['masked_text'][:2200]}",
        f"One of the cited papers is already known to you:\n"
        f"  id={inst['anchor_paper_id']} | title={inst['anchor_title']}",
        f"Beyond that known paper, this paragraph cites {n_missing} further paper(s).",
        f"Candidates:\n{render(pool, corpus)}",
        f'Reply with ONE JSON object: {{"final":["<id>", ...]}} using EXACTLY {k} distinct ids '
        "from the candidates, most likely first."])
    raw = strip_think(pol._chat([dict(role="system", content=SYSTEM),
                                 dict(role="user", content=p)]) or "")
    try:
        obj = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except Exception:
        obj = {}
    ranked, prob = parse_final(obj, set(pool), k)
    if ranked and prob == "short_answer":
        ranked = list(dict.fromkeys(ranked + [c for c in pool if c not in ranked]))[:k]
        prob = None
    ok = prob is None
    return dict(base, ranked=ranked, valid=ok, parser_problem=prob, cited_evidence=list(ranked),
                targets_selected=[t for t in targets if t in ranked],
                n_targets_selected=sum(t in ranked for t in targets),
                evidence_faithful=all(x in pool for x in ranked) if ranked else None,
                retrieval_provenance_path=[], path_faithful=None,
                metrics=met(ranked, targets, ok, k))


def replay(row, inst, pol, corpus):
    pool = row["observed_candidates"][:POOL_K]
    return _select_from_pool(row, inst, pol, corpus, pool,
                             f"matched_pool_replay::{row['condition']}")


# ---------------------------------------------------------------- freezing

def graph_hashes(corpus, variant):
    g = corpus.graphs[variant]
    topo = hashlib.sha256(g.indices.tobytes() + g.indptr.tobytes()).hexdigest()[:16]
    deg = hashlib.sha256(np.sort(g.degree).tobytes()).hexdigest()[:16]
    return topo, deg


def freeze_agent_ids(ebc_dir, out_root, n=150, seed=SEED):
    """Draw the agent evaluation set from the frozen General EBC slice and hash it. No inference."""
    frozen = list(read_jsonl(Path(ebc_dir) / "frozen_ids.jsonl"))
    ids = sorted(f["instance_id"] for f in frozen)
    rr = random.Random(seed)
    rr.shuffle(ids)
    pick = sorted(ids[:n])
    by = {f["instance_id"]: f for f in frozen}
    rows = [dict(instance_id=i, slices=by[i]["slices"],
                 graphhard=("graphhard" in by[i]["slices"]),
                 source_paper_id=by[i]["source_paper_id"], as_of=by[i]["as_of"],
                 anchor_paper_id=by[i]["anchor_paper_id"],
                 target_paper_ids=by[i]["target_paper_ids"],
                 n_targets=by[i]["n_targets"],
                 answer_k=min(ANSWER_CAP, by[i]["n_targets"] + ANSWER_SLACK)) for i in pick]
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "agent_frozen_ids.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    hard = [r["instance_id"] for r in rows if r["graphhard"]]
    man = dict(
        name="ebc_agent_frozen_ids", task=TASK_NAME, protocol=PROTOCOL,
        generated_at=now_iso(), git_commit=commit(),
        seed=seed, n=len(rows), n_graphhard=len(hard),
        drawn_from=str(Path(ebc_dir) / "frozen_ids.jsonl"),
        selection=f"seeded shuffle (seed {seed}) of the frozen General EBC ids, first {n}",
        frozen_before_any_inference=True,
        declared_primary_comparison=(f"{PRIMARY_COMPARISON[0]} > {PRIMARY_COMPARISON[1]} on macro "
                                     f"target {PRIMARY_COMPARISON[2]}"),
        conditions=CONDITIONS + [f"matched_pool_replay::{c}" for c in AGENTS],
        budgets=BUDGET, answer_length_rule=f"min({ANSWER_CAP}, n_targets + {ANSWER_SLACK})",
        dev_slice="EBC validation split (disjoint from the test split these ids come from)",
        provenance_note=PROVENANCE_NOTE,
        hashes=dict(agent_ids_sha256=sha256_list(pick),
                    graphhard_subset_sha256=sha256_list(hard),
                    agent_frozen_ids_file_sha256=hashlib.sha256(
                        open(path, "rb").read()).hexdigest()),
        bundle_size_histogram={str(k): sum(1 for r in rows if r["n_targets"] == k)
                               for k in sorted({r["n_targets"] for r in rows})})
    write_json(out / "agent_freeze_manifest.json", man)
    return rows, man


# ---------------------------------------------------------------- driver

def _done_keys(dirpath, cond):
    """Every instance already written for this (model, condition), across ALL shard files."""
    done = set()
    for f in glob.glob(str(Path(dirpath) / f"{cond}.*.jsonl")):
        for line in open(f):
            try:
                done.add(json.loads(line)["instance_id"])
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-root", required=True,
                    help="e.g. .../evidence_bundle_completion/agent_results")
    ap.add_argument("--ebc-dir", default=None,
                    help="EBC master dir holding frozen_ids.jsonl (default: parent of --out-root)")
    ap.add_argument("--ids", default=None, help="agent_frozen_ids.jsonl (default: inside --out-root)")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--shard", default="0/1", help="i/n - split instances across servers")
    ap.add_argument("--smoke", type=int, default=0,
                    help="run N DEV instances from the EBC validation split (never evaluated)")
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--freeze", action="store_true",
                    help="write agent_frozen_ids.jsonl + hashes and exit (no inference)")
    ap.add_argument("--freeze-n", type=int, default=150)
    a = ap.parse_args()

    root = Path(a.out_root)
    ebc_dir = Path(a.ebc_dir) if a.ebc_dir else root.parent
    if a.freeze:
        rows, man = freeze_agent_ids(ebc_dir, root, n=a.freeze_n)
        print(json.dumps({k: man[k] for k in ("n", "n_graphhard", "hashes",
                                              "bundle_size_histogram")}, indent=1))
        return 0

    si, sn = (int(x) for x in a.shard.split("/"))
    assert 0 <= si < sn, "--shard must be i/n with 0 <= i < n"
    tag = f"s{si}of{sn}"

    all_inst = {r["instance_id"]: r for r in read_jsonl(INSTANCES)}
    if a.smoke:
        # DEV slice: the EBC validation split, structurally disjoint from the test split the
        # frozen evaluation ids are drawn from. Never scored as an evaluation result.
        dev = sorted(i for i, r in all_inst.items() if r["split"] == "validation")
        random.Random(SEED + 7).shuffle(dev)
        keep = sorted(dev[:a.smoke])
        hardmap = {}
    else:
        idspath = Path(a.ids) if a.ids else root / "agent_frozen_ids.jsonl"
        frozen = [json.loads(l) for l in open(idspath)]
        keep = [f["instance_id"] for f in frozen]
        hardmap = {f["instance_id"]: bool(f.get("graphhard")) for f in frozen}
    insts = [all_inst[i] for i in keep]
    for r in insts:
        r["as_of_int"] = date_int(r["as_of"])
        r["graphhard"] = hardmap.get(r["instance_id"])
    mine = [x for j, x in enumerate(insts) if j % sn == si]
    print(f"instances={len(insts)} shard={tag} mine={len(mine)}", flush=True)

    print("loading corpus + graphs (full, rewire) ...", flush=True)
    corpus = EBCCorpus(("full", "rewire"))
    hs = {v: graph_hashes(corpus, v) for v in ("full", "rewire")}
    ghash = {v: hs[v][0] for v in hs}
    dhash = {v: hs[v][1] for v in hs}
    assert ghash["full"] != ghash["rewire"], "correct and rewired graphs must differ"
    assert dhash["full"] == dhash["rewire"], "degree sequences must match"
    print("graph hashes:", hs, flush=True)

    # correct and rewired must open with byte-identical prompts and tool docs
    if not insts:
        print("no instances selected; nothing to do")
        return 0
    probe = mine[0] if mine else insts[0]
    ha = h16(SYSTEM + "||" + opening_prompt(probe, TOOLS_FOR["correct_traversal_agent"],
                                            REQUIRED_FOR["correct_traversal_agent"]) + "||"
             + "|".join(TOOLS_DOC[t] for t in TOOLS_FOR["correct_traversal_agent"]))
    hb = h16(SYSTEM + "||" + opening_prompt(probe, TOOLS_FOR["rewired_traversal_agent"],
                                            REQUIRED_FOR["rewired_traversal_agent"]) + "||"
             + "|".join(TOOLS_DOC[t] for t in TOOLS_FOR["rewired_traversal_agent"]))
    assert ha == hb, "correct and rewired prompts/tool schemas must be byte-identical"
    print("prompt_schema_hash (correct == rewired):", ha, flush=True)

    sub = "smoke" if a.smoke else "episodes"
    outdir = root / sub / a.model
    repdir = root / ("matched_pools_smoke" if a.smoke else "matched_pools") / a.model
    outdir.mkdir(parents=True, exist_ok=True)
    repdir.mkdir(parents=True, exist_ok=True)

    cache = ResponseCache(root / f"cache_{a.model}.jsonl")
    pol = VLLMPolicy(a.model, a.base_url, cache=cache, max_tokens=640)
    pol.prompt_version = PROMPT_VERSION
    ctx = dict(corpus=corpus, rid=a.run_id, ghash=ghash, dhash=dhash)

    for cond in a.conditions:
        path = outdir / f"{cond}.{tag}.jsonl"
        rpath = repdir / f"matched_pool_replay::{cond}.{tag}.jsonl"
        # resume key is (model, condition, instance_id) and is checked across EVERY shard file of
        # this condition, so two workers can never write the same key even with overlapping shards
        done = _done_keys(outdir, cond)
        done_rep = _done_keys(repdir, f"matched_pool_replay::{cond}")
        n_new = 0
        for inst in mine:
            iid = inst["instance_id"]
            if iid in done:
                continue
            row, _ep = run_episode(inst, cond, pol, ctx)
            assert row["resume_key"] == f"{a.model}|{cond}|{iid}"
            with open(path, "a") as f:                    # append-only, resumable by key
                f.write(json.dumps(row, default=str) + "\n")
            done.add(iid)
            if not a.no_replay and cond in AGENTS and iid not in done_rep:
                rrow = replay(row, inst, pol, corpus)
                with open(rpath, "a") as f:
                    f.write(json.dumps(rrow, default=str) + "\n")
                done_rep.add(iid)
            n_new += 1
            if n_new % 10 == 0:
                print(f"  {a.model} {cond} {tag} {n_new} new", flush=True)
        print(f"[{a.model}/{cond}/{tag}] n={len(done)} new={n_new}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
