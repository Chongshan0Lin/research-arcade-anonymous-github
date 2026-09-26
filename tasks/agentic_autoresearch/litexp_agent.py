"""Literature Set Expansion / snowballing agent experiment (runbook §4.4), `litexp_controlled_v1`.

Cleared by the §4.3 launch gate
(`results/graph_autoresearch/<run>/literature_set_expansion/gate_decision.json`, PASS:
hybrid_correct − BM25 = +0.2353 [+0.2245, +0.2460] and hybrid_correct − hybrid_rewired =
+0.1750 [+0.1644, +0.1858] on Recall@50, n=909 paired, zero leakage).

This module DEFINES and RUNS episodes. It starts no inference on import, contacts no server until
`main()` is given a `--base-url`, and freezes its evaluation ids before any model is contacted
(`--freeze`).

WEAK SUPERVISION -- READ BEFORE INTERPRETING ANY NUMBER
    The label set is the source paper's own resolved earlier references. An author's bibliography
    is INCOMPLETE and SELECTIVE: it omits relevant work the authors did not know or chose not to
    cite, omits everything without a resolvable arXiv id, and includes work cited for reasons
    other than topical relevance. It is a REPRODUCIBLE WEAK RELEVANCE LABEL, NOT EXHAUSTIVE
    GROUND TRUTH. Absolute recall understates true topical recall. Only PAIRED differences
    between conditions, which share the identical label set per instance, are interpretable as
    system comparisons.

PROTOCOL, reused wholesale from `controlled_agent.py` (the repaired loop that took Task-A tool
uptake from 0% to 100%) and its EBC sibling `ebc_agent.py`:
  * separate `parse_action` / `parse_final`; an unrecognised object is malformed, NEVER a silent
    final answer;
  * no prepopulated candidate list -- agents open with task context and tool docs only;
  * a mandatory minimum research phase per condition, refused deterministically at NO tool cost;
  * ONE format-repair turn, then the episode ends invalid;
  * ONE forced-final turn when the retrieval budget is exhausted, identical in every condition;
  * a `max_context_tokens` guard, because the server rejects over-long transcripts;
  * budgets: 8 retrieval calls / 6000 observation tokens / 10 candidates per call;
  * anytime predictions after retrieval calls 1/2/4/8, never back-filled from the final answer;
  * attempted / parsed / dispatched / succeeded / blocked / failed counted separately;
  * a tool the condition does not expose is REFUSED BEFORE DISPATCH and returns ZERO ids, so a
    graph-free condition can never receive graph-derived candidates;
  * every `GRAPH_FOR[cond]` lookup is guarded -- a graph-free condition has no adjacency;
  * byte-identical prompts and tool schemas for correct vs rewired, asserted via
    `prompt_schema_hash`; ONLY the backing adjacency differs;
  * resume key `(model, condition, instance_id)` scanned across ALL shard files of a condition, so
    two workers can never double-write.

LITERATURE-SET SPECIFIC
  * the system sees ONLY the source paper's title and abstract, citation markers scrubbed -- what
    a researcher has when formulating a related-work search. The bibliography, citation markers,
    gold ids and source-derived co-citation relations are never shown;
  * identity is resolved on the VERSION-STRIPPED BASE ID everywhere (rule T0 of
    `literature_set_expansion.py`): `paper:X` and `paper:Xv2` are one paper. Without this, masking
    on the literal source id leaves the source's own paragraphs live and its bibliography leaks;
  * BLOCKED NODES = {paper:base(S)} ∪ {every paragraph owned by any spelling of S}. Traversal may
    neither enter nor leave a blocked node, in BOTH graph variants. This removes every edge the
    source emitted, including all source-derived co-citation structure;
  * `add_to_review(paper, short_rationale)` is BOOKKEEPING, not retrieval: it returns no
    candidates, costs no retrieval call, and is capped separately. It exists because §4.4 asks for
    inclusion rationales. IDS ARE SCORED INDEPENDENTLY OF PROSE QUALITY -- rationale text never
    enters any effectiveness metric; it is only checked for faithfulness (does it cite ids the
    agent actually observed?) and reported separately.

ANSWER AND SCORING RULE (fixed before any inference, identical in every condition)
    The model is asked for a ranked set of ANSWER_K = 20 ids it has actually observed. The scored
    list is then, in order: (1) the model's ranking, (2) any `add_to_review` ids not already in it,
    in the order they were added, (3) the remaining observed candidates in first-seen order,
    truncated to SCORE_CAP = 50. Padding is logged in `padded_ids` and applied identically in
    every condition. Recall@20 is therefore the model's own ranking; Recall@50 additionally
    credits what the agent found but did not rank.

DECLARED PRIMARY CAUSAL COMPARISON, fixed in code before any inference:
    correct_snowballing_agent > rewired_snowballing_agent on Recall@50.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from .common import git_commit, now_iso, read_jsonl, write_json
from .evaluate import ndcg_at_k, recall_at_k
from .hybrid_retrieval import RRF_K, rrf
from .literature_set_expansion import (FastBM25, K_FINAL, MAX_HOPS, N_BM25_SCAN, N_SEEDS,
                                       PER_NODE_CAP, average_precision_at_k, base_pid,
                                       degree_equality_check, graph_rank_blocked,
                                       load_graphs_base_normalised, _legal)
from .policies import ResponseCache, VLLMPolicy, strip_think
from .temporal_store import normalize_ts

PROTOCOL = "litexp_controlled_v1"
PROMPT_VERSION = "litexp-v1"
TASK_NAME = "literature_set_expansion"
SEED = 20260925

ANSWER_K = 20                  # ids the model is asked to rank
SCORE_CAP = 50                 # scored list length after the declared padding rule
SCORE_KS = (10, 20, 50)
# Candidates shown to the fixed-list controls and to the matched-pool replay. It is deliberately
# equal to SCORE_CAP: a control capped below the cap at which the agents are scored would lose the
# confirmatory comparison on budget alone rather than on interface quality.
POOL_K = SCORE_CAP
BUDGET = dict(max_tool_calls=8, max_observation_tokens=6000, per_call_candidates=10,
              snippet_chars=220, max_repairs=1, max_refusals=3, max_turns=30,
              max_context_tokens=13000, max_review_adds=30)
ANYTIME_AT = (1, 2, 4, 8)

FIXED = ["bm25_fixed_list", "correct_hybrid_fixed_list"]
AGENTS = ["flat_literature_agent", "correct_snowballing_agent", "rewired_snowballing_agent",
          "compact_hybrid_literature_agent"]
CONDITIONS = FIXED + AGENTS

# DECLARED PRIMARY CAUSAL COMPARISON -- fixed here, before any inference.
PRIMARY_COMPARISON = ("correct_snowballing_agent", "rewired_snowballing_agent", "recall@50")
# Confirmatory family (Holm-adjusted together, and ONLY these):
CONFIRMATORY_FAMILY = (
    [PRIMARY_COMPARISON,
     ("compact_hybrid_literature_agent", "correct_hybrid_fixed_list", "recall@50")]
    + [(a, f"matched_pool_replay::{a}", "recall@50") for a in AGENTS])

# the ONLY thing that differs between the correct and rewired snowballing conditions
GRAPH_FOR = {"correct_snowballing_agent": "full", "rewired_snowballing_agent": "rewire"}
# compact_hybrid's `hybrid_search` is backed by the CORRECT graph (it is the compiled interface
# under test, not a connectivity control), recorded so aggregation can trace provenance.
GRAPH_BACKING = dict(GRAPH_FOR, compact_hybrid_literature_agent="full")

GRAPH_VERBS = ("backward_citations", "forward_citations", "related_by_cocitation")
TOOLS_FOR = {
    "flat_literature_agent": ["search", "open", "add_to_review"],
    "correct_snowballing_agent": ["search", "open", "backward_citations", "forward_citations",
                                  "related_by_cocitation", "add_to_review"],
    "rewired_snowballing_agent": ["search", "open", "backward_citations", "forward_citations",
                                  "related_by_cocitation", "add_to_review"],
    "compact_hybrid_literature_agent": ["search", "open", "hybrid_search", "add_to_review"],
}
# mandatory minimum research phase (runbook 1.3), disclosed in every report.
# The two graph conditions carry IDENTICAL requirements and IDENTICAL budgets.
REQUIRED_FOR = {
    "flat_literature_agent": {"search": 2},
    "correct_snowballing_agent": {"search": 1, "backward_citations": 1},
    "rewired_snowballing_agent": {"search": 1, "backward_citations": 1},
    "compact_hybrid_literature_agent": {"hybrid_search": 2},
}
ACTION_VERBS = ("search", "open", "hybrid_search", "backward_citations", "forward_citations",
                "related_by_cocitation", "add_to_review")
# actions that retrieve: these consume the 8-call retrieval budget
RETRIEVAL_VERBS = tuple(v for v in ACTION_VERBS if v != "add_to_review")

SYSTEM = ("You are a research assistant building the related-work set for a new paper. "
          "Given only its title and abstract, you must find the earlier papers it should cite. "
          "You must investigate using the tools before answering. "
          "Reply with exactly one JSON object per turn and no other text.")

TOOLS_DOC = {
    "search": '{"action":"search","query":"<text>"} - lexical search over the literature',
    "open": '{"action":"open","paper":"<id>"} - read a paper\'s full record',
    "hybrid_search": '{"action":"hybrid_search","query":"<text>"} - combined lexical and '
                     'structural search',
    "backward_citations": '{"action":"backward_citations","paper":"<id>"} - earlier papers this '
                          'paper builds on',
    "forward_citations": '{"action":"forward_citations","paper":"<id>","cutoff":"<date>"} - later '
                         'papers that build on this one, restricted to the cutoff',
    "related_by_cocitation": '{"action":"related_by_cocitation","paper":"<id>"} - papers '
                             'frequently cited alongside this one',
    "add_to_review": '{"action":"add_to_review","paper":"<id>","short_rationale":"<one sentence>"}'
                     ' - add a paper to your review set with a one-sentence reason. This is '
                     'bookkeeping: it costs no search budget and returns no new papers.',
}

WEAK_LABEL_NOTE = (
    "Scored against the source paper's own resolved earlier references: an incomplete and "
    "selective weak relevance label, not exhaustive ground truth. Absolute recall understates "
    "true topical recall; only paired between-condition differences are interpretable.")
SCORING_NOTE = (
    "Ids are scored independently of prose quality (runbook 4.4). Rationale text never enters an "
    "effectiveness metric; it is only checked for faithfulness and reported separately.")

_COMMIT = []


def commit():
    if not _COMMIT:
        _COMMIT.append(git_commit())
    return _COMMIT[0]


def ntok(s):
    return max(1, len(str(s)) // 4)


def h16(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:16]


def sha256_list(xs):
    return hashlib.sha256("\n".join(str(x) for x in xs).encode()).hexdigest()


# ---------------------------------------------------------------- prompts

def render(ids, papers, n=POOL_K, snippet=BUDGET["snippet_chars"]):
    return "\n".join(f"id={p} | title={papers.get(p, {}).get('title')} | "
                     f"abstract={str(papers.get(p, {}).get('abstract'))[:snippet]}"
                     for p in ids[:n])


def turn_spec(k):
    return ('Each turn reply with ONE JSON object, either an action:\n'
            '{"action":"<verb>", ...args..., "current_ranking":["<id>",...]}\n'
            'or, once you have investigated enough, your final answer:\n'
            '{"final":["<id>", ...], "rationales":{"<id>":"<one sentence>"}}\n'
            '"current_ranking" is your best ranking so far from ids you have actually seen; it '
            f'may be empty on the first turn. Only "final" requires exactly {k} distinct ids, '
            'most relevant first. "rationales" is optional and is never scored for wording.')


def opening_prompt(inst, tools, required):
    """No candidate pool: the agent must research. Byte-identical for correct vs rewired."""
    need = ", ".join(f"at least {v} {t}" for t, v in sorted(required.items()))
    return "\n\n".join([
        f"Cutoff: {inst['as_of']} (only papers available by this date may be cited)",
        f"Title of the paper being written:\n{inst['title']}",
        f"Abstract:\n{str(inst['query'])[:2600]}",
        f"Build its related-work set: find and rank the earlier papers it should cite. "
        f"Return {ANSWER_K} ids, most relevant first.",
        "Available tools (one action per turn):\n" + "\n".join(f"- {TOOLS_DOC[t]}" for t in tools),
        f"Required minimum investigation before answering: {need}.",
        f"You have {BUDGET['max_tool_calls']} search/traversal calls. add_to_review does not use "
        "them.",
        turn_spec(ANSWER_K),
    ])


# ---------------------------------------------------------------- parsers

def parse_action(raw):
    """Actions only. NEVER returns a final answer."""
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


def parse_final(obj, allowed, k=ANSWER_K):
    """Final answers only. Requires k distinct ids the agent actually observed."""
    ids = obj.get("final") if isinstance(obj, dict) else None
    if not isinstance(ids, list):
        return [], "final_not_a_list"
    seen = [base_pid(str(x).replace("paper:", "")) for x in ids]
    seen = [x for x in seen if x in allowed]
    seen = list(dict.fromkeys(seen))
    if not seen:
        return [], "no_valid_ids"
    return seen[:k], (None if len(seen) >= k else "short_answer")


def parse_running(obj, allowed, k=ANSWER_K):
    v = obj.get("current_ranking") if isinstance(obj, dict) else None
    if not isinstance(v, list):
        return []
    out = [base_pid(str(x).replace("paper:", "")) for x in v]
    out = [x for x in out if x in allowed]
    return list(dict.fromkeys(out))[:k]


def parse_rationales(obj):
    v = obj.get("rationales") if isinstance(obj, dict) else None
    if not isinstance(v, dict):
        return {}
    return {base_pid(str(kk).replace("paper:", "")): str(vv)[:400] for kk, vv in v.items()}


# ---------------------------------------------------------------- metrics

def met(ranked, gold, ok):
    """Ids only. Prose never enters this function (runbook 4.4)."""
    g = set(gold)
    if not ok or not g:
        return dict({f"recall@{k}": 0.0 for k in SCORE_KS},
                    **{f"ndcg@{k}": 0.0 for k in SCORE_KS},
                    map_at_50=0.0, hit=0.0, valid=float(ok))
    out = {}
    for k in SCORE_KS:
        out[f"recall@{k}"] = recall_at_k(ranked, list(g), k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, list(g), k)
    out["map_at_50"] = average_precision_at_k(ranked, g, 50)
    out["hit"] = float(bool(set(ranked[:SCORE_CAP]) & g))
    out["valid"] = 1.0
    return out


# ---------------------------------------------------------------- episode state

class Ep:
    def __init__(self, inst, cond, model, rid, ghash, dhash, schema_hash):
        self.inst, self.cond, self.model, self.rid = inst, cond, model, rid
        self.ghash, self.dhash, self.schema_hash = ghash, dhash, schema_hash
        self.calls, self.obs_tokens, self.truncations = [], 0, 0
        self.observed, self.opened = [], []
        self.review, self.rationales = [], {}
        self.first_seen = {}                 # base id -> (retrieval call index, tool, from_paper)
        self.attempted = self.parsed = self.dispatched = 0
        self.succeeded = self.blocked = self.failed = self.malformed = 0
        self.repairs = self.refusals = self.review_adds = 0
        self.forced_final = False
        self.uptake = {v: 0 for v in ACTION_VERBS}
        self.anytime = {}
        self.calls_to_first_gold = None
        self.t0 = time.time()

    def n_retrieval_calls(self):
        return sum(1 for c in self.calls if c["tool"] in RETRIEVAL_VERBS)

    def add_obs(self, ids, payload, tool=None, from_paper=None):
        t = ntok(payload)
        if self.obs_tokens + t > BUDGET["max_observation_tokens"]:
            self.truncations += 1
            return False
        self.obs_tokens += t
        for i in ids:
            if i not in self.first_seen:
                self.first_seen[i] = (self.n_retrieval_calls() + 1, tool, from_paper)
            if i not in self.observed:
                self.observed.append(i)
        return True

    def note_gold(self, gold):
        if self.calls_to_first_gold is None and any(g in self.observed for g in gold):
            self.calls_to_first_gold = self.n_retrieval_calls()

    def scored_list(self, ranked):
        """DECLARED padding rule, identical in every condition; see the module docstring."""
        out = list(dict.fromkeys(ranked))
        n_model = len(out)
        for c in self.review + self.observed:
            if len(out) >= SCORE_CAP:
                break
            if c not in out:
                out.append(c)
        return out[:SCORE_CAP], max(0, len(out) - n_model)

    def row(self, ranked, scored, problem, gold, faithful, cited, rationales,
            rationale_faithful, padded=0):
        ok = problem is None
        inst = self.inst
        gset = set(gold)
        graph_only = [g for g in gset & set(scored)
                      if self.first_seen.get(g, (None, None, None))[1] in GRAPH_VERBS
                      or self.first_seen.get(g, (None, None, None))[1] == "hybrid_search"]
        return dict(
            run_id=self.rid, commit=commit(), protocol=PROTOCOL, task=TASK_NAME,
            condition=self.cond, model=self.model, prompt_version=PROMPT_VERSION, seed=SEED,
            resume_key=f"{self.model}|{self.cond}|{inst['source_id']}",
            instance_id=inst["source_id"], source_id=inst["source_id"], as_of=inst["as_of"],
            split=inst.get("split"), cluster=inst.get("cluster"),
            primary_category=inst.get("primary_category"),
            n_gold=len(gset), answer_k=ANSWER_K, score_cap=SCORE_CAP,
            graph_variant_hash=self.ghash, graph_degree_hash=self.dhash,
            prompt_schema_hash=self.schema_hash,
            graph_backing=GRAPH_BACKING.get(self.cond),
            tool_calls=self.calls, n_tool_calls=len(self.calls),
            n_retrieval_calls=self.n_retrieval_calls(),
            calls_attempted=self.attempted, calls_parsed=self.parsed,
            calls_dispatched=self.dispatched, calls_succeeded=self.succeeded,
            blocked_calls=self.blocked, failed_calls=self.failed,
            malformed_actions=self.malformed, repairs=self.repairs,
            premature_finalize_refusals=self.refusals, forced_final=self.forced_final,
            review_adds=self.review_adds, review_set=self.review,
            uptake=dict(self.uptake),
            observed_candidates=self.observed, n_observed=len(self.observed), opened=self.opened,
            observation_tokens=self.obs_tokens, truncation_events=self.truncations,
            matched_pool_hash=h16("|".join(self.observed)),
            anytime=self.anytime, calls_to_first_gold=self.calls_to_first_gold,
            ranked=ranked, scored_ranking=scored, padded_ids=padded,
            cited_evidence=cited, rationales=rationales,
            rationale_ids_observed=rationale_faithful,
            unsupported_inclusions=[x for x in scored if x not in self.observed],
            gold=sorted(gset),
            gold_observed=sorted(gset & set(self.observed)),
            n_gold_observed=len(gset & set(self.observed)),
            n_gold_selected=len(gset & set(scored)),
            graph_only_gold=sorted(graph_only), n_graph_only_gold=len(graph_only),
            parser_problem=problem, valid=ok, evidence_faithful=faithful,
            latency_ms=int((time.time() - self.t0) * 1000),
            weak_label_note=WEAK_LABEL_NOTE, scoring_note=SCORING_NOTE,
            metrics=met(scored, sorted(gset), ok))


# ---------------------------------------------------------------- tools

def _bm25_legal(ctx, inst, query, k):
    """Temporally legal, base-id-normalised BM25 list; the source paper is never a candidate."""
    src, as_of = inst["source_id"], inst["as_of"]
    out, seen = [], set()
    for pid, _s in ctx["bm25"].search(query, N_BM25_SCAN):
        p = base_pid(pid)
        if p in seen or p == src or not _legal(p, ctx["papers"], as_of):
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= k:
            break
    return out


def _neighbours(ctx, inst, cond, pid, direction, blocked, cutoff_arg=None):
    """Direct citation neighbourhood of `pid` in THIS condition's graph.

    Returns [] for a condition with no adjacency. Every id is base-normalised, temporally legal,
    not the source, and never a blocked (source-emitted) node.
    """
    gv = GRAPH_BACKING.get(cond)
    if gv is None:                       # graph-free condition: guarded, never a bare dict index
        return []
    adj = ctx["adjs"].get(gv)
    if adj is None:
        return []
    src, papers = inst["source_id"], ctx["papers"]
    as_of = inst["as_of"]
    if cutoff_arg:                       # forward_citations(paper, cutoff): never later than as_of
        c = normalize_ts(cutoff_arg)
        if c and c < normalize_ts(as_of):
            as_of = cutoff_arg
    node = f"paper:{base_pid(pid)}"
    if node in blocked:
        return []
    out = []
    for nbr, rel, d in sorted(adj.get(node, []), key=lambda x: x[0]):
        if nbr in blocked or not nbr.startswith("paper:") or rel != "cites":
            continue
        if direction == "backward" and d != "out":
            continue
        if direction == "forward" and d != "in":
            continue
        p = base_pid(nbr.split(":", 1)[1])
        if p == src or not _legal(p, papers, as_of):
            continue
        if p not in out:
            out.append(p)
    return out


def _cocitation(ctx, inst, cond, pid, blocked, n):
    """Papers frequently cited alongside `pid`, in THIS condition's graph.

    Co-citation is computed ONLY over pre-cutoff citing papers, and every blocked (source-emitted)
    node is excluded, so the source's own co-citation structure can never appear.
    """
    gv = GRAPH_BACKING.get(cond)
    if gv is None:
        return []
    adj = ctx["adjs"].get(gv)
    if adj is None:
        return []
    src, papers, as_of = inst["source_id"], ctx["papers"], inst["as_of"]
    node = f"paper:{base_pid(pid)}"
    if node in blocked:
        return []
    counts = defaultdict(int)
    citers = [x for x, rel, d in adj.get(node, [])
              if rel == "cites" and d == "in" and x not in blocked and x.startswith("paper:")]
    for c in sorted(citers)[:PER_NODE_CAP]:
        if not _legal(base_pid(c.split(":", 1)[1]), papers, as_of):
            continue                      # a future citer may not be used as a co-citation bridge
        for nbr, rel, d in sorted(adj.get(c, []), key=lambda x: x[0])[:PER_NODE_CAP]:
            if rel != "cites" or d != "out" or nbr in blocked or not nbr.startswith("paper:"):
                continue
            p = base_pid(nbr.split(":", 1)[1])
            if p == src or p == base_pid(pid) or not _legal(p, papers, as_of):
                continue
            counts[p] += 1
    return sorted(counts, key=lambda p: (-counts[p], p))[:n]


def _dispatch(verb, arg, inst, ctx, cond, blocked):
    """Execute one action. Returns (base ids, payload, error, is_retrieval).

    A tool the condition does not expose is BLOCKED HERE, before anything is executed, and returns
    ZERO ids -- so a graph-free condition can never receive graph-derived candidates.
    """
    n = BUDGET["per_call_candidates"]
    papers = ctx["papers"]
    if verb not in TOOLS_FOR.get(cond, ()):
        return [], {"status": "BLOCKED",
                    "reason": f"tool {verb!r} unavailable in this condition"}, None, True

    if verb == "add_to_review":
        pid = base_pid(str(arg.get("paper", arg.get("paper_id", ""))).replace("paper:", "").strip())
        if pid not in ctx["ep"].observed:
            return [], {"status": "ERROR",
                        "reason": "you may only add papers you have actually seen"}, \
                "unobserved_id", False
        ep = ctx["ep"]
        if ep.review_adds >= BUDGET["max_review_adds"]:
            return [], {"status": "BLOCKED", "reason": "review set is full"}, None, False
        ep.review_adds += 1
        if pid not in ep.review:
            ep.review.append(pid)
        r = str(arg.get("short_rationale", ""))[:400]
        if r:
            ep.rationales[pid] = r
        return [], {"status": "OK", "review_set_size": len(ep.review)}, None, False

    if verb == "search":
        q = str(arg.get("query", ""))[:400]
        if not q.strip():
            return [], {"status": "ERROR", "reason": "empty_query"}, "empty_query", True
        ids = _bm25_legal(ctx, inst, q, n)
        return ids, render(ids, papers, n=n), None, True

    if verb == "hybrid_search":
        q = str(arg.get("query", ""))[:400]
        if not q.strip():
            return [], {"status": "ERROR", "reason": "empty_query"}, "empty_query", True
        gv = GRAPH_BACKING.get(cond)
        bl = _bm25_legal(ctx, inst, q, POOL_K)
        gl = []
        if gv is not None and ctx["adjs"].get(gv) is not None:
            gl = graph_rank_blocked(bl[:N_SEEDS], ctx["adjs"][gv], papers, inst["as_of"],
                                    inst["source_id"], blocked, k=POOL_K)[0]
        ids = rrf([bl, gl], k=RRF_K, budget=n) if gl else bl[:n]
        return ids, render(ids, papers, n=n), None, True

    if verb in ("backward_citations", "forward_citations"):
        pid = str(arg.get("paper", arg.get("paper_id", ""))).replace("paper:", "").strip()
        if not pid:
            return [], {"status": "ERROR", "reason": "missing paper id"}, "missing_id", True
        direction = "backward" if verb == "backward_citations" else "forward"
        ids = _neighbours(ctx, inst, cond, pid, direction, blocked,
                          cutoff_arg=arg.get("cutoff") if direction == "forward" else None)[:n]
        if not ids:
            return [], {"status": "EMPTY",
                        "reason": "no connected papers available by the cutoff"}, None, True
        return ids, render(ids, papers, n=n), None, True

    if verb == "related_by_cocitation":
        pid = str(arg.get("paper", arg.get("paper_id", ""))).replace("paper:", "").strip()
        if not pid:
            return [], {"status": "ERROR", "reason": "missing paper id"}, "missing_id", True
        ids = _cocitation(ctx, inst, cond, pid, blocked, n)
        if not ids:
            return [], {"status": "EMPTY",
                        "reason": "no co-cited papers available by the cutoff"}, None, True
        return ids, render(ids, papers, n=n), None, True

    if verb == "open":
        pid = base_pid(str(arg.get("paper", arg.get("paper_id", ""))).replace("paper:", "").strip())
        if (pid and pid != inst["source_id"] and pid in papers
                and _legal(pid, papers, inst["as_of"])):
            return [pid], render([pid], papers, n=1, snippet=800), None, True
        return [], {"status": "ERROR", "reason": "unknown_or_future_id"}, \
            "unknown_or_future_id", True

    return [], {"status": "BLOCKED",
                "reason": f"tool {verb!r} unavailable in this condition"}, None, True


# ---------------------------------------------------------------- episode

def blocked_nodes(inst, ctx):
    """T3: the source paper plus every paragraph it emits, under EVERY id spelling."""
    src = inst["source_id"]
    return {f"paper:{src}"} | set(ctx["owned"].get(src, ()))


def fixed_pool(inst, cond, ctx, blocked):
    bl = _bm25_legal(ctx, inst, inst["query"], POOL_K)
    if cond == "bm25_fixed_list":
        return bl
    gl = graph_rank_blocked(bl[:N_SEEDS], ctx["adjs"]["full"], ctx["papers"], inst["as_of"],
                            inst["source_id"], blocked, k=POOL_K)[0]
    return rrf([bl, gl], k=RRF_K, budget=POOL_K) if gl else bl


def run_fixed(inst, cond, pol, ctx):
    """Fixed-list control: no tools, same answer format, so it is comparable to the agents."""
    blocked = blocked_nodes(inst, ctx)
    gold = sorted(set(inst["gold"]))
    pool = fixed_pool(inst, cond, ctx, blocked)
    ep = Ep(inst, cond, pol.model, ctx["rid"], "none", "none", h16(cond))
    ctx = dict(ctx, ep=ep)
    ep.add_obs(pool, render(pool, ctx["papers"]), tool="fixed_list")
    ep.note_gold(gold)
    base = ep.row([], [], "placeholder", gold, None, [], {}, None)
    return dict(_select_from_pool(base, inst, pol, ctx, pool, cond), condition=cond), ep


def run_episode(inst, cond, pol, ctx):
    if cond in FIXED:
        return run_fixed(inst, cond, pol, ctx)
    papers, rid = ctx["papers"], ctx["rid"]
    gold = sorted(set(inst["gold"]))
    blocked = blocked_nodes(inst, ctx)
    tools, required = TOOLS_FOR[cond], REQUIRED_FOR[cond]
    gv = GRAPH_FOR.get(cond, "none")
    prompt = opening_prompt(inst, tools, required)
    # correct vs rewired must be byte-identical here; only the adjacency behind the traversal
    # tools differs
    schema_hash = h16(SYSTEM + "||" + prompt + "||" + "|".join(TOOLS_DOC[t] for t in tools))
    ep = Ep(inst, cond, pol.model, rid, ctx["ghash"].get(gv, "none"),
            ctx["dhash"].get(gv, "none"), schema_hash)
    ctx = dict(ctx, ep=ep)

    msgs = [dict(role="system", content=SYSTEM), dict(role="user", content=prompt)]
    ranked, problem, final_obj = [], "budget_exhausted", None

    for _turn in range(BUDGET["max_turns"]):
        if ep.n_retrieval_calls() >= BUDGET["max_tool_calls"]:
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
                                                    "object and nothing else. "
                                                    + turn_spec(ANSWER_K)))]
                continue
            problem = "parse_failed"             # never silently becomes a final answer
            break
        ep.parsed += 1

        if verb == "finalize":
            missing = {t: v for t, v in required.items() if ep.uptake.get(t, 0) < v}
            if missing and ep.refusals < BUDGET["max_refusals"]:
                ep.refusals += 1                 # deterministic; consumes no retrieval call
                need = ", ".join(f"at least {v} {t}" for t, v in sorted(missing.items()))
                msgs += [dict(role="assistant", content=str(raw)[:400]),
                         dict(role="user", content=(
                             f"You have not completed the required investigation: {need}. "
                             "Issue that action now."))]
                continue
            final_obj = obj
            break

        ids, payload, err, is_retrieval = _dispatch(verb, obj, inst, ctx, cond, blocked)
        ep.dispatched += 1
        if verb not in tools:
            ep.blocked += 1
        elif err:
            ep.failed += 1
        else:
            ep.succeeded += 1
            ep.uptake[verb] = ep.uptake.get(verb, 0) + 1
        from_paper = (base_pid(str(obj.get("paper", obj.get("paper_id", "")))
                               .replace("paper:", "").strip())
                      if verb in GRAPH_VERBS + ("open",) else None)
        ok = ep.add_obs(ids, payload, tool=verb, from_paper=from_paper)
        if verb == "open" and ids:
            for i in ids:
                if i not in ep.opened:
                    ep.opened.append(i)
        ep.calls.append(dict(tool=verb, arguments=obj if isinstance(obj, dict) else {},
                             returned_ids=ids, truncated=not ok, error=err,
                             blocked=verb not in tools, retrieval=is_retrieval))
        ep.note_gold(gold)
        running = parse_running(obj, set(ep.observed))
        nrc = ep.n_retrieval_calls()
        if is_retrieval and nrc in ANYTIME_AT and str(nrc) not in ep.anytime:
            scored_now, _p = ep.scored_list(running)
            ep.anytime[str(nrc)] = dict(          # never back-filled from the final answer
                ranking=running, n_gold_observed=sum(g in ep.observed for g in gold),
                metrics=met(scored_now, gold, bool(running)))
        msgs += [dict(role="assistant", content=json.dumps(
                     {kk: vv for kk, vv in obj.items() if kk != "current_ranking"})),
                 dict(role="user", content="OBSERVATION:\n"
                      + (str(payload) if ok else "[TRUNCATED: observation budget exhausted]"))]

    if final_obj is None and problem == "budget_exhausted" and ep.observed:
        # The agent spent its whole retrieval budget and was never given a turn to answer. Exactly
        # one forced-answer turn, identical in every condition, so a researching agent is not
        # scored zero purely for running out of calls. Costs no retrieval call, adds no candidates.
        ep.attempted += 1
        raw = pol._chat(msgs + [dict(role="user", content=(
            "Your search budget is exhausted. Reply now with ONE JSON object and nothing else: "
            f'{{"final":["<id>", ...]}} using EXACTLY {ANSWER_K} distinct ids you have already '
            "seen, most relevant first."))])
        _v, obj, perr = parse_action(raw)
        if not perr and isinstance(obj, dict) and "final" in obj:
            final_obj, problem = obj, None
            ep.forced_final = True

    padded, cited, faithful, rats, rat_faith = 0, [], None, {}, None
    scored = []
    if final_obj is not None:
        ranked, problem = parse_final(final_obj, set(ep.observed))
        cited = list(ranked)
        faithful = all(x in ep.observed for x in cited) if cited else None
        rats = dict(ep.rationales)
        rats.update(parse_rationales(final_obj))
        if rats:
            # faithfulness of the RATIONALE KEYS only; the prose itself is never scored
            rat_faith = all(k in ep.observed for k in rats)
        if ranked and problem == "short_answer":
            problem = None                        # padded below by the declared rule
        if ranked:
            scored, padded = ep.scored_list(ranked)
    return ep.row(ranked, scored, problem, gold, faithful, cited, rats, rat_faith, padded), ep


# ---------------------------------------------------------------- matched-pool replay

def _select_from_pool(row, inst, pol, ctx, pool, cond_label):
    """One-shot selector over exactly `pool`, under the same answer format and token ceiling."""
    papers = ctx["papers"]
    gold = sorted(set(inst["gold"]))
    base = dict(row, condition=cond_label, matched_pool_hash=h16("|".join(pool)),
                tool_calls=[], n_tool_calls=0, n_retrieval_calls=0, anytime={},
                observed_candidates=list(pool), n_observed=len(pool), opened=[],
                review_set=[], review_adds=0,
                gold_observed=sorted(set(gold) & set(pool)),
                n_gold_observed=len(set(gold) & set(pool)),
                uptake={v: 0 for v in ACTION_VERBS}, calls_attempted=0, calls_parsed=0,
                calls_dispatched=0, calls_succeeded=0, blocked_calls=0, failed_calls=0,
                malformed_actions=0, repairs=0, premature_finalize_refusals=0,
                forced_final=False, truncation_events=0, calls_to_first_gold=None,
                graph_only_gold=[], n_graph_only_gold=0,
                resume_key=f"{row['model']}|{cond_label}|{row['instance_id']}")
    if not pool:
        return dict(base, ranked=[], scored_ranking=[], valid=False, parser_problem="empty_pool",
                    metrics=met([], gold, False))
    p = "\n\n".join([
        f"Cutoff: {inst['as_of']}",
        f"Title of the paper being written:\n{inst['title']}",
        f"Abstract:\n{str(inst['query'])[:2600]}",
        f"Candidates:\n{render(pool, papers)}",
        f'Reply with ONE JSON object: {{"final":["<id>", ...]}} using EXACTLY '
        f"{min(ANSWER_K, len(pool))} distinct ids from the candidates, most relevant first."])
    raw = strip_think(pol._chat([dict(role="system", content=SYSTEM),
                                 dict(role="user", content=p)]) or "")
    try:
        obj = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except Exception:
        obj = {}
    ranked, prob = parse_final(obj, set(pool), min(ANSWER_K, len(pool)))
    if ranked and prob == "short_answer":
        prob = None
    scored = list(dict.fromkeys(list(ranked) + [c for c in pool if c not in ranked]))[:SCORE_CAP]
    ok = prob is None
    return dict(base, ranked=ranked, scored_ranking=(scored if ok else []),
                valid=ok, parser_problem=prob, cited_evidence=list(ranked),
                padded_ids=max(0, len(scored) - len(ranked)),
                unsupported_inclusions=[x for x in scored if x not in pool],
                n_gold_selected=len(set(gold) & set(scored)),
                evidence_faithful=all(x in pool for x in ranked) if ranked else None,
                metrics=met(scored if ok else [], gold, ok))


def replay(row, inst, pol, ctx):
    pool = row["observed_candidates"][:POOL_K]
    return _select_from_pool(row, inst, pol, ctx, pool,
                             f"matched_pool_replay::{row['condition']}")


# ---------------------------------------------------------------- freezing

def graph_hashes(adj):
    nodes = sorted(adj.keys())[:20000]
    edges = "\n".join(f"{n}>" + ",".join(sorted(x[0] for x in adj.get(n, []))) for n in nodes)
    degs = sorted(len(adj.get(n, [])) for n in nodes)
    return (hashlib.sha256(edges.encode()).hexdigest()[:16],
            hashlib.sha256(str(degs).encode()).hexdigest()[:16])


def freeze_agent_ids(lse_dir, out_root, n=150, seed=SEED):
    """Draw the agent evaluation set from the frozen screening set. NO INFERENCE HAPPENS HERE.

    Drawn from the SCREEN split only -- the split the §4.3 launch gate was decided on. The DEV
    split is reserved for smoke runs and is structurally disjoint (whole (year, category) clusters
    are assigned to one side or the other), so no smoke instance can appear in the agent set.
    """
    frozen = [r for r in read_jsonl(Path(lse_dir) / "frozen_ids.jsonl") if r["split"] == "screen"]
    by = {r["source_id"]: r for r in frozen}
    ids = sorted(by)
    rr = random.Random(seed)
    rr.shuffle(ids)
    pick = sorted(ids[:n])
    rows = [dict(source_id=i, instance_id=i, as_of=by[i]["as_of"], split=by[i]["split"],
                 cluster=by[i]["cluster"], primary_category=by[i]["primary_category"],
                 title=by[i]["title"], query=by[i]["query"],
                 gold=by[i]["gold"], n_gold=by[i]["n_gold"],
                 unresolved_rate=by[i]["unresolved_rate"]) for i in pick]
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "agent_frozen_ids.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    idsonly = out / "agent_frozen_ids_only.txt"
    idsonly.write_text("\n".join(pick) + "\n")
    man = dict(
        name="litexp_agent_frozen_ids", task=TASK_NAME, protocol=PROTOCOL,
        generated_at=now_iso(), git_commit=commit(), seed=seed, n=len(rows),
        drawn_from=str(Path(lse_dir) / "frozen_ids.jsonl"),
        selection=f"seeded shuffle (seed {seed}) of the frozen SCREEN-split source ids, first {n}",
        frozen_before_any_inference=True,
        dev_slice=("the DEV split of the same frozen file; whole (year, primary category) "
                   "clusters are assigned to dev or screen, so the smoke slice is structurally "
                   "disjoint from these ids"),
        declared_primary_comparison=(f"{PRIMARY_COMPARISON[0]} > {PRIMARY_COMPARISON[1]} on "
                                     f"{PRIMARY_COMPARISON[2]}"),
        confirmatory_family=[f"{a} > {b} on {m}" for a, b, m in CONFIRMATORY_FAMILY],
        everything_else="secondary_unadjusted",
        conditions=CONDITIONS + [f"matched_pool_replay::{c}" for c in AGENTS],
        budgets=BUDGET, answer_k=ANSWER_K, score_cap=SCORE_CAP,
        scoring_rule=("model ranking, then add_to_review ids in add order, then remaining "
                      f"observed candidates in first-seen order, truncated to {SCORE_CAP}; "
                      "identical in every condition"),
        weak_label_note=WEAK_LABEL_NOTE, scoring_note=SCORING_NOTE,
        gold_size=dict(mean=round(sum(r["n_gold"] for r in rows) / max(1, len(rows)), 2),
                       min=min((r["n_gold"] for r in rows), default=0),
                       max=max((r["n_gold"] for r in rows), default=0)),
        hashes=dict(agent_ids_sha256=sha256_list(pick),
                    agent_frozen_ids_file_sha256=hashlib.sha256(
                        open(path, "rb").read()).hexdigest(),
                    agent_ids_only_file_sha256=hashlib.sha256(
                        open(idsonly, "rb").read()).hexdigest()))
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
                    help="e.g. .../literature_set_expansion/agent_results")
    ap.add_argument("--lse-dir", default=None,
                    help="literature_set_expansion dir holding frozen_ids.jsonl "
                         "(default: parent of --out-root)")
    ap.add_argument("--ids", default=None,
                    help="agent_frozen_ids.jsonl (default: inside --out-root)")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--shard", default="0/1", help="i/n - split instances across servers")
    ap.add_argument("--smoke", type=int, default=0,
                    help="run N DEV-split instances (never scored as an evaluation result)")
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--freeze", action="store_true",
                    help="write agent_frozen_ids.jsonl + hashes and exit (no inference)")
    ap.add_argument("--freeze-n", type=int, default=150)
    a = ap.parse_args()

    root = Path(a.out_root)
    lse_dir = Path(a.lse_dir) if a.lse_dir else root.parent
    if a.freeze:
        rows, man = freeze_agent_ids(lse_dir, root, n=a.freeze_n)
        print(json.dumps({k: man[k] for k in ("n", "hashes", "gold_size",
                                              "declared_primary_comparison")}, indent=1))
        return 0

    si, sn = (int(x) for x in a.shard.split("/"))
    assert 0 <= si < sn, "--shard must be i/n with 0 <= i < n"
    tag = f"s{si}of{sn}"

    if a.smoke:
        # DEV slice: whole (year, category) clusters disjoint from the screen split the frozen
        # agent ids are drawn from. Never scored as an evaluation result.
        dev = [r for r in read_jsonl(lse_dir / "frozen_ids.jsonl") if r["split"] == "dev"]
        dev.sort(key=lambda r: r["source_id"])
        random.Random(SEED + 7).shuffle(dev)
        insts = sorted(dev[:a.smoke], key=lambda r: r["source_id"])
    else:
        idspath = Path(a.ids) if a.ids else root / "agent_frozen_ids.jsonl"
        insts = [json.loads(l) for l in open(idspath)]
    frozen_ids = {r["source_id"] for r in insts}
    if a.smoke:
        agent_path = Path(a.ids) if a.ids else root / "agent_frozen_ids.jsonl"
        if agent_path.exists():
            evalset = {json.loads(l)["source_id"] for l in open(agent_path)}
            assert not (frozen_ids & evalset), \
                "smoke slice must be disjoint from the frozen agent evaluation ids"
    mine = [x for j, x in enumerate(insts) if j % sn == si]
    print(f"instances={len(insts)} shard={tag} mine={len(mine)}", flush=True)
    if not mine:
        print("nothing to do for this shard")
        return 0

    print("loading corpus + base-normalised graphs (full, rewire) ...", flush=True)
    from .run_agent import load_literature_corpus
    papers, bm, _a, _p = load_literature_corpus(())
    adj, para_owner, ident = load_graphs_base_normalised(("full", "rewire"))
    deg = degree_equality_check(adj["full"], adj["rewire"])
    assert deg["out_degree_sequences_identical"] and deg["in_degree_sequences_identical"], \
        "rewired variant is not degree-preserving after base-id merging"
    owned = defaultdict(set)
    for para, owner in para_owner.items():
        owned[base_pid(owner)].add(para)
    hs = {v: graph_hashes(adj[v]) for v in ("full", "rewire")}
    ghash = {v: hs[v][0] for v in hs}
    dhash = {v: hs[v][1] for v in hs}
    assert ghash["full"] != ghash["rewire"], "correct and rewired graphs must differ"
    assert dhash["full"] == dhash["rewire"], "degree histograms must match"
    print("graph hashes:", hs, flush=True)
    print("identity merge:", json.dumps(ident), flush=True)

    # correct and rewired must open with byte-identical prompts and tool docs
    probe = mine[0]
    ha = h16(SYSTEM + "||" + opening_prompt(probe, TOOLS_FOR["correct_snowballing_agent"],
                                            REQUIRED_FOR["correct_snowballing_agent"]) + "||"
             + "|".join(TOOLS_DOC[t] for t in TOOLS_FOR["correct_snowballing_agent"]))
    hb = h16(SYSTEM + "||" + opening_prompt(probe, TOOLS_FOR["rewired_snowballing_agent"],
                                            REQUIRED_FOR["rewired_snowballing_agent"]) + "||"
             + "|".join(TOOLS_DOC[t] for t in TOOLS_FOR["rewired_snowballing_agent"]))
    assert ha == hb, "correct and rewired prompts/tool schemas must be byte-identical"
    assert BUDGET["max_tool_calls"] > 0
    print("prompt_schema_hash (correct == rewired):", ha, flush=True)

    sub = "smoke" if a.smoke else "episodes"
    outdir = root / sub / a.model
    repdir = root / ("matched_pools_smoke" if a.smoke else "matched_pools") / a.model
    outdir.mkdir(parents=True, exist_ok=True)
    repdir.mkdir(parents=True, exist_ok=True)

    cache = ResponseCache(root / f"cache_{a.model}.jsonl")
    pol = VLLMPolicy(a.model, a.base_url, cache=cache, max_tokens=900)
    pol.prompt_version = PROMPT_VERSION
    ctx = dict(papers=papers, bm25=FastBM25(bm), adjs=adj, owned=owned, rid=a.run_id,
               ghash=ghash, dhash=dhash)

    for cond in a.conditions:
        path = outdir / f"{cond}.{tag}.jsonl"
        rpath = repdir / f"matched_pool_replay::{cond}.{tag}.jsonl"
        # resume key is (model, condition, instance_id), checked across EVERY shard file of this
        # condition, so two workers can never write the same key even with overlapping shards
        done = _done_keys(outdir, cond)
        done_rep = _done_keys(repdir, f"matched_pool_replay::{cond}")
        n_new = 0
        for inst in mine:
            iid = inst["source_id"]
            if iid in done:
                continue
            row, _ep = run_episode(inst, cond, pol, ctx)
            assert row["resume_key"] == f"{a.model}|{cond}|{iid}"
            with open(path, "a") as f:                     # append-only, resumable by key
                f.write(json.dumps(row, default=str) + "\n")
            done.add(iid)
            if not a.no_replay and cond in AGENTS and iid not in done_rep:
                rrow = replay(row, inst, pol, ctx)
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
