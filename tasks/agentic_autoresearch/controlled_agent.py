"""controlled_interaction_v1: Task-A agent loop with a mandatory minimum research phase.

EXPLICITLY POST-HOC. Run `20260924_182602_a26e2e` (`natural_action`) stands unchanged as the
natural-interface result; `canonical_agent.py` is untouched so it stays reproducible.

Why this module exists: under the natural interface both checkpoints emitted a well-formed final
answer on turn 1 (2614/2614 cached generations), because the first user turn already carried a
10-candidate BM25 pool and the prompt's closing line demanded a 5-id submission. Tool use was never
necessary. See `tool_loop_root_cause.md`.

Fixes, per runbook 1.2-1.4:
  * separate action / final parsers; an unrecognized object is malformed, never a silent submit;
  * length-5 applies only to `final`;
  * no prepopulated candidate pool - agents open with task context and tool docs only;
  * a mandatory minimum research phase per condition, disclosed in every report;
  * one deterministic format-repair turn, then the episode ends invalid rather than as an answer;
  * anytime top-5 after each observation, stored at calls 1/2/4/8, never back-filled;
  * attempted / parsed / dispatched / succeeded / blocked / failed counted separately.

Correct and rewired conditions share byte-identical prompts and tool schemas; only the backing
adjacency differs. Asserted at runtime via `prompt_schema_hash`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time

from .canonical import current_run_id, provenance, run_dir
from .candidate_utility import bm25_ranking, bm25_seeds, _legal
from .common import DATA, read_jsonl
from .evaluate import rr, top1
from .hybrid_retrieval import graph_rank, rrf
from .policies import ResponseCache, VLLMPolicy, strip_think
from .temporal_store import normalize_ts

PROTOCOL = "controlled_interaction_v1"
PROMPT_VERSION = "ctrl-v1"
POOL_K, ANSWER_K = 20, 5
BUDGET = dict(max_tool_calls=8, max_observation_tokens=6000, per_call_candidates=10,
              snippet_chars=220, max_repairs=1, max_refusals=3, max_turns=16,
              max_context_tokens=13000)
ANYTIME_AT = (1, 2, 4, 8)

CONDITIONS = ["flat_controlled_agent", "correct_graph_controlled_agent",
              "rewired_graph_controlled_agent", "compact_hybrid_controlled_agent"]
# One-shot controls re-run under the ctrl-v1 prompt and parser. The natural-action run's one-shot
# results are NOT reused: runbook 2.1 permits reuse only when prompt, parser and answer semantics
# are identical, and ctrl-v1 answers are {"final":[...]} rather than {"tool":"submit",...}.
ONESHOT = ["bm25_one_shot_ctrl", "correct_hybrid_one_shot_ctrl"]
GRAPH_FOR = {"correct_graph_controlled_agent": "untyped",
             "rewired_graph_controlled_agent": "rewire"}
# tools per condition; correct and rewired are deliberately identical
TOOLS_FOR = {
    "flat_controlled_agent": ["search", "open"],
    "correct_graph_controlled_agent": ["search", "open", "traverse"],
    "rewired_graph_controlled_agent": ["search", "open", "traverse"],
    "compact_hybrid_controlled_agent": ["search", "open", "hybrid_search"],
}
# mandatory minimum research phase (runbook 1.3), disclosed in all reports
REQUIRED_FOR = {
    "flat_controlled_agent": {"search": 1},
    "correct_graph_controlled_agent": {"search": 1, "traverse": 1},
    "rewired_graph_controlled_agent": {"search": 1, "traverse": 1},
    "compact_hybrid_controlled_agent": {"hybrid_search": 1},
}
ACTION_VERBS = ("search", "open", "traverse", "hybrid_search")

SYSTEM = ("You are a research assistant identifying which paper was cited at a masked citation. "
          "You must investigate using the tools before answering. "
          "Reply with exactly one JSON object per turn and no other text.")

TOOLS_DOC = {
    "search": '{"action":"search","query":"<text>"} - lexical search over the literature',
    "open": '{"action":"open","paper_id":"<id>"} - read a paper\'s full record',
    "traverse": '{"action":"traverse","paper_id":"<id>","direction":"backward"} - papers connected to this one',
    "hybrid_search": '{"action":"hybrid_search","query":"<text>"} - combined lexical and structural search',
}

TURN_SPEC = (
    'Each turn reply with ONE JSON object, either an action:\n'
    '{"action":"<verb>", ...args..., "current_top5":["<id>",...]}\n'
    'or, once you have investigated enough, your final answer:\n'
    '{"final":["<id>","<id>","<id>","<id>","<id>"]}\n'
    '"current_top5" is your best ranking so far from ids you have actually seen; it may be empty '
    'on the first turn. Only "final" requires exactly 5 distinct ids.'
)


def ntok(s):
    return max(1, len(str(s)) // 4)


def h16(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:16]


def render(pool, papers, n=ANSWER_K * 4, snippet=BUDGET["snippet_chars"]):
    return "\n".join(f"id={p} | title={papers.get(p,{}).get('title')} | "
                     f"abstract={str(papers.get(p,{}).get('abstract'))[:snippet]}" for p in pool[:n])


def opening_prompt(inst, tools, required):
    """No candidate pool: the agent must research. Identical text for correct vs rewired."""
    need = ", ".join(f"at least {v} {k}" for k, v in sorted(required.items()))
    return "\n\n".join([
        f"Cutoff: {inst['as_of']}",
        f"Source paper: {inst.get('source_title')}",
        f"Paragraph with a removed citation:\n{inst['masked_text'][:2200]}",
        "Available tools (one action per turn):\n" + "\n".join(f"- {TOOLS_DOC[t]}" for t in tools),
        f"Required minimum investigation before answering: {need}.",
        TURN_SPEC,
    ])


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


def parse_final(obj, allowed):
    """Final answers only. Requires exactly ANSWER_K distinct ids the agent actually observed."""
    ids = obj.get("final") if isinstance(obj, dict) else None
    if not isinstance(ids, list):
        return [], "final_not_a_list"
    seen = [str(x) for x in ids if str(x) in allowed]
    seen = list(dict.fromkeys(seen))
    if not seen:
        return [], "no_valid_ids"
    return seen[:ANSWER_K], (None if len(seen) >= ANSWER_K else "short_answer")


def parse_top5(obj, allowed):
    v = obj.get("current_top5") if isinstance(obj, dict) else None
    if not isinstance(v, list):
        return []
    out = [str(x) for x in v if str(x) in allowed]
    return list(dict.fromkeys(out))[:ANSWER_K]


class Ep:
    def __init__(self, inst, cond, model, rid, graph_hash, degree_hash, schema_hash):
        self.inst, self.cond, self.model, self.rid = inst, cond, model, rid
        self.graph_hash, self.degree_hash, self.schema_hash = graph_hash, degree_hash, schema_hash
        self.calls, self.obs_tokens, self.truncations = [], 0, 0
        self.observed, self.opened = [], []
        self.attempted = self.parsed = self.dispatched = 0
        self.succeeded = self.blocked = self.failed = self.malformed = 0
        self.repairs = self.refusals = 0
        self.forced_final = False
        self.uptake = {v: 0 for v in ACTION_VERBS}
        self.anytime = {}
        self.calls_to_first_gold = None
        self.t0 = time.time()

    def add_obs(self, ids, payload):
        t = ntok(payload)
        if self.obs_tokens + t > BUDGET["max_observation_tokens"]:
            self.truncations += 1
            return False
        self.obs_tokens += t
        for i in ids:
            if i not in self.observed:
                self.observed.append(i)
        return True

    def note_gold(self, gold):
        if self.calls_to_first_gold is None and gold in self.observed:
            self.calls_to_first_gold = len(self.calls)

    def pad(self, ranked, pool):
        out = list(dict.fromkeys(ranked))
        for c in pool:
            if len(out) >= ANSWER_K:
                break
            if c not in out:
                out.append(c)
        return out[:ANSWER_K], len(out) - len(dict.fromkeys(ranked))

    def met(self, ranked, gold, ok):
        g = gold[0]
        return dict(top1=top1(ranked, gold) if ok else 0.0,
                    mrr=rr(ranked, gold) if ok else 0.0,
                    recall_at5=float(g in ranked[:ANSWER_K]) if ok else 0.0,
                    valid=float(ok))

    def row(self, ranked, problem, gold, faithful, cited, padded=0):
        g = gold[0]
        ok = problem is None
        return dict(run_id=self.rid, commit=provenance()["git_commit"], protocol=PROTOCOL,
                    condition=self.cond, model=self.model, prompt_version=PROMPT_VERSION,
                    instance_id=self.inst["instance_id"], as_of=self.inst["as_of"],
                    graph_variant_hash=self.graph_hash, graph_degree_hash=self.degree_hash,
                    prompt_schema_hash=self.schema_hash, seed=20260925,
                    tool_calls=self.calls, n_tool_calls=len(self.calls),
                    calls_attempted=self.attempted, calls_parsed=self.parsed,
                    calls_dispatched=self.dispatched, calls_succeeded=self.succeeded,
                    blocked_calls=self.blocked, failed_calls=self.failed,
                    malformed_actions=self.malformed, repairs=self.repairs,
                    premature_finalize_refusals=self.refusals,
                    forced_final=self.forced_final,
                    uptake=dict(self.uptake),
                    observed_candidates=self.observed, opened=self.opened,
                    observation_tokens=self.obs_tokens, truncation_events=self.truncations,
                    matched_pool_hash=h16("|".join(self.observed)),
                    anytime=self.anytime, calls_to_first_gold=self.calls_to_first_gold,
                    ranked=ranked, cited_evidence=cited, gold=gold,
                    gold_observed=g in self.observed, gold_opened=g in self.opened,
                    gold_rank=(ranked.index(g) + 1 if g in ranked else 0),
                    parser_problem=problem, valid=ok, padded_ids=padded,
                    evidence_faithful=faithful,
                    latency_ms=int((time.time() - self.t0) * 1000),
                    metrics=self.met(ranked, gold, ok))


def _dispatch(verb, arg, inst, ctx, cond, masked):
    """Execute one action. Returns (ids, payload, error)."""
    papers, bm, adjs = ctx["papers"], ctx["bm"], ctx["adjs"]
    cutoff = normalize_ts(inst["as_of"])
    n = BUDGET["per_call_candidates"]
    if verb == "search":
        q = dict(inst, masked_text=str(arg.get("query", ""))[:400])
        ids = bm25_ranking(q, bm, papers, k=n)
        return ids, render(ids, papers, n=n), None
    if verb == "hybrid_search":
        q = dict(inst, masked_text=str(arg.get("query", ""))[:400])
        gl = graph_rank(q, bm25_seeds(q, bm, papers), adjs["untyped"], papers, masked, k=POOL_K)[0]
        bl = bm25_ranking(q, bm, papers, k=POOL_K)
        ids = rrf([bl, gl], budget=n)
        return ids, render(ids, papers, n=n), None
    if verb == "traverse":
        node = f"paper:{str(arg.get('paper_id','')).replace('paper:','')}"
        gv = GRAPH_FOR.get(cond)
        if gv is None:            # belt-and-braces: a graph-free condition has no adjacency
            return [], {"status": "BLOCKED", "reason": "traverse unavailable in this condition"}, None
        nb = [x for x, _r, _d in adjs[gv].get(node, []) if x.startswith("paper:")]
        ids = [x.split(":", 1)[1] for x in nb
               if _legal(x.split(":", 1)[1], papers, cutoff) and (node, x) not in masked][:n]
        if not ids:
            return [], {"status": "EMPTY", "reason": "no connected papers available by cutoff"}, None
        return ids, render(ids, papers, n=n), None
    if verb == "open":
        pid = str(arg.get("paper_id", "")).replace("paper:", "")
        if pid in papers and _legal(pid, papers, cutoff):
            return [pid], render([pid], papers, snippet=800), None
        return [], {"status": "ERROR", "reason": "unknown_or_future_id"}, "unknown_or_future_id"
    return [], {"status": "BLOCKED", "reason": f"tool {verb!r} unavailable in this condition"}, None


def run_oneshot(inst, cond, pol, ctx):
    """Fixed-list control: no tools, ctrl-v1 answer format, so it is comparable to the agents."""
    papers, bm, adjs = ctx["papers"], ctx["bm"], ctx["adjs"]
    gold = [inst["gold_target_id"]]
    g = gold[0]
    masked = {(f"paper:{inst['source_paper_id']}", f"paper:{g}"),
              (f"paper:{g}", f"paper:{inst['source_paper_id']}"),
              (f"para:{inst.get('paragraph_global_id')}", f"paper:{g}"),
              (f"paper:{g}", f"para:{inst.get('paragraph_global_id')}")}
    bl = bm25_ranking(inst, bm, papers, k=POOL_K)
    if cond == "bm25_one_shot_ctrl":
        pool = bl
    else:
        gl = graph_rank(inst, bm25_seeds(inst, bm, papers), adjs["untyped"], papers, masked, k=POOL_K)[0]
        pool = rrf([bl, gl], budget=POOL_K)
    ep = Ep(inst, cond, pol.model, ctx["rid"], "none", "none", h16(cond))
    ep.add_obs(pool[:ANSWER_K * 4], render(pool, papers))
    row = replay(ep.row([], "placeholder", gold, None, []), inst, pol, papers)
    return dict(row, condition=cond, observed_candidates=ep.observed,
                matched_pool_hash=h16("|".join(ep.observed))), None


def run_episode(inst, cond, pol, ctx):
    if cond in ONESHOT:
        return run_oneshot(inst, cond, pol, ctx)
    papers, rid = ctx["papers"], ctx["rid"]
    gold = [inst["gold_target_id"]]
    g = gold[0]
    masked = {(f"paper:{inst['source_paper_id']}", f"paper:{g}"),
              (f"paper:{g}", f"paper:{inst['source_paper_id']}"),
              (f"para:{inst.get('paragraph_global_id')}", f"paper:{g}"),
              (f"paper:{g}", f"para:{inst.get('paragraph_global_id')}")}
    tools, required = TOOLS_FOR[cond], REQUIRED_FOR[cond]
    gv = GRAPH_FOR.get(cond, "none")
    prompt = opening_prompt(inst, tools, required)
    # correct vs rewired must be byte-identical here; only the adjacency behind `traverse` differs
    schema_hash = h16(prompt + "||" + "|".join(TOOLS_DOC[t] for t in tools))
    ep = Ep(inst, cond, pol.model, rid, ctx["ghash"].get(gv, "none"),
            ctx["dhash"].get(gv, "none"), schema_hash)

    msgs = [dict(role="system", content=SYSTEM), dict(role="user", content=prompt)]
    ranked, problem, final_obj = [], "budget_exhausted", None

    for _turn in range(BUDGET["max_turns"]):
        if len(ep.calls) >= BUDGET["max_tool_calls"]:
            problem = "budget_exhausted"
            break
        if sum(ntok(m["content"]) for m in msgs) > BUDGET["max_context_tokens"]:
            # deterministic, condition-independent: stop acting and take the forced-answer turn
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
                         dict(role="user", content=(
                             "FORMAT ERROR: reply with exactly one JSON object and nothing else. "
                             + TURN_SPEC))]
                continue
            problem = "parse_failed"          # never silently becomes a final answer
            break
        ep.parsed += 1

        if verb == "finalize":
            missing = {k: v for k, v in required.items() if ep.uptake.get(k, 0) < v}
            if missing and ep.refusals < BUDGET["max_refusals"]:
                ep.refusals += 1                      # does not consume a tool call
                need = ", ".join(f"at least {v} {k}" for k, v in sorted(missing.items()))
                msgs += [dict(role="assistant", content=str(raw)[:400]),
                         dict(role="user", content=(
                             f"You have not completed the required investigation: {need}. "
                             "Issue that action now."))]
                continue
            final_obj = obj
            break

        if verb not in tools:
            # Refuse BEFORE dispatching. Dispatching first would execute the tool and feed its
            # candidates into `observed`, so a graph-free condition emitting `hybrid_search` or
            # `traverse` would silently receive graph-derived candidates. Never returns ids.
            ep.blocked += 1
            ids, payload, err = [], {"status": "BLOCKED",
                                     "reason": f"tool {verb!r} unavailable in this condition"}, None
            ep.dispatched += 1
            ok = ep.add_obs(ids, payload)
            ep.calls.append(dict(tool=verb, arguments=obj if isinstance(obj, dict) else {},
                                 returned_ids=[], truncated=not ok, error=None, blocked=True,
                                 latency_ms=0))
            msgs += [dict(role="assistant", content=json.dumps(
                        {k: v for k, v in obj.items() if k != "current_top5"})),
                     dict(role="user", content="OBSERVATION:\n" + str(payload))]
            continue
        ids, payload, err = _dispatch(verb, obj, inst, ctx, cond, masked)
        ep.dispatched += 1
        if err:
            ep.failed += 1
        else:
            ep.succeeded += 1
            ep.uptake[verb] = ep.uptake.get(verb, 0) + 1
        ok = ep.add_obs(ids, payload)
        ep.calls.append(dict(tool=verb, arguments=obj if isinstance(obj, dict) else {},
                             returned_ids=ids, truncated=not ok, error=err,
                             blocked=verb not in tools,
                             latency_ms=0))
        ep.note_gold(g)
        top5 = parse_top5(obj, set(ep.observed))
        if len(ep.calls) in ANYTIME_AT:
            ep.anytime[str(len(ep.calls))] = dict(
                top5=top5, gold_observed=g in ep.observed,
                metrics=ep.met(top5, gold, bool(top5)))
        msgs += [dict(role="assistant", content=json.dumps(
                    {k: v for k, v in obj.items() if k != "current_top5"})),
                 dict(role="user", content="OBSERVATION:\n"
                      + (str(payload) if ok else "[TRUNCATED: observation budget exhausted]"))]

    if final_obj is None and problem == "budget_exhausted" and ep.observed:
        # The agent spent its whole tool budget and was never given a turn to answer. Give exactly
        # one forced-answer turn, identically in every condition, so a researching agent is not
        # scored zero purely for running out of calls. Costs no tool call and adds no candidates.
        ep.attempted += 1
        raw = pol._chat(msgs + [dict(role="user", content=(
            "Your tool budget is exhausted. Reply now with ONE JSON object and nothing else: "
            '{"final":["<id>","<id>","<id>","<id>","<id>"]} using EXACTLY 5 distinct ids you have '
            "already seen."))])
        _v, obj, perr = parse_action(raw)
        if not perr and isinstance(obj, dict) and "final" in obj:
            final_obj, problem = obj, None
            ep.forced_final = True

    padded, cited, faithful = 0, [], None
    if final_obj is not None:
        ranked, problem = parse_final(final_obj, set(ep.observed))
        cited = list(ranked)
        faithful = all(x in ep.observed for x in cited) if cited else None
        if ranked and problem == "short_answer":
            ranked, padded = ep.pad(ranked, ep.observed)
            problem = None
    return ep.row(ranked, problem, gold, faithful, cited, padded), ep


def replay(row, inst, pol, papers):
    """One-shot selector over exactly the ordered unique candidates the agent observed."""
    pool = row["observed_candidates"][:ANSWER_K * 4]
    base = dict(row, condition=f"matched_pool_replay::{row['condition']}",
                matched_pool_hash=h16("|".join(pool)), tool_calls=[], n_tool_calls=0, anytime={})
    if not pool:
        return dict(base, ranked=[], valid=False, parser_problem="empty_pool",
                    metrics=dict(top1=0.0, mrr=0.0, recall_at5=0.0, valid=0.0))
    body = render(pool, papers)
    p = "\n\n".join([f"Cutoff: {inst['as_of']}", f"Source paper: {inst.get('source_title')}",
                     f"Paragraph with a removed citation:\n{inst['masked_text'][:2200]}",
                     f"Candidates:\n{body}",
                     'Reply with ONE JSON object: {"final":["<id>","<id>","<id>","<id>","<id>"]} '
                     'using EXACTLY 5 distinct ids from the candidates.'])
    raw = strip_think(pol._chat([dict(role="system", content=SYSTEM), dict(role="user", content=p)]) or "")
    try:
        obj = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except Exception:
        obj = {}
    ranked, prob = parse_final(obj, set(pool))
    if ranked and prob == "short_answer":
        ranked = list(dict.fromkeys(ranked + [c for c in pool if c not in ranked]))[:ANSWER_K]
        prob = None
    g = row["gold"][0]
    ok = prob is None
    return dict(base, ranked=ranked, valid=ok, parser_problem=prob,
                gold_rank=(ranked.index(g) + 1 if g in ranked else 0),
                metrics=dict(top1=top1(ranked, [g]) if ok else 0.0,
                             mrr=rr(ranked, [g]) if ok else 0.0,
                             recall_at5=float(g in ranked[:ANSWER_K]) if ok else 0.0,
                             valid=float(ok)))


def graph_hashes(gr):
    nodes = sorted(gr.keys())[:5000]
    edges = "\n".join(f"{n}>" + ",".join(sorted(x[0] for x in gr.get(n, []))) for n in nodes)
    degs = sorted(len(gr.get(n, [])) for n in nodes)
    return (hashlib.sha256(edges.encode()).hexdigest()[:16],
            hashlib.sha256(str(degs).encode()).hexdigest()[:16])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-root", required=True, help="master dir, e.g. .../controlled_task_a")
    ap.add_argument("--ids", default=None, help="frozen ids jsonl")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n - split instances across servers")
    ap.add_argument("--no-replay", action="store_true")
    a = ap.parse_args()
    from pathlib import Path
    from .run_agent import load_literature_corpus

    papers, bm, adj, _ = load_literature_corpus(("untyped", "rewire"))
    hs = {k: graph_hashes(adj[k]) for k in ("untyped", "rewire")}
    ghash, dhash = {k: v[0] for k, v in hs.items()}, {k: v[1] for k, v in hs.items()}
    assert ghash["untyped"] != ghash["rewire"], "correct and rewired graphs must differ"
    assert dhash["untyped"] == dhash["rewire"], "degree histograms must match"
    print("graph hashes:", hs, flush=True)

    allv = [r for r in read_jsonl(DATA / "literature_evidence" / "a1_instances.jsonl")
            if r["split"] == "test" and not r.get("source_version_unverifiable", False)]
    random.Random(20260922).shuffle(allv)
    if a.smoke:
        insts = allv[200:200 + a.smoke]                       # disjoint dev slice, never evaluated
    else:
        keep = [json.loads(l)["instance_id"] for l in open(a.ids)]
        idx = {r["instance_id"]: r for r in allv}
        insts = [idx[i] for i in keep]
    si, sn = (int(x) for x in a.shard.split("/"))
    insts = [x for j, x in enumerate(insts) if j % sn == si]

    root = Path(a.out_root)
    sub = "smoke" if a.smoke else "episodes"
    outdir, repdir = root / sub / a.model, root / ("matched_pools_smoke" if a.smoke else "matched_pools") / a.model
    outdir.mkdir(parents=True, exist_ok=True)
    repdir.mkdir(parents=True, exist_ok=True)

    cache = ResponseCache(root / f"cache_{a.model}.jsonl")
    pol = VLLMPolicy(a.model, a.base_url, cache=cache, max_tokens=640)
    pol.prompt_version = PROMPT_VERSION
    ctx = dict(papers=papers, bm=bm, adjs=adj, rid=a.run_id, ghash=ghash, dhash=dhash)

    for cond in a.conditions:
        path = outdir / f"{cond}.s{si}.jsonl"
        rpath = repdir / f"{cond}.s{si}.jsonl"
        done = {json.loads(l)["instance_id"] for l in open(path)} if path.exists() else set()
        n_new = 0
        for i, inst in enumerate(insts, 1):
            if inst["instance_id"] in done:
                continue
            row, ep = run_episode(inst, cond, pol, ctx)
            with open(path, "a") as f:                        # append-only, resumable by key
                f.write(json.dumps(row) + "\n")
            if not a.no_replay:
                with open(rpath, "a") as f:
                    f.write(json.dumps(replay(row, inst, pol, papers)) + "\n")
            n_new += 1
            if n_new % 10 == 0:
                print(f"  {a.model} {cond} shard{si} {n_new} new", flush=True)
        tot = n_new + len(done)
        print(f"[{a.model}/{cond}/s{si}] n={tot} new={n_new}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
