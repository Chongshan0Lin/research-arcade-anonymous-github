"""Part II: canonical Task-A agentic experiment (C1-C6 + matched-pool replay).

Six conditions on one frozen 200-instance set, identical budgets/prompts/parser:
  C1 bm25_one_shot              fixed BM25 list, no tools
  C2 correct_hybrid_one_shot    fixed BM25+graph (RRF) list, no tools
  C3 flat_search_agent          search/open tools only; graph calls BLOCKED and logged
  C4 correct_connectivity_agent flat tools + untyped traversal on the correct graph
  C5 rewired_connectivity_agent identical to C4, degree-preserving rewired adjacency only
  C6 compact_hybrid_agent       flat tools + hybrid_search (graph used for candidate generation,
                                no raw relation labels exposed)
Plus matched_pool_replay::<cond>: a one-shot selector over exactly the candidates the agent
actually observed, to separate sequential behaviour from simply seeing a better pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict

from .canonical import current_run_id, provenance, run_dir
from .candidate_utility import bm25_ranking, bm25_seeds, _legal
from .common import DATA, RESULTS, read_jsonl, write_json, write_jsonl
from .evaluate import paired_bootstrap_ci, paired_permutation_p, holm, rr, top1
from .hybrid_retrieval import graph_rank, rrf
from .policies import ResponseCache, VLLMPolicy, strip_think
from .temporal_store import normalize_ts

PROMPT_VERSION = "canon-v1"
POOL_K, ANSWER_K = 20, 5
BUDGET = dict(max_tool_calls=8, max_observation_tokens=6000, per_call_candidates=10,
              snippet_chars=220, max_retries=1)
CONDITIONS = ["bm25_one_shot", "correct_hybrid_one_shot", "flat_search_agent",
              "correct_connectivity_agent", "rewired_connectivity_agent", "compact_hybrid_agent"]
AGENT_CONDS = CONDITIONS[2:]
GRAPH_FOR = {"correct_connectivity_agent": "untyped", "rewired_connectivity_agent": "rewire"}

SYSTEM = ("You identify which paper was cited at a masked citation. Use only the candidates you "
          "are shown or retrieve. Reply with exactly one JSON object and no other text.")
ANSWER_SPEC = ('Reply with ONE JSON object: {"tool": "submit", "ranked_paper_ids": '
               '["<id>","<id>","<id>","<id>","<id>"]} using EXACTLY 5 distinct ids you have seen.')
TOOLS_DOC = {
    "search": 'search the literature: {"tool":"search","query":"<text>"}',
    "open": 'read a candidate: {"tool":"open","paper_id":"<id>"}',
    "neighbors": 'connected papers: {"tool":"neighbors","paper_id":"<id>"}',
    "hybrid_search": 'combined lexical+structural search: {"tool":"hybrid_search","query":"<text>"}',
}


def ntok(s):
    return max(1, len(str(s)) // 4)


def pool_hash(ids):
    return hashlib.sha256("|".join(ids).encode()).hexdigest()[:16]


def render(pool, papers, n=ANSWER_K * 4, snippet=BUDGET["snippet_chars"]):
    return "\n".join(f"id={p} | title={papers.get(p,{}).get('title')} | "
                     f"abstract={str(papers.get(p,{}).get('abstract'))[:snippet]}" for p in pool[:n])


def task_prompt(inst, body, tools=None):
    parts = [f"Cutoff: {inst['as_of']}", f"Source paper: {inst.get('source_title')}",
             f"Paragraph with a removed citation:\n{inst['masked_text'][:2200]}"]
    if tools:
        parts.append("Available tools (one per turn):\n" + "\n".join(f"- {TOOLS_DOC[t]}" for t in tools))
    if body:
        parts.append(f"Candidates:\n{body}")
    parts.append(ANSWER_SPEC)
    return "\n\n".join(parts)


def parse(raw, allowed):
    raw = strip_think(raw or "")
    try:
        obj = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except Exception:
        return None, None, "parser_failure"
    tool = obj.get("tool")
    if tool in ("search", "open", "neighbors", "hybrid_search"):
        return tool, obj, None
    ids = obj.get("ranked_paper_ids") or []
    uniq = [str(x) for x in ids if str(x) in allowed]
    uniq = list(dict.fromkeys(uniq))
    if not uniq:
        return "submit", [], "no_valid_ids"
    return "submit", uniq[:ANSWER_K], (None if len(uniq) >= ANSWER_K else "short_answer")


class Ep:
    """One episode: enforces budgets online and logs everything the runbook requires."""

    def __init__(self, inst, cond, model, rid, graph_hash):
        self.inst, self.cond, self.model, self.rid = inst, cond, model, rid
        self.graph_hash = graph_hash
        self.calls, self.obs_tokens, self.truncations = [], 0, 0
        self.observed, self.opened = [], []
        self.blocked = self.errors = 0
        self.degree_hash = "none"
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

    def pad(self, ranked, pool):
        """Enforce exactly ANSWER_K ids so conditions are never compared at unequal list length."""
        out = list(dict.fromkeys(ranked))
        for c in pool:
            if len(out) >= ANSWER_K:
                break
            if c not in out:
                out.append(c)
        return out[:ANSWER_K], len(out) - len(dict.fromkeys(ranked))

    def row(self, ranked, problem, gold, faithful, cited, padded=0):
        g = gold[0]
        return dict(run_id=self.rid, commit=provenance()["git_commit"], condition=self.cond,
                    model=self.model, prompt_version=PROMPT_VERSION, instance_id=self.inst["instance_id"],
                    as_of=self.inst["as_of"], graph_variant_hash=self.graph_hash,
                    graph_degree_hash=self.degree_hash,
                    seed=20260922, tool_calls=self.calls,
                    n_tool_calls=len(self.calls), blocked_calls=self.blocked, tool_errors=self.errors,
                    observed_candidates=self.observed, opened=self.opened,
                    observation_tokens=self.obs_tokens, truncation_events=self.truncations,
                    matched_pool_hash=pool_hash(self.observed),
                    ranked=ranked, cited_evidence=cited, gold=gold,
                    gold_observed=g in self.observed, gold_opened=g in self.opened,
                    gold_rank=(ranked.index(g) + 1 if g in ranked else 0),
                    parser_problem=problem, valid=problem is None, padded_ids=padded,
                    evidence_faithful=faithful, latency_ms=int((time.time() - self.t0) * 1000),
                    metrics=dict(top1=top1(ranked, gold) if problem is None else 0.0,
                                 mrr=rr(ranked, gold) if problem is None else 0.0,
                                 recall_at5=float(g in ranked[:ANSWER_K]) if problem is None else 0.0,
                                 valid=float(problem is None)))


def run_episode(inst, cond, pol, ctx):
    papers, bm, adjs, rid, ghash = ctx["papers"], ctx["bm"], ctx["adjs"], ctx["rid"], ctx["ghash"]
    gold = [inst["gold_target_id"]]
    masked = {(f"paper:{inst['source_paper_id']}", f"paper:{gold[0]}"),
              (f"paper:{gold[0]}", f"paper:{inst['source_paper_id']}"),
              (f"para:{inst.get('paragraph_global_id')}", f"paper:{gold[0]}"),
              (f"paper:{gold[0]}", f"para:{inst.get('paragraph_global_id')}")}
    ep = Ep(inst, cond, pol.model, rid, ghash.get(GRAPH_FOR.get(cond, "none"), "none"))
    ep.degree_hash = ctx["dhash"].get(GRAPH_FOR.get(cond, "none"), "none")
    bm_list = bm25_ranking(inst, bm, papers, k=POOL_K)
    seeds = bm25_seeds(inst, bm, papers)
    hyb = rrf([bm_list, graph_rank(inst, seeds, adjs["untyped"], papers, masked, k=POOL_K)[0]],
              budget=POOL_K)

    if cond in ("bm25_one_shot", "correct_hybrid_one_shot"):
        pool = bm_list if cond == "bm25_one_shot" else hyb
        ep.add_obs(pool[:ANSWER_K * 4], render(pool, papers))
        tool, ans, problem = parse(pol._chat([dict(role="system", content=SYSTEM),
                                              dict(role="user", content=task_prompt(inst, render(pool, papers)))]),
                                   set(pool))
        ranked = ans if tool == "submit" and isinstance(ans, list) else []
        cited = list(ranked)
        padded = 0
        if ranked and problem == "short_answer":
            ranked, padded = ep.pad(ranked, pool)
            problem = None
        return ep.row(ranked, problem or (None if ranked else "no_valid_ids"), gold,
                      all(x in ep.observed for x in cited) if cited else None, cited,
                      padded), None

    tools = ["search", "open"] + (["neighbors"] if cond in GRAPH_FOR else [])
    tools += ["hybrid_search"] if cond == "compact_hybrid_agent" else []
    opening = render(bm_list[:ANSWER_K * 2], papers)
    ep.add_obs(bm_list[:ANSWER_K * 2], opening)
    msgs = [dict(role="system", content=SYSTEM),
            dict(role="user", content=task_prompt(inst, opening, tools))]
    ranked, problem = [], "budget_exhausted"
    for _ in range(BUDGET["max_tool_calls"]):
        tool, arg, prob = parse(pol._chat(msgs), set(ep.observed))
        if tool == "submit":
            ranked, problem = (arg if isinstance(arg, list) else []), prob
            break
        t0, ids, payload, err = time.time(), [], "", None
        if tool not in tools:
            # any tool outside this condition's interface is refused, so a condition can never
            # reach the graph through a tool it was not granted
            ep.blocked += 1
            payload = {"status": "BLOCKED", "reason": f"tool {tool!r} unavailable in this condition"}
        elif tool == "search":
            hits = [p for p in bm25_ranking(dict(inst, masked_text=str(arg.get("query", ""))[:400]),
                                            bm, papers, k=BUDGET["per_call_candidates"])]
            ids, payload = hits, render(hits, papers, n=BUDGET["per_call_candidates"])
        elif tool == "hybrid_search":
            q = dict(inst, masked_text=str(arg.get("query", ""))[:400])
            sd = bm25_seeds(q, bm, papers)
            gl = graph_rank(q, sd, adjs["untyped"], papers, masked, k=POOL_K)[0]
            bl = bm25_ranking(q, bm, papers, k=POOL_K)
            ids = rrf([bl, gl], budget=BUDGET["per_call_candidates"])
            payload = render(ids, papers, n=BUDGET["per_call_candidates"])
            ep.calls.append(dict(tool="hybrid_search_channels", lexical_only=len(set(bl) - set(gl)),
                                 graph_only=len(set(gl) - set(bl)), overlap=len(set(bl) & set(gl))))
        elif tool == "neighbors":
            node = f"paper:{str(arg.get('paper_id','')).replace('paper:','')}"
            nb = [n for n, _r, _d in adjs[GRAPH_FOR[cond]].get(node, []) if n.startswith("paper:")]
            ids = [n.split(":", 1)[1] for n in nb
                   if (n, ) and _legal(n.split(":", 1)[1], papers, normalize_ts(inst["as_of"]))
                   and (node, n) not in masked][:BUDGET["per_call_candidates"]]
            payload = render(ids, papers, n=BUDGET["per_call_candidates"])
        elif tool == "open":
            pid = str(arg.get("paper_id", "")).replace("paper:", "")
            if pid in papers and _legal(pid, papers, normalize_ts(inst["as_of"])):
                ids, payload = [pid], render([pid], papers, snippet=800)
                ep.opened.append(pid)
            else:
                err = "unknown_or_future_id"
                ep.errors += 1
                payload = {"status": "ERROR", "reason": err}
        ok = ep.add_obs(ids, payload)
        ep.calls.append(dict(tool=tool, arguments=arg if isinstance(arg, dict) else {},
                             returned_ids=ids, truncated=not ok, error=err,
                             latency_ms=int((time.time() - t0) * 1000)))
        msgs += [dict(role="assistant", content=json.dumps({"tool": tool})),
                 dict(role="user", content=f"OBSERVATION:\n{payload if ok else '[TRUNCATED: budget]'}")]
    cited = list(ranked)
    faithful = all(x in ep.observed for x in cited) if cited else None
    padded = 0
    if ranked:
        ranked, padded = ep.pad(ranked, ep.observed + bm_list)
    return ep.row(ranked, problem if not cited else None, gold, faithful, cited, padded), ep


def replay(ep_row, inst, pol, papers):
    pool = ep_row["observed_candidates"][:ANSWER_K * 4]
    if not pool:
        return dict(ep_row, condition=f"matched_pool_replay::{ep_row['condition']}",
                    ranked=[], valid=False, parser_problem="empty_pool",
                    metrics=dict(top1=0.0, mrr=0.0, recall_at5=0.0, valid=0.0))
    tool, ans, prob = parse(pol._chat([dict(role="system", content=SYSTEM),
                                       dict(role="user", content=task_prompt(inst, render(pool, papers)))]),
                            set(pool))
    ranked = ans if tool == "submit" and isinstance(ans, list) else []
    if ranked and prob == "short_answer":
        ranked = list(dict.fromkeys(ranked + [c for c in pool if c not in ranked]))[:ANSWER_K]
        prob = None
    g = ep_row["gold"][0]
    return dict(ep_row, condition=f"matched_pool_replay::{ep_row['condition']}", ranked=ranked,
                valid=prob is None, parser_problem=prob, matched_pool_hash=pool_hash(pool),
                metrics=dict(top1=top1(ranked, [g]) if prob is None else 0.0,
                             mrr=rr(ranked, [g]) if prob is None else 0.0,
                             recall_at5=float(g in ranked[:ANSWER_K]) if prob is None else 0.0,
                             valid=float(prob is None)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--ids", default=None, help="frozen ids jsonl")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--smoke", type=int, default=0, help="use N disjoint dev instances instead")
    a = ap.parse_args()
    from .run_agent import load_literature_corpus

    rid = a.run_id or current_run_id(create=True)
    d = run_dir(rid, create=True)
    papers, bm, adj, _ = load_literature_corpus(("untyped", "rewire"))
    def graph_hashes(g):
        nodes = sorted(g.keys())[:5000]
        edges = "\n".join(f"{n}>" + ",".join(sorted(x[0] for x in g.get(n, []))) for n in nodes)
        degs = sorted(len(g.get(n, [])) for n in nodes)
        return (hashlib.sha256(edges.encode()).hexdigest()[:16],
                hashlib.sha256(str(degs).encode()).hexdigest()[:16])

    hashes = {k: graph_hashes(adj[k]) for k in ("untyped", "rewire")}
    ghash = {k: v[0] for k, v in hashes.items()}
    dhash = {k: v[1] for k, v in hashes.items()}
    print("graph hashes:", {k: hashes[k] for k in hashes}, flush=True)
    allv = [r for r in read_jsonl(DATA / "literature_evidence" / "a1_instances.jsonl")
            if r["split"] == "test" and not r.get("source_version_unverifiable", False)]
    random.Random(20260922).shuffle(allv)
    if a.smoke:
        insts = allv[200:200 + a.smoke]                      # disjoint dev slice
    else:
        keep = [json.loads(l)["instance_id"] for l in open(a.ids)]
        idx = {r["instance_id"]: r for r in allv}
        insts = [idx[i] for i in keep]
    cache = ResponseCache(d / f"cache_{a.model}.jsonl")
    pol = VLLMPolicy(a.model, a.base_url, cache=cache, max_tokens=512)
    pol.prompt_version = PROMPT_VERSION
    ctx = dict(papers=papers, bm=bm, adjs=adj, rid=rid, ghash=ghash, dhash=dhash)
    sub = "smoke" if a.smoke else "episodes"
    repsub = "matched_pools_smoke" if a.smoke else "matched_pools"
    outdir = d / sub / a.model
    outdir.mkdir(parents=True, exist_ok=True)
    (d / repsub / a.model).mkdir(parents=True, exist_ok=True)

    for cond in a.conditions:
        path = outdir / f"{cond}.jsonl"
        done = {json.loads(l)["instance_id"] for l in open(path)} if path.exists() else set()
        rows, reps = [], []
        for i, inst in enumerate(insts, 1):
            if inst["instance_id"] in done:
                continue
            row, ep = run_episode(inst, cond, pol, ctx)
            rows.append(row)
            if ep is not None:
                reps.append(replay(row, inst, pol, papers))
            if i % 25 == 0:
                print(f"  {a.model} {cond} {i}/{len(insts)}", flush=True)
        with open(path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        if reps:
            with open(d / repsub / a.model / f"{cond}.jsonl", "a") as f:
                for r in reps:
                    f.write(json.dumps(r) + "\n")
        n = len(rows) + len(done)
        v = sum(r["valid"] for r in rows) / max(1, len(rows))
        print(f"[{a.model}/{cond}] n={n} valid={v:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
