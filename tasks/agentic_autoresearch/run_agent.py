"""Bounded single-agent runs over paired instances and graph conditions.

Resumable and idempotent: a completed (instance_id, condition, model, prompt_version) is never
re-run unless --force is given. Full prompts/observations go to trajectories_*.jsonl; aggregate
metrics stay in predictions_*.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

from .baselines import load_instances
from .canonical import current_run_id, run_dir, write_run_metadata
from .common import DATA, RESULTS, db_conn, read_jsonl, write_json, write_manifest
from .environment import BM25, Episode, LiteratureEvidenceEnv, RTLocEnv
from .evaluate import metrics_for
from .policies import (ControlledGraphPolicy, MockPolicy, ResponseCache, VLLMPolicy,
                       build_system_prompt)
from .schemas import REQUIRED_ANSWER_LEN
from .schemas import CONDITION_VARIANT
from .temporal_store import TemporalObject, TemporalStore

PROMPT_VERSION = "v1"
RUN_ID_HOLDER = {}


# ---------------- data loading ----------------

def load_rtloc_graphs(variants):
    out = {}
    for v in variants:
        p = DATA / "rtloc" / "graphs" / f"rt_loc_{v}.jsonl"
        out[v] = {g["inst_id"]: g for g in read_jsonl(p)}
    return out


def load_literature_corpus(variants=("full",)):
    import pandas as pd
    conn = db_conn()
    df = pd.read_sql("select coalesce(base_arxiv_id, arxiv_id) as pid, title, abstract, "
                     "min(submit_date::text) as date from papers group by 1,2,3", conn)
    papers = {r.pid: dict(title=r.title, abstract=r.abstract, date=r.date) for r in df.itertuples()}
    bm = BM25({pid: f"{p['title']} {p['abstract']}" for pid, p in papers.items()})
    adj = {}
    for v in variants:
        d = defaultdict(list)
        for e in read_jsonl(DATA / "literature_evidence" / "graphs" / f"citation_graph_{v}.jsonl"):
            d[e["s"]].append((e["t"], e["r"], "out"))
            d[e["t"]].append((e["s"], e["r"], "in"))
        adj[v] = d
    # paragraph ownership always comes from the UNPERTURBED graph: it is ground truth about which
    # paper contains which paragraph, not a relation under test, and the auditor must see the same
    # node set in every condition
    para_owner = {}
    for e in read_jsonl(DATA / "literature_evidence" / "graphs" / "citation_graph_full.jsonl"):
        if e["r"] == "has_paragraph":
            para_owner[e["t"]] = e["s"].split(":", 1)[1]
    return papers, bm, adj, para_owner


def literature_store(papers, para_owner=None):
    st = TemporalStore()
    st.add_many([(f"paper:{pid}", "paper", p["date"]) for pid, p in papers.items()])
    # Register paragraph nodes so the auditor can catch future paragraphs. Only paragraphs whose
    # owning paper has a known date are registered; an unknown owner must stay "unknown" rather than
    # being reported as a violation (unknown != illegal).
    for para, owner in (para_owner or {}).items():
        d = papers.get(owner, {}).get("date")
        if d:
            st.add_many([(para, "paragraph", d)])
    return st


def rtloc_store(graphs_full):
    st = TemporalStore()
    for g in graphs_full.values():
        for nid, n in g["nodes"].items():
            st.add(TemporalObject(nid, n["type"], n.get("time", "1970-01-01")))
    return st


def paragraph_lookup_factory():
    conn = db_conn()

    def lookup(paper_id, rng):
        cur = conn.cursor()
        cur.execute("select id, paper_section, left(content, 1200) from paragraphs "
                    "where paper_arxiv_id like %s order by id limit 10", (paper_id + "%",))
        return [dict(paragraph_id=r[0], section=r[1], text=r[2]) for r in cur.fetchall()]
    return lookup


# ---------------- prompts ----------------

def user_prompt(inst, view="standard"):
    if inst["task"] == "literature_evidence":
        text = inst["delex_text"] if view == "delex" else inst["masked_text"]
        parts = [f"Cutoff (as_of): {inst['as_of']}",
                 f"Source paper title: {inst.get('source_title')}",
                 f"Section: {inst.get('section')}",
                 "A citation has been removed and replaced with [MASKED_CITATION].",
                 f"Paragraph:\n{text}"]
        if inst.get("prev_paragraph"):
            parts.append(f"Previous paragraph:\n{inst['prev_paragraph'][:1200]}")
        if inst.get("next_paragraph"):
            parts.append(f"Next paragraph:\n{inst['next_paragraph'][:1200]}")
        parts.append("Task: identify the paper that was cited at [MASKED_CITATION]. "
                     "You MUST finish with EXACTLY 5 distinct paper ids that you actually observed "
                     "via tools, most likely first. "
                     "submit_answer({\"ranked_paper_ids\": [id1, id2, id3, id4, id5], "
                     "\"evidence_node_ids\": [...], \"evidence_edges\": [...], \"rationale\": \"...\", "
                     "\"confidence\": 0.0-1.0}).")
        return "\n\n".join(parts)
    return "\n\n".join([
        f"Cutoff (as_of): {inst['as_of']}",
        f"The paper under review has {inst['n_paragraphs']} paragraphs (ids p1..p{inst['n_paragraphs']}) "
        f"and {inst['n_input_notes']} review-thread notes posted before the cutoff.",
        "Task: predict which paragraphs of the current version the authors will edit in the next "
        "revision, given the review state. Inspect the reviews and the paper, then finish with "
        "EXACTLY 10 distinct paragraph ids, most likely first. "
        "submit_answer({\"ranked_paragraph_ids\": [\"p7\", ...10 ids...], \"proposed_actions\": [...], "
        "\"evidence_note_ids\": [...], \"confidence\": 0.0-1.0})."])


# ---------------- runner ----------------

def run_episode(env, store, inst, condition, policy_factory, view="standard"):
    ep = Episode(env, inst, condition, prompt_version=PROMPT_VERSION)
    pol = policy_factory()
    pol.reset(ep, build_system_prompt(inst["task"]), user_prompt(inst, view))
    raw_log = []
    while not ep.finished and not ep.out_of_budget():
        try:
            tool, args = pol.act(ep)
        except Exception as e:                       # malformed output after one retry
            ep.stats["policy_parse_failures"] += 1
            raw_log.append(dict(error=str(e), raw=getattr(pol, "last_raw", None)))
            break
        obs = ep.call(tool, args)
        raw_log.append(dict(tool=tool, arguments=args, observation=obs))
        pol.observe(tool, args, obs)
    audit = store.audit(ep.observed_ids, inst["as_of"])
    need = REQUIRED_ANSWER_LEN[inst["task"]]
    if inst["task"] == "literature_evidence":
        ranked = list(dict.fromkeys(x.replace("paper:", "")
                                    for x in (ep.answer or {}).get("ranked_paper_ids", [])))
        gold = [inst["gold_target_id"]]
        pool = [x.replace("paper:", "") for x in dict.fromkeys(ep.observed_ids) if x.startswith("paper:")]
    else:
        ranked = list(dict.fromkeys((ep.answer or {}).get("ranked_paragraph_ids", [])))
        gold = inst["gold_paragraph_ids"]
        pool = [x for x in dict.fromkeys(ep.observed_ids) if x.startswith("p")]
    # every system returns exactly `need` ids; short answers are padded from observations in the
    # order they were seen, so ranking metrics are comparable across systems (padding is logged)
    raw_len = len(ranked)
    padded = 0
    for cand in pool:
        if len(ranked) >= need:
            break
        if cand not in ranked:
            ranked.append(cand)
            padded += 1
    ranked = ranked[:need]
    observed = set(ep.observed_ids)
    ev = (ep.answer or {}).get("evidence_node_ids", []) or (ep.answer or {}).get("evidence_note_ids", [])
    faithful = all(e in observed for e in ev) if ev else None
    target_rank = (ranked.index(gold[0]) + 1) if (gold and gold[0] in ranked) else 0
    pred = dict(instance_id=inst["instance_id"], condition=condition, task=inst["task"],
                run_id=RUN_ID_HOLDER.get("run_id"),
                raw_answer_len=raw_len, padded_from_observations=padded,
                unique_predictions=len(set(ranked)), target_rank=target_rank,
                candidate_pool_size=len(pool), submit_attempts=ep.stats.get("submit_attempts", 0),
                variant=CONDITION_VARIANT.get(condition), model=getattr(pol, "name", "?"),
                prompt_version=PROMPT_VERSION, answered=ep.answer is not None, ranked=ranked,
                gold=gold, tool_calls=ep.calls_used, observed_tokens=ep.observed_tokens,
                latency_ms=ep.summary()["latency_ms"], attempted_relation_calls=ep.stats.get(
                    "attempted_relation_calls", 0), tool_errors=ep.stats.get("tool_errors", 0),
                evidence_faithful=faithful,
                future_leakage_violations=audit["future_leakage_violations"],
                metrics=metrics_for(inst["task"], ranked, gold,
                                    extra=dict(answered=float(ep.answer is not None)),
                                    faithful=faithful))
    traj = dict(instance_id=inst["instance_id"], condition=condition, steps=ep.trajectory,
                raw=raw_log, system_prompt_version=PROMPT_VERSION)
    return pred, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["literature_evidence", "rtloc"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--paired-instance-file", default=None)
    ap.add_argument("--subset", default="instances_clean.jsonl")
    ap.add_argument("--policy", default="mock", choices=["mock", "vllm", "controlled"])
    ap.add_argument("--cache", default=None, help="shared response cache (determinism invariant)")
    ap.add_argument("--setting", default="optional", choices=["optional", "controlled"],
                    help="optional = free tool use; controlled = identical graph exposure")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--view", default="standard", choices=["standard", "delex"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    rid = a.run_id or current_run_id(create=True)
    RUN_ID_HOLDER["run_id"] = rid
    out_dir = run_dir(rid, a.task, create=True)
    instances = load_instances(a.task, a.split, a.paired_instance_file, a.subset)
    if a.limit:
        instances = instances[: a.limit]
    variants = sorted({CONDITION_VARIANT[c] for c in a.conditions})
    if a.task == "rtloc":
        graphs = load_rtloc_graphs(variants + (["full"] if "full" not in variants else []))
        env = RTLocEnv(graphs, rtloc_store(graphs["full"]))
        store = env.store
        instances = [i for i in instances if i["instance_id"] in graphs["full"]]
    else:
        papers, bm, adj, para_owner = load_literature_corpus(tuple(variants))
        store = literature_store(papers, para_owner)
        env = LiteratureEvidenceEnv(papers, bm, adj, store, paragraph_lookup_factory(), para_owner)

    cache = ResponseCache(a.cache or (out_dir / "response_cache.jsonl"))

    def policy_factory():
        if a.policy == "mock":
            return MockPolicy()
        llm = VLLMPolicy(a.model, a.base_url, cache=cache)
        if a.policy == "controlled" or a.setting == "controlled":
            return ControlledGraphPolicy(llm)
        return llm

    for cond in a.conditions:
        suffix = "" if a.setting == "optional" else f"_{a.setting}"
        pred_path = out_dir / f"predictions_{cond}{suffix}.jsonl"
        traj_path = out_dir / f"trajectories_{cond}{suffix}.jsonl"
        done = {}
        if a.resume and pred_path.exists() and not a.force:
            for r in read_jsonl(pred_path):
                if r.get("prompt_version") == PROMPT_VERSION:
                    done[r["instance_id"]] = r
        todo = [i for i in instances if i["instance_id"] not in done]
        print(f"[{cond}] {len(todo)} to run, {len(done)} cached", flush=True)
        preds, trajs = [], []
        t0 = time.time()
        for n, inst in enumerate(todo, 1):
            p, t = run_episode(env, store, inst, cond, policy_factory, a.view)
            preds.append(p)
            trajs.append(t)
            if n % 25 == 0:
                print(f"  {n}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
        rows = [done.get(i["instance_id"]) or next(p for p in preds if p["instance_id"] == i["instance_id"])
                for i in instances]                      # identical ordering across conditions
        with open(pred_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        mode = "a" if (a.resume and traj_path.exists()) else "w"
        with open(traj_path, mode) as f:
            for t in trajs:
                f.write(json.dumps(t, default=str) + "\n")
        keys = sorted({k for r in rows for k in r["metrics"]})
        print(f"[{cond}] n={len(rows)} " + json.dumps(
            {k: round(sum(r["metrics"].get(k, 0) for r in rows) / len(rows), 4) for k in keys}), flush=True)
    write_run_metadata(rid, phase=f"agent_{a.task}_{a.setting}")
    write_manifest(f"agent_run_{a.task}_{a.split}_{a.setting}", inputs=[DATA / a.task], seed=a.seed,
                   funnel=dict(n_instances=len(instances)),
                   filters=dict(split=a.split, subset=a.subset, view=a.view,
                                paired_file=a.paired_instance_file),
                   extra=dict(conditions=a.conditions, policy=a.policy, model=a.model,
                              prompt_version=PROMPT_VERSION, decoding=dict(temperature=0.0),
                              setting=a.setting,
                              response_cache=dict(hits=cache.hits, misses=cache.misses,
                                                  path=str(cache.path))))
    print(f"response cache: {cache.hits} hits / {cache.misses} misses "
          f"(identical transcripts reuse identical responses)")


if __name__ == "__main__":
    main()
