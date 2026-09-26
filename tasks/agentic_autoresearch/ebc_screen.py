"""EBC retrieval screening, paired statistics and the agent-launch gate (runbook section 3.4).

CPU only, no LLM, no GPU, no agent is run here.

    python -m tasks.agentic_autoresearch.ebc_screen --out <master>/evidence_bundle_completion

Arms (all share the identical frozen instance IDs, identical cutoffs, identical candidate budget
K=50, and identical source-paper / future-node masking; correct and rewired differ ONLY in the
backing adjacency):

    bm25                        lexical BM25 over the masked paragraph
    graph_correct               Adamic-Adar expansion from the anchor, correct pre-cutoff graph
    hybrid_correct              RRF(bm25, graph_correct)
    graph_rewired               same expansion, degree-preserving rewired graph
    hybrid_rewired              RRF(bm25, graph_rewired)
    graph_random_anchor         negative control: expansion from a random legal paper
    hybrid_random_anchor        RRF(bm25, graph_random_anchor)
    graph_degree_matched_anchor negative control: random paper with the anchor's degree
    anchor_text_only            BM25 over the anchor's title+abstract instead of the paragraph

DECLARED PRIMARY COMPARISON (fixed before any arm was executed):
    hybrid_correct - hybrid_rewired on macro target Recall@20.
Confirmatory family for Holm adjustment:
    {hybrid_correct - hybrid_rewired, graph_correct - graph_rewired} at Recall@20.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import file_hash, git_commit, now_iso, read_jsonl, write_json
from .ebc_build import AUDIT_FIELDS, LABELS, INSTANCES, sha256_list
from .ebc_common import (BOOT_SEED, BUDGET_POOL, EBC_SEED, EBCCorpus, INF_DATE, K_FINAL, KS,
                         MAX_HOPS, N_BOOT, PARAMS, TASK_NAME, base_id, date_int)
from .ebc_retrieval import instance_metrics, run_arms
from .evaluate import holm

ARMS = ["bm25", "graph_correct", "hybrid_correct", "graph_rewired", "hybrid_rewired",
        "graph_random_anchor", "hybrid_random_anchor", "graph_degree_matched_anchor",
        "anchor_text_only"]
AUDITED_ARMS = ["bm25", "graph_correct", "hybrid_correct", "graph_rewired", "hybrid_rewired"]
AUDIT_TOPK = 20

PRIMARY = ("hybrid_correct", "hybrid_rewired", "recall@20")
FAMILY = [("hybrid_correct", "hybrid_rewired", "recall@20"),
          ("graph_correct", "graph_rewired", "recall@20")]
SECONDARY = [
    ("hybrid_correct", "bm25", "recall@20"),
    ("graph_correct", "bm25", "recall@20"),
    ("graph_correct", "graph_random_anchor", "recall@20"),
    ("graph_correct", "graph_degree_matched_anchor", "recall@20"),
    ("hybrid_correct", "anchor_text_only", "recall@20"),
    ("hybrid_correct", "hybrid_random_anchor", "recall@20"),
    ("hybrid_correct", "hybrid_rewired", "recall@10"),
    ("hybrid_correct", "hybrid_rewired", "recall@5"),
    ("hybrid_correct", "hybrid_rewired", "set_f1@10"),
    ("hybrid_correct", "hybrid_rewired", "mrr"),
    ("graph_correct", "graph_rewired", "recall@10"),
    ("graph_correct", "graph_rewired", "recall@5"),
    ("graph_correct", "graph_rewired", "mrr"),
    ("graph_correct", "graph_rewired", "set_f1@10"),
]

METRIC_COLS = ([f"recall@{k}" for k in KS] + [f"set_recall@{k}" for k in KS] +
               [f"hit@{k}" for k in KS] + ["set_f1@10", "set_f1@20", "r_precision", "mrr",
                                           "graph_only_discoveries"])

GATE_MIN_DELTA = 0.05
GATE_MIN_GRAPHHARD_REACH = 0.30


# ---------------------------------------------------------------- statistics

def paired_boot(a, b, n_boot=N_BOOT, seed=BOOT_SEED, alpha=0.05, chunk=1000):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    n = d.size
    if n == 0:
        return dict(diff=None, lo=None, hi=None, n=0, n_boot=n_boot, seed=seed)
    rs = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        means[done:done + m] = d[rs.integers(0, n, size=(m, n))].mean(axis=1)
        done += m
    means.sort()
    return dict(diff=float(d.mean()), lo=float(np.quantile(means, alpha / 2)),
                hi=float(np.quantile(means, 1 - alpha / 2)), n=int(n),
                n_boot=int(n_boot), seed=int(seed),
                mean_a=float(a.mean()), mean_b=float(b.mean()))


def paired_perm_p(a, b, n_perm=N_BOOT, seed=BOOT_SEED, chunk=1000):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = d.size
    if n == 0:
        return None
    obs = abs(d.mean())
    rs = np.random.default_rng(seed + 1)
    count, done = 0, 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = rs.integers(0, 2, size=(m, n)) * 2 - 1
        count += int((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).sum())
        done += m
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------- screening

def screen(corpus, insts, per_inst_path, audit_path, resume=True):
    done = set()
    if resume and Path(per_inst_path).exists():
        done = {r["instance_id"] for r in read_jsonl(per_inst_path)}
        print(f"   resuming: {len(done)} instances already screened", flush=True)
    todo = [i for i in insts if i["instance_id"] not in done]
    if not todo:
        return
    fh = open(per_inst_path, "a")
    af = open(audit_path, "a", newline="")
    aw = csv.DictWriter(af, fieldnames=AUDIT_FIELDS)
    t0 = time.time()
    for n, inst in enumerate(todo, 1):
        rng_pick = random.Random(f"{EBC_SEED}|rand_anchor|{inst['instance_id']}")
        inst["_random_pool"] = corpus.legal_paper_pool(inst["as_of_int"])
        ranked, diag = run_arms(corpus, inst, rng_pick)
        targets = inst["target_paper_ids"]
        bm20 = diag["bm25_top20"]
        row = dict(instance_id=inst["instance_id"], source_paper_id=inst["source_paper_id"],
                   as_of=inst["as_of"], n_targets=len(targets),
                   anchor_paper_id=inst["anchor_paper_id"],
                   anchor_in_graph=bool(diag["anchor_in_graph"]),
                   anchor_degree=diag["anchor_degree"],
                   random_anchor_id=diag["random_anchor_id"],
                   degree_matched_anchor_id=diag["degree_matched_anchor_id"], arms={})
        for arm in ARMS:
            lst = ranked.get(arm, [])
            row["arms"][arm] = dict(instance_metrics(lst, targets, bm20),
                                    top20=lst[:20], n_returned=len(lst))
        reach = {}
        for t in targets:
            r = diag["reach"][t]
            reach[t] = dict(correct_hops=r["correct"]["hops"], correct_witness=r["correct"]["witness"],
                            rewired_hops=r["rewired"]["hops"], in_budget_pool=bool(r["in_budget_pool"]))
        row["reach"] = reach
        row["audit"] = audit_instance(corpus, inst, ranked, diag, aw)
        fh.write(json.dumps(row, default=str) + "\n")
        if n % 250 == 0:
            fh.flush()
            af.flush()
            el = time.time() - t0
            print(f"   {n}/{len(todo)}  {el:.0f}s  ({el / n * 1000:.0f} ms/inst)", flush=True)
    fh.close()
    af.close()


def audit_instance(corpus, inst, ranked, diag, aw):
    """Per-candidate temporal / source-edge audit. Returns the per-instance violation summary."""
    s = inst["source_paper_id"]
    as_i = inst["as_of_int"]
    s_group = corpus.source_group(s)
    s_nodes = corpus.nodes_for_paper(s)
    legal = diag["legal_mask"]
    anchor = inst["anchor_paper_id"]
    tset = set(inst["target_paper_ids"])
    para_node = corpus.node_index.get(f"para:{inst['paragraph_global_id']}")
    summary = dict(
        n_candidates_audited=0, future_candidate=0, source_paper_candidate=0,
        anchor_returned_as_candidate=0, source_owned_candidate=0, illegal_witness=0,
        source_node_masked=bool(not s_nodes.size or not legal[s_nodes].any()),
        eval_paragraph_masked=bool(para_node is None or not legal[para_node]),
        anchor_legal=bool(date_int(corpus.date.get(anchor)) <= as_i),
        all_targets_legal=all(date_int(corpus.date.get(t)) <= as_i for t in tset))
    # 1. exhaustive check over EVERY candidate of EVERY arm at full depth K_FINAL
    for arm in ARMS:
        for pid in ranked.get(arm, []):
            summary["n_candidates_audited"] += 1
            if date_int(corpus.date.get(pid)) > as_i:
                summary["future_candidate"] += 1
            if base_id(pid) == base_id(s):
                summary["source_paper_candidate"] += 1
            if base_id(pid) == base_id(anchor):
                summary["anchor_returned_as_candidate"] += 1
            nds = corpus.nodes_for_paper(pid)
            if nds.size and int(corpus.group[nds[0]]) == s_group:
                summary["source_owned_candidate"] += 1
    # 2. 2-hop witnesses used by the reachability labels must themselves be legal
    for t, r in diag["reach"].items():
        for lab in ("correct", "rewired"):
            w = r[lab]["witness"]
            if w is None:
                continue
            wi = corpus.node_index.get(w)
            if wi is None or not legal[wi]:
                summary["illegal_witness"] += 1
    summary["violations"] = (summary["future_candidate"] + summary["source_paper_candidate"]
                             + summary["anchor_returned_as_candidate"]
                             + summary["source_owned_candidate"] + summary["illegal_witness"]
                             + (0 if summary["source_node_masked"] else 1)
                             + (0 if summary["eval_paragraph_masked"] else 1)
                             + (0 if summary["anchor_legal"] else 1)
                             + (0 if summary["all_targets_legal"] else 1))
    # 3. per-candidate audit rows (top-20 of the five primary arms, de-duplicated per instance)
    seen = {}
    for arm in AUDITED_ARMS:
        for rank, pid in enumerate(ranked.get(arm, [])[:AUDIT_TOPK], 1):
            if pid in seen:
                seen[pid]["arm"] += f"|{arm}"
                continue
            nds = corpus.nodes_for_paper(pid)
            grp = int(corpus.group[nds[0]]) if nds.size else -1
            d = corpus.date.get(pid)
            di = date_int(d)
            own = corpus.group_base[grp] if grp >= 0 else ""
            wit = wdate = wlegal = ""
            hop = ""
            if pid in tset:
                rr = diag["reach"][pid]["correct"]
                hop = rr["hops"] if rr["hops"] is not None else ""
                if rr["witness"]:
                    wit = rr["witness"]
                    wi = corpus.node_index.get(wit)
                    wdate = int(corpus.avail[wi]) if wi is not None else ""
                    wlegal = bool(wi is not None and legal[wi])
            bad = (di > as_i) or base_id(pid) in (base_id(s), base_id(anchor)) or grp == s_group
            seen[pid] = dict(row_kind="candidate", instance_id=inst["instance_id"],
                             source_paper_id=s, as_of=inst["as_of"], arm=arm, rank=rank,
                             object_id=pid, object_kind="paper", object_date=d or "",
                             temporally_legal=di <= as_i, strictly_before_cutoff=di < as_i,
                             is_source_paper=base_id(pid) == base_id(s),
                             is_anchor=base_id(pid) == base_id(anchor),
                             is_target=pid in tset, owner_paper_id=own,
                             edge_created_by_source=bool(grp == s_group),
                             hop=hop, witness_intermediate=wit, witness_date=wdate,
                             witness_legal=wlegal, violation=bool(bad),
                             violation_reason=("" if not bad else
                                               "future_candidate" if di > as_i else
                                               "source_paper" if base_id(pid) == base_id(s) else
                                               "anchor_leak" if base_id(pid) == base_id(anchor) else
                                               "edge_created_by_source"))
    for r in seen.values():
        aw.writerow(r)
    return summary


# ---------------------------------------------------------------- aggregation

def aggregate(rows, slice_name):
    out = []
    for arm in ARMS:
        a = dict(slice=slice_name, arm=arm, n=len(rows))
        for m in METRIC_COLS:
            a[m] = float(np.mean([r["arms"][arm][m] for r in rows])) if rows else None
        a["mean_n_returned"] = float(np.mean([r["arms"][arm]["n_returned"] for r in rows])) if rows else None
        a["graph_only_discovery_rate"] = (
            float(np.mean([r["arms"][arm]["graph_only_discoveries"] > 0 for r in rows])) if rows else None)
        out.append(a)
    return out


def reach_summary(rows, slice_name):
    tot = c1 = c2 = rw = bud = 0
    for r in rows:
        for t, v in r["reach"].items():
            tot += 1
            if v["correct_hops"] == 1:
                c1 += 1
            if v["correct_hops"] in (1, 2):
                c2 += 1
            if v["rewired_hops"] in (1, 2):
                rw += 1
            if v["in_budget_pool"]:
                bud += 1
    f = (lambda x: x / tot if tot else None)
    return dict(slice=slice_name, n_instances=len(rows), n_targets=tot,
                target_reachable_1hop_correct=f(c1), target_reachable_2hop_correct=f(c2),
                target_reachable_2hop_rewired=f(rw),
                target_reachable_under_tool_budget=f(bud),
                anchor_in_graph_rate=(float(np.mean([r["anchor_in_graph"] for r in rows]))
                                      if rows else None))


def comparisons(rows, slice_name, pairs, family_flag):
    out = []
    for hi, lo, metric in pairs:
        a = [r["arms"][hi][metric] for r in rows]
        b = [r["arms"][lo][metric] for r in rows]
        ci = paired_boot(a, b)
        p = paired_perm_p(a, b)
        out.append(dict(slice=slice_name, comparison=f"{hi}-{lo}", metric=metric,
                        family=family_flag, mean_hi=ci["mean_a"], mean_lo=ci["mean_b"],
                        diff=ci["diff"], ci_lo=ci["lo"], ci_hi=ci["hi"], n_paired=ci["n"],
                        n_boot=ci["n_boot"], boot_seed=ci["seed"], p_raw=p,
                        ci_strictly_positive=bool(ci["lo"] is not None and ci["lo"] > 0),
                        is_primary=bool(family_flag == "primary" and (hi, lo, metric) == PRIMARY)))
    return out


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true", help="ignore existing per-instance rows")
    a = ap.parse_args()
    out = Path(a.out)
    frozen = list(read_jsonl(out / "frozen_ids.jsonl"))
    fid = {f["instance_id"]: f for f in frozen}
    print(f"frozen general n={len(frozen)} graphhard n={sum('graphhard' in f['slices'] for f in frozen)}")

    all_inst = {r["instance_id"]: r for r in read_jsonl(INSTANCES)}
    insts = [all_inst[i] for i in sorted(fid)]
    for i in insts:
        i["as_of_int"] = date_int(i["as_of"])

    print("loading corpus + graphs ...", flush=True)
    corpus = EBCCorpus(("full", "rewire"))

    per = out / "retrieval_per_instance.jsonl"
    audit = out / "temporal_audit.csv"
    if a.force and per.exists():
        per.unlink()
    print("screening ...", flush=True)
    screen(corpus, insts, per, audit, resume=not a.force)

    rows = [r for r in read_jsonl(per)]
    rows = {r["instance_id"]: r for r in rows}
    rows = [rows[i] for i in sorted(fid) if i in rows]
    gen = rows
    hard = [r for r in rows if "graphhard" in fid[r["instance_id"]]["slices"]]
    print(f"screened general={len(gen)} graphhard={len(hard)}")

    # ---- results table ----
    res = aggregate(gen, "general") + aggregate(hard, "graphhard")
    cols = ["slice", "arm", "n"] + METRIC_COLS + ["mean_n_returned", "graph_only_discovery_rate"]
    with open(out / "retrieval_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in res:
            w.writerow(r)

    # ---- paired comparisons ----
    comps = []
    for name, rs in (("general", gen), ("graphhard", hard)):
        comps += comparisons(rs, name, FAMILY, "primary")
        comps += comparisons(rs, name, SECONDARY, "secondary")
    for name in ("general", "graphhard"):
        fam = {c["comparison"] + "@" + c["metric"]: c["p_raw"] for c in comps
               if c["slice"] == name and c["family"] == "primary"}
        adj = holm(fam)
        for c in comps:
            if c["slice"] == name and c["family"] == "primary":
                k = c["comparison"] + "@" + c["metric"]
                c["p_adj_holm"] = adj[k]["p_adj"]
                c["reject_0_05"] = adj[k]["reject_0_05"]
            elif c["slice"] == name:
                c.setdefault("p_adj_holm", None)
                c.setdefault("reject_0_05", None)
    ccols = ["slice", "comparison", "metric", "family", "is_primary", "mean_hi", "mean_lo", "diff",
             "ci_lo", "ci_hi", "ci_strictly_positive", "p_raw", "p_adj_holm", "reject_0_05",
             "n_paired", "n_boot", "boot_seed"]
    with open(out / "retrieval_comparisons.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ccols)
        w.writeheader()
        for c in comps:
            w.writerow({k: c.get(k) for k in ccols})

    # ---- reachability ----
    reach = [reach_summary(gen, "general"), reach_summary(hard, "graphhard")]
    write_json(out / "reachability.json", reach)

    # ---- leakage audit roll-up ----
    keys = ["future_candidate", "source_paper_candidate", "anchor_returned_as_candidate",
            "source_owned_candidate", "illegal_witness"]
    tot = {k: int(sum(r["audit"][k] for r in rows)) for k in keys}
    tot["instances_with_unmasked_source_node"] = int(sum(not r["audit"]["source_node_masked"] for r in rows))
    tot["instances_with_unmasked_eval_paragraph"] = int(sum(not r["audit"]["eval_paragraph_masked"] for r in rows))
    tot["instances_with_illegal_anchor"] = int(sum(not r["audit"]["anchor_legal"] for r in rows))
    tot["instances_with_illegal_target"] = int(sum(not r["audit"]["all_targets_legal"] for r in rows))
    tot["candidates_audited"] = int(sum(r["audit"]["n_candidates_audited"] for r in rows))
    tot["total_violations"] = int(sum(r["audit"]["violations"] for r in rows))
    write_json(out / "temporal_audit_summary.json",
               dict(generated_at=now_iso(), n_instances=len(rows), totals=tot,
                    zero_violations=tot["total_violations"] == 0,
                    audit_csv=str(audit), audit_csv_sha256=file_hash(audit, limit=1 << 30)))

    # ---- gate ----
    def get(slice_name, hi, lo, metric):
        for c in comps:
            if (c["slice"], c["comparison"], c["metric"]) == (slice_name, f"{hi}-{lo}", metric):
                return c
        return None

    pg = get("general", "graph_correct", "graph_rewired", "recall@20")
    ph = get("general", *PRIMARY[:2], PRIMARY[2])
    hard_reach = reach[1]["target_reachable_under_tool_budget"]
    c1_graph = bool(pg and pg["diff"] is not None and pg["diff"] >= GATE_MIN_DELTA)
    c1_hyb = bool(ph and ph["diff"] is not None and ph["diff"] >= GATE_MIN_DELTA)
    c1 = c1_graph or c1_hyb
    c2 = bool(ph and ph["ci_strictly_positive"])
    c3 = bool(hard_reach is not None and hard_reach >= GATE_MIN_GRAPHHARD_REACH)
    c4 = tot["total_violations"] == 0
    passed = c1 and c2 and c3 and c4
    reasons = []
    if not c1:
        reasons.append(f"neither correct graph (delta={pg['diff'] if pg else None:.4f}) nor correct "
                       f"hybrid (delta={ph['diff'] if ph else None:.4f}) beats its rewired "
                       f"counterpart by >= {GATE_MIN_DELTA} absolute Recall@20")
    if not c2:
        reasons.append("paired 95% CI for the declared primary correct-minus-rewired comparison "
                       f"(hybrid_correct - hybrid_rewired @ recall@20) is not strictly positive: "
                       f"[{ph['ci_lo'] if ph else None}, {ph['ci_hi'] if ph else None}]")
    if not c3:
        reasons.append(f"only {hard_reach:.4f} of GraphHard targets are reachable under the "
                       f"declared tool budget ({PARAMS['tool_budget_calls']} calls x "
                       f"{PARAMS['candidates_per_call']} candidates), below {GATE_MIN_GRAPHHARD_REACH}")
    if not c4:
        reasons.append(f"temporal/source-edge leakage is non-zero: {tot['total_violations']} violations")
    gate = dict(
        task=TASK_NAME, gate="EBC agent-launch gate (runbook 3.4)", generated_at=now_iso(),
        git_commit=git_commit(), decision="PASS" if passed else "FAIL",
        stopped_not_failed=not passed,
        action=("EBC agent experiment (runbook 3.5) may proceed" if passed else
                "STOP: do not run EBC agents (runbook stopping rule 4); save the negative "
                "retrieval result"),
        declared_primary_comparison="hybrid_correct - hybrid_rewired @ macro target recall@20 "
                                    "(declared before any arm was executed)",
        criteria=dict(
            c1_delta_recall20_ge_0_05=dict(
                passed=c1, threshold=GATE_MIN_DELTA,
                graph_correct_minus_rewired=None if not pg else pg["diff"],
                hybrid_correct_minus_rewired=None if not ph else ph["diff"],
                met_by=("both" if (c1_hyb and c1_graph) else "hybrid" if c1_hyb
                        else "graph" if c1_graph else None)),
            c2_primary_ci_strictly_positive=dict(
                passed=c2, ci=[ph["ci_lo"], ph["ci_hi"]] if ph else None,
                diff=ph["diff"] if ph else None, p_raw=ph["p_raw"] if ph else None,
                p_adj_holm=ph.get("p_adj_holm") if ph else None,
                n_paired=ph["n_paired"] if ph else None, n_boot=N_BOOT, boot_seed=BOOT_SEED),
            c3_graphhard_reachability_ge_0_30=dict(
                passed=c3, threshold=GATE_MIN_GRAPHHARD_REACH, observed=hard_reach,
                n_graphhard_instances=reach[1]["n_instances"],
                n_graphhard_targets=reach[1]["n_targets"],
                tool_budget=dict(calls=PARAMS["tool_budget_calls"],
                                 candidates_per_call=PARAMS["candidates_per_call"],
                                 observable_pool=BUDGET_POOL)),
            c4_zero_temporal_leakage=dict(passed=c4, totals=tot)),
        reasons=reasons or ["all four criteria met"],
        frozen=dict(general_n=len(gen), graphhard_n=len(hard),
                    general_ids_sha256=sha256_list([r["instance_id"] for r in gen]),
                    graphhard_ids_sha256=sha256_list([r["instance_id"] for r in hard]),
                    frozen_ids_file_sha256=file_hash(out / "frozen_ids.jsonl")),
        parameters=PARAMS, arms=ARMS)
    write_json(out / "gate_decision.json", gate)

    # ---- console summary ----
    print("\n== retrieval_results.csv (macro target recall) ==")
    print(f"{'slice':10s} {'arm':30s} {'n':>5s} {'R@5':>7s} {'R@10':>7s} {'R@20':>7s} "
          f"{'setR@20':>8s} {'setF1@10':>9s} {'MRR':>7s} {'gonly':>7s}")
    for r in res:
        print(f"{r['slice']:10s} {r['arm']:30s} {r['n']:5d} {r['recall@5']:7.4f} "
              f"{r['recall@10']:7.4f} {r['recall@20']:7.4f} {r['set_recall@20']:8.4f} "
              f"{r['set_f1@10']:9.4f} {r['mrr']:7.4f} {r['graph_only_discoveries']:7.4f}")
    print("\n== primary family ==")
    for c in comps:
        if c["family"] == "primary":
            print(f"  [{c['slice']}] {c['comparison']} @{c['metric']}: diff={c['diff']:+.4f} "
                  f"CI=[{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}] p={c['p_raw']:.5f} "
                  f"p_holm={c.get('p_adj_holm')}")
    print("\n== reachability ==", json.dumps(reach, indent=1))
    print("== leakage ==", json.dumps(tot, indent=1))
    print(f"\n== GATE: {gate['decision']} ==")
    for r in gate["reasons"]:
        print("  -", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
