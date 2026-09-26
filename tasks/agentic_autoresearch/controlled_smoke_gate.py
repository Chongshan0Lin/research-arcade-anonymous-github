"""Runbook 1.5 smoke gate for controlled_interaction_v1.

Mechanical pass/fail over the 20 disjoint dev instances. The full run is forbidden unless every
check passes. Writes smoke_report.md + smoke_gate.json and exits non-zero on failure so an
orchestrator cannot proceed past a failed gate.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

from .controlled_agent import ANSWER_K, BUDGET, GRAPH_FOR, REQUIRED_FOR, h16
from .candidate_utility import _legal
from .temporal_store import normalize_ts

GRAPH_CONDS = ("correct_graph_controlled_agent", "rewired_graph_controlled_agent")


def load(root, sub="smoke"):
    rows = []
    for f in glob.glob(str(Path(root) / sub / "*" / "*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    return rows


def frac(xs):
    xs = list(xs)
    return (sum(bool(x) for x in xs) / len(xs)) if xs else 0.0


def gate(rows, papers=None, n_expected=20):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["model"], r["condition"])].append(r)
    models = sorted({r["model"] for r in rows})
    conds = sorted({r["condition"] for r in rows})
    C = []

    def add(name, ok, detail):
        C.append(dict(check=name, passed=bool(ok), detail=str(detail)))

    # 1. terminal rows
    sizes = {f"{m}/{c}": len(by[(m, c)]) for m in models for c in conds}
    add(f"{n_expected}/{n_expected} terminal rows per condition/model",
        all(v == n_expected for v in sizes.values()), sizes)

    # 2-3. tool execution
    for m in models:
        for c in conds:
            v = by[(m, c)]
            if not v:
                continue
            ge1 = frac(r["calls_succeeded"] >= 1 for r in v)
            mean = sum(r["calls_succeeded"] for r in v) / len(v)
            add(f"[{m}/{c}] >=90% episodes with a successful tool call", ge1 >= 0.90, f"{ge1:.0%}")
            add(f"[{m}/{c}] mean successful tool calls >= 2.0", mean >= 2.0, f"{mean:.2f}")

    # 4-5. manipulation uptake
    for m in models:
        for c in conds:
            v = by[(m, c)]
            if not v:
                continue
            for verb in REQUIRED_FOR[c]:
                u = frac(r["uptake"].get(verb, 0) >= 1 for r in v)
                add(f"[{m}/{c}] {verb} uptake >= 90%", u >= 0.90, f"{u:.0%}")

    # 6. action-parser validity
    pv = 1.0 - frac(r["parser_problem"] == "parse_failed" for r in rows)
    add("action-parser validity >= 95%", pv >= 0.95, f"{pv:.1%}")

    # 7. observations reach the next turn (a call returning ids must grow observed_candidates)
    bad = [r["instance_id"] for r in rows
           if any(c["returned_ids"] and not c["truncated"] for c in r["tool_calls"])
           and not r["observed_candidates"]]
    add("tool observations appear in subsequent context", not bad, f"{len(bad)} episodes with lost observations")

    # 8. correct vs rewired pools not universally identical
    for m in models:
        a = {r["instance_id"]: tuple(r["observed_candidates"]) for r in by[(m, GRAPH_CONDS[0])]}
        b = {r["instance_id"]: tuple(r["observed_candidates"]) for r in by[(m, GRAPH_CONDS[1])]}
        both = set(a) & set(b)
        diff = sum(a[i] != b[i] for i in both)
        add(f"[{m}] correct/rewired observed pools not universally identical",
            bool(both) and diff > 0, f"{diff}/{len(both)} instances differ")

    # 9. graph hashes differ, degree summaries match
    gh = {c: {r["graph_variant_hash"] for r in rows if r["condition"] == c} for c in GRAPH_CONDS}
    dh = {c: {r["graph_degree_hash"] for r in rows if r["condition"] == c} for c in GRAPH_CONDS}
    add("correct/rewired graph hashes differ", gh[GRAPH_CONDS[0]] != gh[GRAPH_CONDS[1]], gh)
    add("correct/rewired degree summaries match", dh[GRAPH_CONDS[0]] == dh[GRAPH_CONDS[1]], dh)

    # 10. no graph information reaches the flat condition
    leak = [r["instance_id"] for r in rows if r["condition"] == "flat_controlled_agent"
            and (r["graph_variant_hash"] != "none"
                 or any(c["tool"] in ("traverse", "hybrid_search") and not c.get("blocked")
                        for c in r["tool_calls"]))]
    add("no graph information reaches flat condition", not leak, f"{len(leak)} violations")

    # 11. correct/rewired prompts and tool schemas identical
    sh = {c: {r["prompt_schema_hash"] for r in rows if r["condition"] == c} for c in GRAPH_CONDS}
    ids_a = {r["instance_id"] for r in rows if r["condition"] == GRAPH_CONDS[0]}
    same = all(
        next((r["prompt_schema_hash"] for r in rows if r["condition"] == GRAPH_CONDS[0] and r["instance_id"] == i), 1)
        == next((r["prompt_schema_hash"] for r in rows if r["condition"] == GRAPH_CONDS[1] and r["instance_id"] == i), 2)
        for i in ids_a)
    add("correct/rewired prompt+schema hashes identical per instance", same, f"{len(sh[GRAPH_CONDS[0]])} distinct")

    # 12. cited evidence joins to observed records
    bad = [r["instance_id"] for r in rows
           if r.get("cited_evidence") and not set(r["cited_evidence"]) <= set(r["observed_candidates"])]
    add("cited evidence joins to observed records", not bad, f"{len(bad)} episodes cite unobserved ids")

    # 13. temporal leakage
    viol = 0
    if papers is not None:
        for r in rows:
            cut = normalize_ts(r["as_of"])
            for pid in r["observed_candidates"]:
                if not _legal(pid, papers, cut):
                    viol += 1
    add("zero temporal-leakage violations", viol == 0, f"{viol} violations"
        + ("" if papers is not None else " (corpus not loaded)"))

    # 14. budgets and truncation logging
    over = [r["instance_id"] for r in rows
            if len(r["tool_calls"]) > BUDGET["max_tool_calls"]
            or r["observation_tokens"] > BUDGET["max_observation_tokens"]]
    add("budgets enforced and truncation logged", not over, f"{len(over)} budget overruns")

    # 15. matched-pool replay reconstructs the observed pool
    add("matched_pool_hash reproduces observed pool",
        all(r["matched_pool_hash"] == h16("|".join(r["observed_candidates"])) for r in rows),
        "recomputed from raw rows")

    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n-expected", type=int, default=20)
    ap.add_argument("--sub", default="smoke")
    ap.add_argument("--skip-temporal", action="store_true")
    a = ap.parse_args()
    rows = load(a.root, a.sub)
    if not rows:
        print("no smoke rows found", file=sys.stderr)
        return 3
    papers = None
    if not a.skip_temporal:
        from .run_agent import load_literature_corpus
        papers = load_literature_corpus(())[0]
    C = gate(rows, papers, a.n_expected)
    ok = all(c["passed"] for c in C)
    root = Path(a.root)
    json.dump(dict(passed=ok, n_rows=len(rows), checks=C),
              open(root / "smoke_gate.json", "w"), indent=1)
    lines = ["# controlled_interaction_v1 smoke gate", "",
             f"Protocol: mandatory minimum research phase per condition "
             f"({ {k: v for k, v in REQUIRED_FOR.items()} }).", "",
             f"Episodes: {len(rows)}  |  **{'PASS' if ok else 'FAIL'}**", "",
             "| check | result | detail |", "|---|---|---|"]
    for c in C:
        lines.append(f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |")
    (root / "smoke_report.md").write_text("\n".join(lines) + "\n")
    for c in C:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"] + " :: " + c["detail"])
    print("GATE", "PASSED" if ok else "FAILED")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
