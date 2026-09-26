"""Lightweight release checks. No network, no models, no GPU, no database.

Run with:  python -m pytest tests -q     (or: python tests/test_release_integrity.py)
"""
import csv, json, glob, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "frozen_artifacts", "autoresearch")


def _rows(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def test_modules_import():
    sys.path.insert(0, ROOT)
    for m in ("controlled_aggregate", "ebc_aggregate", "litexp_aggregate",
              "appendix_tables", "graph_autoresearch_report", "evaluate"):
        __import__(f"tasks.agentic_autoresearch.{m}")


def test_frozen_id_counts():
    """The split arithmetic the paper relies on."""
    n = lambda p: sum(1 for _ in open(p))
    assert n(f"{A}/literature_set_expansion/frozen_ids.jsonl") == 1200
    assert n(f"{A}/literature_set_expansion/agent_results/agent_frozen_ids.jsonl") == 150
    assert n(f"{A}/evidence_bundle_completion/agent_results/agent_frozen_ids.jsonl") == 150
    splits = {}
    for line in open(f"{A}/literature_set_expansion/frozen_ids.jsonl"):
        splits[json.loads(line)["split"]] = splits.get(json.loads(line)["split"], 0) + 1
    assert splits == {"dev": 291, "screen": 909}, splits


def test_agent_set_nested_in_screen():
    """All 150 LSE agent instances must come from the held-out screen split."""
    split = {json.loads(l)["source_id"]: json.loads(l)["split"]
             for l in open(f"{A}/literature_set_expansion/frozen_ids.jsonl")}
    ag = [json.loads(l) for l in open(f"{A}/literature_set_expansion/agent_results/agent_frozen_ids.jsonl")]
    assert {split[r["source_id"]] for r in ag} == {"screen"}


def test_headline_values():
    """Fixed-retrieval headline numbers, to 4 decimals."""
    csc = {r["arm"]: r for r in _rows(f"{A}/evidence_bundle_completion/retrieval_results.csv")
           if r.get("slice") == "general"}
    assert abs(float(csc["hybrid_correct"]["recall@20"]) - 0.6303) < 5e-4
    assert abs(float(csc["hybrid_rewired"]["recall@20"]) - 0.3396) < 5e-4
    lse = {r["arm"]: r for r in _rows(f"{A}/literature_set_expansion/retrieval_results.csv")
           if r["subset"] == "screen"}
    assert abs(float(lse["hybrid_correct"]["recall@50"]) - 0.4266) < 5e-4
    assert abs(float(lse["hybrid_rewired"]["recall@50"]) - 0.2516) < 5e-4


def test_graph_manipulation_is_clean():
    """Correct and rewired must differ in adjacency, match in degree, share the prompt."""
    rows = []
    for f in glob.glob(f"{A}/evidence_bundle_completion/agent_results/episodes/*/*.jsonl"):
        rows += [json.loads(l) for l in open(f)]
    cor = {r["graph_variant_hash"] for r in rows if r["condition"] == "correct_traversal_agent"}
    rew = {r["graph_variant_hash"] for r in rows if r["condition"] == "rewired_traversal_agent"}
    assert cor and rew and cor != rew
    dc = {r["graph_degree_hash"] for r in rows if r["condition"] == "correct_traversal_agent"}
    dr = {r["graph_degree_hash"] for r in rows if r["condition"] == "rewired_traversal_agent"}
    assert dc == dr
    assert not [r for r in rows for c in r.get("tool_calls", [])
                if c.get("blocked") and c.get("returned_ids")], "blocked call returned candidates"


def test_no_excluded_material_present():
    assert not glob.glob(f"{A}/**/cache_*.jsonl", recursive=True)
    assert not glob.glob(f"{A}/**/superseded_v1", recursive=True)
    assert not glob.glob(f"{A}/**/incomplete_32b", recursive=True)
    assert not [p for p in glob.glob(f"{ROOT}/**/*", recursive=True)
                if os.path.isfile(p) and os.path.getsize(p) > 50 * 1024 ** 2]


def test_no_identity_strings():
    import re
    pat = re.compile(r"(/data|/home|/mnt)/[A-Za-z0-9_]+/|exx-[A-Za-z0-9]{6,}"
                     r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(edu|com|org|net)")
    bad = []
    for p in glob.glob(f"{ROOT}/**/*", recursive=True):
        if not os.path.isfile(p) or p.endswith((".gz", ".pdf", ".png")):
            continue
        try:
            t = open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        if pat.search(t):
            bad.append(os.path.relpath(p, ROOT))
    assert not bad, f"identity-like strings in: {bad[:5]}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"{len(fns)}/{len(fns)} release checks passed")
