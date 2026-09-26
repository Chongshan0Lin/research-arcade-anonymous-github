"""Canonical run state: immutable run ids, provenance hashes, and result de-duplication.

Every result row belongs to exactly one run. A report may only aggregate a single run id, so
numbers can never again be mixed across code versions or partial re-runs.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .common import DATA, MANIFESTS, RESULTS, file_hash, git_commit, read_jsonl, text_hash, write_json

RUN_ID_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{6}$")
# (run_id, task, condition, instance_id, model, prompt_version) identifies a result row
DEDUP_KEY = ("run_id", "task", "condition", "instance_id", "model", "prompt_version")


def new_run_id():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{os.urandom(3).hex()}"


def current_run_id(create=False):
    """Run id from $AAR_RUN_ID, else the newest existing run, else (optionally) a fresh one."""
    rid = os.environ.get("AAR_RUN_ID")
    if rid:
        if not RUN_ID_RE.match(rid):
            raise ValueError(f"malformed AAR_RUN_ID {rid!r}")
        return rid
    runs = sorted(p.name for p in RESULTS.glob("*") if RUN_ID_RE.match(p.name))
    if runs:
        return runs[-1]
    if create:
        return new_run_id()
    raise FileNotFoundError("no canonical run found under results/agentic_autoresearch/")


def run_dir(run_id=None, task=None, create=False):
    rid = run_id or current_run_id(create=create)
    d = RESULTS / rid
    if task:
        d = d / task
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def prompt_hash(task):
    from .policies import build_system_prompt
    return text_hash(build_system_prompt(task))


def provenance(task=None, graph_variant=None, model=None):
    """Hashes that pin a result to its exact inputs (printed at the top of every report)."""
    ds = {}
    for name in ["literature_evidence_a1", "literature_evidence_a2", "rtloc_instances",
                 "literature_evidence_graph_hard"]:
        p = MANIFESTS / f"{name}.json"
        if p.exists():
            ds[name] = file_hash(p)[:16]
    graphs = {}
    for p in sorted(DATA.glob("*/graphs/*.jsonl")):
        if graph_variant and graph_variant not in p.stem:
            continue
        graphs[f"{p.parent.parent.name}/{p.stem}"] = file_hash(p, limit=32 << 20)[:16]
    out = dict(git_commit=git_commit(), dataset_manifest_hashes=ds, graph_hashes=graphs, model=model)
    if task:
        out["prompt_hash"] = prompt_hash(task)
    return out


def write_run_metadata(run_id, **extra):
    d = run_dir(run_id, create=True)
    meta = dict(run_id=run_id, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **provenance(), **extra)
    write_json(d / "run_metadata.json", meta)
    return meta


def stamp_rows(rows, run_id, task):
    for r in rows:
        r.setdefault("run_id", run_id)
        r.setdefault("task", task)
    return rows


def dedup(rows):
    """Keep the LAST row per dedup key; returns (rows, n_duplicates_dropped)."""
    seen = {}
    for r in rows:
        seen[tuple(str(r.get(k)) for k in DEDUP_KEY)] = r
    return list(seen.values()), len(rows) - len(seen)


def load_run_rows(task, run_id=None, pattern="predictions_*.jsonl"):
    """Load every prediction row of one run, refusing to mix run ids."""
    rid = run_id or current_run_id()
    d = run_dir(rid, task)
    out, ids = [], set()
    for f in sorted(d.glob(pattern)):
        rows = list(read_jsonl(f))
        for r in rows:
            ids.add(r.get("run_id", rid))
        out += rows
    if len(ids - {None}) > 1:
        raise RuntimeError(f"results for task {task} mix run ids {sorted(ids)}; refusing to aggregate")
    rows, dropped = dedup(out)
    return rows, dict(run_id=rid, files=len(list(d.glob(pattern))), rows=len(rows),
                      duplicates_dropped=dropped)


def provenance_header(run_id, task=None, model=None, n_results=None):
    p = provenance(task=task, model=model)
    lines = [f"- run_id: `{run_id}`", f"- git commit: `{p['git_commit']}`",
             f"- model: `{model or 'n/a'}`", f"- result rows: {n_results if n_results is not None else 'n/a'}"]
    if "prompt_hash" in p:
        lines.append(f"- prompt hash: `{p['prompt_hash']}`")
    lines.append("- dataset manifest hashes: " +
                 ", ".join(f"`{k}`={v}" for k, v in sorted(p["dataset_manifest_hashes"].items())))
    lines.append("- graph hashes: " + ", ".join(f"`{k}`={v}" for k, v in sorted(p["graph_hashes"].items())))
    return "\n".join(lines)
