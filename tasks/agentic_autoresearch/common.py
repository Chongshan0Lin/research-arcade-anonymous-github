"""Shared paths, IO, hashing and manifest writing for the agentic auto-research benchmark."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROV = REPO / "data" / "provenance"
DATA = REPO / "data" / "agentic_autoresearch"
RESULTS = REPO / "results" / "agentic_autoresearch"
MANIFESTS = DATA / "manifests"
PROMPTS = Path(__file__).resolve().parent / "prompts"

SCHEMA_VERSION = "1.0"
DEFAULT_SEED = 20260922

# reuse the provenance loaders (DB connection, normalizers) without copying logic
sys.path.insert(0, str(REPO / "tasks" / "provenance"))


def db_conn():
    from common import db_conn as _c  # tasks/provenance/common.py
    return _c()


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def file_hash(path, limit=None):
    """sha256 of a file (optionally only the first `limit` bytes, for very large inputs)."""
    h = hashlib.sha256()
    p = Path(path)
    if not p.exists() or p.is_dir():
        return None
    with open(p, "rb") as f:
        read = 0
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            if limit and read + len(b) > limit:
                h.update(b[: limit - read])
                break
            h.update(b)
            read += len(b)
    return h.hexdigest()


def text_hash(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return path


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    return path


def write_manifest(name, *, inputs, funnel, filters, seed, splits=None, temporal_policy=None,
                   graph_variant=None, extra=None):
    """Every generated dataset/result gets one of these (spec section 2)."""
    man = dict(
        name=name, schema_version=SCHEMA_VERSION, generated_at=now_iso(), git_commit=git_commit(),
        inputs={str(p): dict(path=str(p), sha256=file_hash(p, limit=64 << 20),
                             bytes=(os.path.getsize(p) if Path(p).is_file() else None),
                             is_dir=Path(p).is_dir()) for p in inputs},
        funnel=funnel, filters=filters, seed=seed, splits=splits, temporal_policy=temporal_policy,
        graph_variant=graph_variant, **(extra or {}))
    write_json(MANIFESTS / f"{name}.json", man)
    return man


def rng(seed=DEFAULT_SEED):
    return random.Random(seed)


def split_by_group(groups, seed=DEFAULT_SEED, fractions=(0.7, 0.15, 0.15)):
    """Deterministic train/validation/test assignment over grouping keys (source paper or
    near-duplicate paragraph group), so no group spans two splits."""
    keys = sorted(set(groups))
    r = random.Random(seed)
    r.shuffle(keys)
    n = len(keys)
    n_tr = int(n * fractions[0])
    n_va = int(n * (fractions[0] + fractions[1]))
    out = {}
    for i, k in enumerate(keys):
        out[k] = "train" if i < n_tr else ("validation" if i < n_va else "test")
    return out
