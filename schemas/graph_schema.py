"""Instance / trajectory / answer schemas and lightweight validators (no external deps)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"
TASKS = ("literature_evidence", "rtloc")
# every system must return exactly this many distinct ids, so ranking metrics are comparable
REQUIRED_ANSWER_LEN = {"literature_evidence": 5, "rtloc": 10}
SPLITS = ("train", "validation", "test")
CONDITIONS = ("one_shot_bm25", "one_shot_semantic", "flat_agent", "untyped_graph_agent",
              "typed_graph_agent", "rewired_graph_agent", "type_shuffled_agent", "oracle")
# graph variant backing each agent condition
CONDITION_VARIANT = {"flat_agent": "flat", "untyped_graph_agent": "untyped",
                     "typed_graph_agent": "full", "rewired_graph_agent": "rewire",
                     "type_shuffled_agent": "type_shuffle"}

ISO = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?")


class SchemaError(ValueError):
    pass


def _req(d, key, types, where):
    if key not in d or d[key] is None:
        raise SchemaError(f"{where}: missing '{key}'")
    if not isinstance(d[key], types):
        raise SchemaError(f"{where}: '{key}' must be {types}, got {type(d[key])}")
    return d[key]


@dataclass
class Instance:
    """Common envelope required by spec 3.1; `payload` holds task-specific fields."""
    instance_id: str
    task: str
    as_of: str
    source_paper_id: str
    split: str
    payload: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self):
        d = asdict(self)
        d.update(d.pop("payload"))
        return d


def validate_instance(d: dict, where="instance"):
    _req(d, "instance_id", str, where)
    t = _req(d, "task", str, where)
    if t not in TASKS:
        raise SchemaError(f"{where}: unknown task {t}")
    a = _req(d, "as_of", str, where)
    if not ISO.match(a):
        raise SchemaError(f"{where}: as_of '{a}' is not ISO-8601")
    _req(d, "source_paper_id", str, where)
    s = _req(d, "split", str, where)
    if s not in SPLITS:
        raise SchemaError(f"{where}: bad split {s}")
    if d.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(f"{where}: schema_version mismatch")
    return True


# ---------------- answers ----------------

def validate_answer(task: str, payload: Any, observed=None, enforce=True):
    """Validate a submit_answer payload. Returns a normalized dict or raises SchemaError.

    `enforce` requires exactly REQUIRED_ANSWER_LEN distinct ids (and, when `observed` is given,
    that they were actually seen in observations)."""
    need = REQUIRED_ANSWER_LEN[task]
    if not isinstance(payload, dict):
        raise SchemaError("submit_answer: arguments must be an object")
    if task == "literature_evidence":
        ids = _req(payload, "ranked_paper_ids", list, "submit_answer")
        if not ids:
            raise SchemaError("submit_answer: ranked_paper_ids is empty")
        if not all(isinstance(x, str) for x in ids):
            raise SchemaError("submit_answer: ranked_paper_ids must be strings")
        uniq = list(dict.fromkeys(x.strip() for x in ids if isinstance(x, str) and x.strip()))
        if enforce:
            if len(uniq) != need:
                raise SchemaError(
                    f"submit_answer: ranked_paper_ids must contain exactly {need} DISTINCT paper ids "
                    f"(got {len(uniq)}). Rank your best {need} candidates, most likely first.")
            if observed is not None:
                miss = [x for x in uniq if f"paper:{x}" not in observed and x not in observed]
                if miss:
                    raise SchemaError(
                        f"submit_answer: these ids were never returned by a tool: {miss[:3]}. "
                        "Only submit paper ids you actually observed.")
        out = dict(ranked_paper_ids=uniq[:need],
                   evidence_node_ids=[x for x in payload.get("evidence_node_ids", []) if isinstance(x, str)],
                   evidence_edges=[e for e in payload.get("evidence_edges", []) if isinstance(e, dict)
                                   and {"source", "relation", "target"} <= set(e)],
                   rationale=str(payload.get("rationale", ""))[:2000],
                   confidence=_confidence(payload))
        return out
    if task == "rtloc":
        ids = _req(payload, "ranked_paragraph_ids", list, "submit_answer")
        if not ids:
            raise SchemaError("submit_answer: ranked_paragraph_ids is empty")
        norm = []
        for x in ids:
            if isinstance(x, int):
                x = f"p{x}"
            if not isinstance(x, str):
                raise SchemaError("submit_answer: ranked_paragraph_ids must be strings like 'p7'")
            norm.append(x if x.startswith("p") else f"p{x}")
        acts = [a for a in payload.get("proposed_actions", []) if isinstance(a, dict) and "paragraph_id" in a]
        norm = list(dict.fromkeys(norm))
        if enforce and len(norm) != need:
            raise SchemaError(
                f"submit_answer: ranked_paragraph_ids must contain exactly {need} DISTINCT paragraph "
                f"ids (got {len(norm)}), most likely first.")
        return dict(ranked_paragraph_ids=norm[:need], proposed_actions=acts[:20],
                    evidence_note_ids=[x for x in payload.get("evidence_note_ids", []) if isinstance(x, str)],
                    confidence=_confidence(payload))
    raise SchemaError(f"unknown task {task}")


def _confidence(payload):
    try:
        c = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return min(max(c, 0.0), 1.0)


# ---------------- trajectories ----------------

def validate_step(d: dict):
    for k, ty in [("step", int), ("tool", str), ("arguments", dict), ("observation_ids", list),
                  ("observation_token_count", int), ("latency_ms", (int, float))]:
        _req(d, k, ty, "trajectory step")
    if "error" not in d:
        raise SchemaError("trajectory step: missing 'error' (use null)")
    return True
