"""TemporalStore: the single gate every tool must pass observations through.

An object is legal for an instance iff its first-available timestamp is <= the instance `as_of`.
Objects with a missing/ambiguous timestamp are never silently treated as clean: depending on
`missing_policy` they are excluded ("exclude", default) or surfaced as temporally unverifiable
("unverifiable"), which marks the instance rather than leaking the object.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def normalize_ts(ts):
    """Normalize to a lexicographically comparable 'YYYY-MM-DD HH:MM:SS'. None if unparseable."""
    if ts is None:
        return None
    s = str(ts).strip()
    if not s or s.lower() in ("nan", "none", "nat"):
        return None
    m = _DATE.match(s)
    if not m:
        return None
    rest = s[10:].strip().replace("T", " ")
    rest = rest.split("+")[0].split("Z")[0].strip()
    if not rest:
        rest = "00:00:00"
    parts = rest.split(":")
    while len(parts) < 3:
        parts.append("00")
    hh, mm, ss = (p[:2].rjust(2, "0") for p in parts[:3])
    return f"{s[:10]} {hh}:{mm}:{ss}"


@dataclass(frozen=True)
class TemporalObject:
    obj_id: str
    kind: str           # paper | paragraph | note | edge | revision | ...
    available_at: str | None


class TemporalViolation(RuntimeError):
    pass


class TemporalStore:
    """Registry of objects + their first-available timestamps, queried under one `as_of`."""

    def __init__(self, objects=None, missing_policy="exclude"):
        assert missing_policy in ("exclude", "unverifiable")
        self.missing_policy = missing_policy
        self._obj: dict[str, TemporalObject] = {}
        self.stats = Counter()
        for o in (objects or []):
            self.add(o)

    def add(self, obj: TemporalObject):
        self._obj[obj.obj_id] = TemporalObject(obj.obj_id, obj.kind, normalize_ts(obj.available_at))

    def add_many(self, triples):
        for obj_id, kind, ts in triples:
            self.add(TemporalObject(obj_id, kind, ts))

    def known(self, obj_id):
        return obj_id in self._obj

    def available_at(self, obj_id):
        o = self._obj.get(obj_id)
        return o.available_at if o else None

    def is_visible(self, obj_id, as_of):
        """True iff the object exists and was available by `as_of`."""
        o = self._obj.get(obj_id)
        if o is None:
            self.stats["unknown_object"] += 1
            return False
        if o.available_at is None:
            self.stats["missing_timestamp"] += 1
            return False if self.missing_policy == "exclude" else False
        as_of = normalize_ts(as_of)
        if as_of is None:
            raise TemporalViolation(f"instance as_of is unparseable: {as_of!r}")
        ok = o.available_at <= as_of
        self.stats["visible" if ok else "filtered_future"] += 1
        return ok

    def is_unverifiable(self, obj_id):
        o = self._obj.get(obj_id)
        return o is not None and o.available_at is None

    def filter_ids(self, ids, as_of):
        """Split ids into (visible, hidden). Order preserved."""
        vis, hid = [], []
        for i in ids:
            (vis if self.is_visible(i, as_of) else hid).append(i)
        return vis, hid

    def assert_visible(self, obj_id, as_of):
        if not self.is_visible(obj_id, as_of):
            raise TemporalViolation(
                f"object {obj_id} (available_at={self.available_at(obj_id)}) is not legal at as_of={as_of}")
        return True

    def audit(self, observation_ids, as_of):
        """Independent post-hoc check used by the evaluator: how many observed objects were illegal."""
        violations = [i for i in observation_ids if self.known(i) and not self._legal(i, as_of)]
        unknown = [i for i in observation_ids if not self.known(i)]
        return dict(future_leakage_violations=len(violations), violating_ids=violations[:50],
                    unknown_object_ids=unknown[:50], n_unknown=len(unknown))

    def _legal(self, obj_id, as_of):
        o = self._obj[obj_id]
        if o.available_at is None:
            return False
        return o.available_at <= normalize_ts(as_of)

    def __len__(self):
        return len(self._obj)
