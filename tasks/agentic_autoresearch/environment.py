"""Time-sandboxed environments for Task A (literature evidence) and Task B (RT-Loc).

The environment owns state, budgets, temporal filtering and tool validation. Every observation
carries stable `observation_ids` so the evaluator can later check evidence faithfulness and
independently audit future leakage.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict

from .schemas import CONDITION_VARIANT, validate_answer, SchemaError
from .temporal_store import TemporalObject, TemporalStore, normalize_ts
from .tools import RELATION_TOOLS, ToolError, validate_call

TOKEN = re.compile(r"[a-z0-9]+")
# tools whose OBSERVATIONS depend on the active graph variant (relation tools plus the OpenReview
# thread tools, whose reply_to fields are graph edges) -- i.e. real graph exposure
GRAPH_EXPOSING_TOOLS = {"get_citations", "get_neighbors", "find_paths", "get_reviews", "get_thread"}


def classify_error(tool, message):
    """Tool-error taxonomy (spec Phase 0.9)."""
    m = (message or "").lower()
    if "not available at" in m or "cutoff" in m:
        return "temporal_rejection"
    if "budget" in m:
        return "budget_rejection"
    if "unknown node_id" in m or "unknown paper_id" in m or "never returned by a tool" in m:
        return "unknown_id"
    if tool == "submit_answer":
        return "schema_failure"
    if "unknown tool" in m or "requires" in m or "must be" in m or "not available for this task" in m:
        return "invalid_arguments"
    return "internal_error"
DEFAULT_LIMITS = {"literature_evidence": dict(max_tool_calls=12, max_observed_tokens=16000),
                  "rtloc": dict(max_tool_calls=10, max_observed_tokens=16000)}


def ntok(text):
    return max(1, len(str(text)) // 4)


class BM25:
    """Small inverted-index BM25 so retrieval is identical across conditions and needs no GPU."""

    def __init__(self, docs: dict, k1=1.5, b=0.75):
        self.k1, self.b, self.ids = k1, b, list(docs)
        self.tf, self.len = {}, {}
        df = Counter()
        self.index = defaultdict(list)
        for i, (did, text) in enumerate(docs.items()):
            c = Counter(TOKEN.findall(str(text).lower()))
            self.tf[did] = c
            self.len[did] = sum(c.values()) or 1
            for w in c:
                df[w] += 1
                self.index[w].append(did)
        n = max(1, len(docs))
        self.avg = sum(self.len.values()) / n
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def search(self, query, k=10, allowed=None):
        q = Counter(TOKEN.findall(query.lower()))
        scores = defaultdict(float)
        for w, _ in q.items():
            if w not in self.index:
                continue
            idf = self.idf[w]
            for did in self.index[w]:
                if allowed is not None and did not in allowed:
                    continue
                f = self.tf[did][w]
                L = self.len[did]
                scores[did] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
        return sorted(scores.items(), key=lambda x: (-x[1], str(x[0])))[:k]


class Episode:
    """One (instance, condition) run: bounded tool loop with trajectory logging."""

    def __init__(self, env, instance, condition, limits=None, prompt_version="v1"):
        self.env, self.instance, self.condition = env, instance, condition
        self.task = instance["task"]
        self.limits = limits or DEFAULT_LIMITS[self.task]
        self.prompt_version = prompt_version
        self.variant = CONDITION_VARIANT.get(condition, "full")
        self.trajectory, self.observed_ids = [], []
        self.observed_tokens = 0
        self.answer = None
        self.stats = Counter()
        self.finished = False
        self.max_submit_retries = 2      # after this many rejected submits, accept and pad
        # Edges that encode the answer must never be observable (spec 3.3: no gold labels or target
        # identifiers in observations). For Task A the removed citation is itself an edge of the
        # citation graph, so it is masked in BOTH directions, for the source paper and for the
        # paragraph the citation was removed from.
        self.masked_pairs = set()
        if self.task == "literature_evidence":
            src, gold = instance.get("source_paper_id"), instance.get("gold_target_id")
            pg = instance.get("paragraph_global_id")
            nodes = [f"paper:{src}"] + ([f"para:{pg}"] if pg is not None else [])
            for n in nodes:
                self.masked_pairs.add((n, f"paper:{gold}"))
                self.masked_pairs.add((f"paper:{gold}", n))

    # ---- budget ----
    @property
    def calls_used(self):
        return len(self.trajectory)

    def budget_left(self):
        return (self.limits["max_tool_calls"] - self.calls_used,
                self.limits["max_observed_tokens"] - self.observed_tokens)

    def out_of_budget(self):
        c, t = self.budget_left()
        return c <= 0 or t <= 0

    def call(self, tool, arguments):
        t0 = time.time()
        err, obs = None, None
        try:
            args = validate_call(self.task, tool, arguments)
            if tool == "submit_answer":
                payload = args.get("payload", args)
                self.stats["submit_attempts"] += 1
                strict = self.stats["submit_attempts"] <= self.max_submit_retries
                self.answer = validate_answer(self.task, payload, observed=set(self.observed_ids),
                                              enforce=strict)
                self.answer["raw_len"] = len(payload.get("ranked_paper_ids")
                                             or payload.get("ranked_paragraph_ids") or [])
                self.finished = True
                obs = dict(status="ACCEPTED")
            elif tool in RELATION_TOOLS and self.condition == "flat_agent":
                self.stats["attempted_relation_calls"] += 1
                obs = dict(status="RELATIONS_UNAVAILABLE",
                           message="Graph relations are not available in this environment.")
            else:
                obs = self.env.dispatch(self, tool, args)
        except (ToolError, SchemaError) as e:
            err = str(e)
            obs = dict(status="ERROR", message=err)
            self.stats["tool_errors"] += 1
            self.stats[f"err_{classify_error(tool, err)}"] += 1
        except Exception as e:                                   # environment failure, not the model
            err = f"{type(e).__name__}: {e}"
            obs = dict(status="ERROR", message=err)
            self.stats["tool_errors"] += 1
            self.stats["err_internal_error"] += 1
        if tool in RELATION_TOOLS:
            self.stats["relation_call_attempts"] += 1
            if isinstance(obs, dict) and obs.get("status") == "OK":
                self.stats["relation_calls_successful"] += 1
        if tool in GRAPH_EXPOSING_TOOLS and isinstance(obs, dict) and obs.get("status") == "OK":
            self.stats["graph_exposing_observations"] += 1
        if isinstance(obs, dict) and obs.get("status") == "OK" and not obs.get("observation_ids"):
            self.stats["err_empty_result"] += 1
        ids = obs.get("observation_ids", []) if isinstance(obs, dict) else []
        tok = ntok(obs)
        self.observed_ids += ids
        self.observed_tokens += tok
        self.trajectory.append(dict(step=len(self.trajectory) + 1, tool=tool, arguments=arguments,
                                    observation_ids=ids, observation_token_count=tok,
                                    latency_ms=int((time.time() - t0) * 1000), error=err))
        return obs

    def summary(self):
        return dict(instance_id=self.instance["instance_id"], condition=self.condition,
                    task=self.task, variant=self.variant, prompt_version=self.prompt_version,
                    tool_calls=self.calls_used, observed_tokens=self.observed_tokens,
                    answered=self.answer is not None, stats=dict(self.stats),
                    latency_ms=sum(s["latency_ms"] for s in self.trajectory))


# ---------------- Task B environment ----------------

class RTLocEnv:
    """Serves one pre-built per-instance graph (nodes = paper/notes/paragraphs/sections/labels)."""

    def __init__(self, graphs_by_variant, store: TemporalStore = None):
        self.graphs = graphs_by_variant            # {variant: {inst_id: graph}}
        self.store = store or TemporalStore()
        self._bm25 = {}

    def graph(self, ep):
        return self.graphs[ep.variant][ep.instance["instance_id"]]

    def _paras(self, g):
        return {k: v for k, v in g["nodes"].items() if v["type"] == "paragraph"}

    def bm25(self, ep):
        key = (ep.instance["instance_id"],)
        if key not in self._bm25:
            g = self.graphs["full"][ep.instance["instance_id"]]
            self._bm25[key] = BM25({k: v["text"] for k, v in self._paras(g).items()})
        return self._bm25[key]

    def dispatch(self, ep, tool, a):
        g = self.graph(ep)
        as_of = ep.instance["as_of"]
        nodes = g["nodes"]
        if tool == "search_papers":
            hits = self.bm25(ep).search(a["query"], a["k"])
            return dict(status="OK", results=[dict(paragraph_id=i, score=round(s, 3),
                                                   text=nodes[i]["text"][:600]) for i, s in hits],
                        observation_ids=[i for i, _ in hits])
        if tool == "get_paragraphs":
            want = a.get("ids_or_range")
            paras = sorted(self._paras(g), key=lambda k: nodes[k]["idx"])
            if isinstance(want, list) and len(want) == 2 and all(isinstance(x, int) for x in want):
                sel = [p for p in paras if want[0] <= nodes[p]["idx"] <= want[1]][:20]
            elif isinstance(want, list) and want:
                sel = [f"p{x}" if not str(x).startswith("p") else x for x in want][:20]
                sel = [s for s in sel if s in nodes]
            else:
                sel = paras[:20]
            return dict(status="OK", paragraphs=[dict(paragraph_id=p, idx=nodes[p]["idx"],
                                                      text=nodes[p]["text"][:1200]) for p in sel],
                        observation_ids=sel)
        if tool in ("get_reviews", "get_thread"):
            notes = [(k, v) for k, v in nodes.items() if v["type"] == "note"]
            legal = [(k, v) for k, v in notes if normalize_ts(v.get("time")) <= normalize_ts(as_of)]
            ep.stats["filtered_future_notes"] += len(notes) - len(legal)
            if tool == "get_reviews":
                legal = [(k, v) for k, v in legal if v.get("ntype") == "official_review"]
            reply = {a_: b for a_, b, r in g["edges"] if r == "reply_to" and a_.startswith("n")}
            return dict(status="OK", notes=[dict(note_id=k, type=v.get("ntype"), time=v.get("time"),
                                                 reply_to=reply.get(k), text=v["text"][:1500])
                                            for k, v in sorted(legal, key=lambda x: x[1]["time"])],
                        observation_ids=[k for k, _ in legal])
        if tool == "get_neighbors":
            nid, rt, k = a["node_id"], a.get("relation_types"), a["k"]
            if nid not in nodes:
                raise ToolError(f"unknown node_id '{nid}'")
            out = []
            for s, t, r in g["edges"]:
                if rt and r not in rt:
                    continue
                if s == nid:
                    out.append((t, r))
                elif t == nid:
                    out.append((s, r))
            out = out[:k]
            return dict(status="OK", neighbors=[dict(node_id=n, relation=r, type=nodes[n]["type"],
                                                     text=str(nodes[n].get("text", ""))[:300])
                                                for n, r in out],
                        observation_ids=[n for n, _ in out])
        if tool == "find_paths":
            return _find_paths(g["edges"], a["source_ids"], a["target_id"], a["max_hops"], nodes)
        if tool == "inspect_history":
            notes = [v for v in nodes.values() if v["type"] == "note"
                     and normalize_ts(v.get("time")) <= normalize_ts(as_of)]
            return dict(status="OK", as_of=as_of, n_notes_available=len(notes),
                        n_paragraphs=len(self._paras(g)),
                        first_note_time=min([v["time"] for v in notes], default=None),
                        last_note_time=max([v["time"] for v in notes], default=None),
                        observation_ids=[])
        raise ToolError(f"tool '{tool}' not implemented for this task")


# ---------------- Task A environment ----------------

class LiteratureEvidenceEnv:
    """Serves the temporally filtered paper corpus + the active citation-graph variant."""

    def __init__(self, papers, bm25, adjacency_by_variant, store: TemporalStore,
                 paragraph_lookup=None, para_owner=None):
        self.papers = papers                       # {paper_id: {title, abstract, date}}
        self._bm25 = bm25
        self.adj = adjacency_by_variant            # {variant: {node: [(nbr, rel, direction)]}}
        self.store = store
        self.paragraph_lookup = paragraph_lookup or (lambda pid, rng: [])
        self.para_owner = para_owner or {}         # para:<id> -> owning paper id

    def _legal(self, pid, as_of):
        return self.store.is_visible(f"paper:{pid}", as_of)

    def node_legal(self, node_id, as_of):
        """Temporal legality for ANY graph node: papers by their own date, paragraphs by the date
        of the paper that contains them (a paragraph of a future paper is future content)."""
        if node_id.startswith("paper:"):
            return self._legal(node_id.split(":", 1)[1], as_of)
        if node_id.startswith("para:"):
            owner = self.para_owner.get(node_id)
            return self._legal(owner, as_of) if owner else False
        return True

    def dispatch(self, ep, tool, a):
        as_of = ep.instance["as_of"]
        if tool == "search_papers":
            hits = self._bm25.search(a["query"], a["k"] * 3)
            out = []
            for pid, s in hits:
                if not self._legal(pid, as_of):
                    ep.stats["filtered_future_papers"] += 1
                    continue
                p = self.papers[pid]
                out.append(dict(paper_id=pid, score=round(s, 3), title=p["title"],
                                abstract=str(p.get("abstract"))[:400], date=p["date"]))
                if len(out) >= a["k"]:
                    break
            return dict(status="OK", results=out, observation_ids=[f"paper:{o['paper_id']}" for o in out])
        if tool == "get_paper":
            pid = a["paper_id"].replace("paper:", "")
            if pid not in self.papers:
                raise ToolError(f"unknown paper_id '{pid}'")
            if not self._legal(pid, as_of):
                return dict(status="NOT_AVAILABLE_AT_CUTOFF", paper_id=pid, observation_ids=[])
            p = self.papers[pid]
            return dict(status="OK", paper_id=pid, title=p["title"], date=p["date"],
                        abstract=str(p.get("abstract"))[:1500], observation_ids=[f"paper:{pid}"])
        if tool == "get_paragraphs":
            pid = str(a.get("paper_id", "")).replace("paper:", "")
            if not self._legal(pid, as_of):
                return dict(status="NOT_AVAILABLE_AT_CUTOFF", observation_ids=[])
            paras = self.paragraph_lookup(pid, a.get("ids_or_range"))
            # paragraphs materialized from the database inherit their (already legality-checked)
            # paper's date, so the independent auditor can verify them later
            date = self.papers.get(pid, {}).get("date")
            for p in paras[:10]:
                self.store.add(TemporalObject(f"para:{p['paragraph_id']}", "paragraph", date))
            return dict(status="OK", paragraphs=paras[:10],
                        observation_ids=[f"para:{p['paragraph_id']}" for p in paras[:10]])
        if tool in ("get_citations", "get_neighbors"):
            if tool == "get_citations":
                node = f"paper:{a['paper_id'].replace('paper:', '')}"
                rels, direction, k = ["cites"], a["direction"], a["k"]
            else:
                node, rels, k = a["node_id"], a.get("relation_types"), a["k"]
                direction = None
            out = []
            for nbr, rel, d in self.adj[ep.variant].get(node, []):
                if (node, nbr) in ep.masked_pairs:
                    ep.stats["masked_gold_edges_hidden"] += 1
                    continue
                if rels and rel not in rels:
                    continue
                if direction and d != direction:
                    continue
                if not self.node_legal(nbr, as_of):
                    ep.stats["filtered_future_edges"] += 1
                    continue
                meta = self.papers.get(nbr.split(":", 1)[1], {}) if nbr.startswith("paper:") else {}
                out.append(dict(node_id=nbr, relation=rel, direction=d, title=meta.get("title"),
                                date=meta.get("date")))
                if len(out) >= k:
                    break
            return dict(status="OK", neighbors=out, observation_ids=[o["node_id"] for o in out])
        if tool == "find_paths":
            # paths may only traverse nodes that already existed at the cutoff
            edges = [(s, t, r) for s, nb in self.adj[ep.variant].items() for t, r, d in nb
                     if d == "out" and (s, t) not in ep.masked_pairs
                     and self.node_legal(s, as_of) and self.node_legal(t, as_of)]
            out = _find_paths(edges, [x for x in a["source_ids"] if self.node_legal(x, as_of)],
                              a["target_id"], a["max_hops"], {})
            out["observation_ids"] = [i for i in out["observation_ids"] if self.node_legal(i, as_of)]
            return out
        if tool == "inspect_history":
            pid = str(a.get("paper_id", ep.instance["source_paper_id"])).replace("paper:", "")
            p = self.papers.get(pid, {})
            return dict(status="OK", paper_id=pid, as_of=as_of, first_version_date=p.get("date"),
                        observation_ids=[])
        raise ToolError(f"tool '{tool}' not implemented for this task")


def _find_paths(edges, sources, target, max_hops, nodes, max_paths=5):
    adj = defaultdict(list)
    for s, t, r in edges:
        adj[s].append((t, r))
        adj[t].append((s, r))
    found = []
    for src in sources[:5]:
        stack = [(src, [src], [])]
        seen = {src}
        while stack and len(found) < max_paths:
            node, path, rels = stack.pop()
            if len(path) - 1 >= max_hops:
                continue
            for nbr, rel in adj.get(node, []):
                if nbr in path:
                    continue
                if nbr == target:
                    found.append(dict(nodes=path + [nbr], relations=rels + [rel]))
                    if len(found) >= max_paths:
                        break
                elif nbr not in seen:
                    seen.add(nbr)
                    stack.append((nbr, path + [nbr], rels + [rel]))
    ids = sorted({n for p in found for n in p["nodes"]})
    return dict(status="OK" if found else "NO_PATH_FOUND", paths=found, observation_ids=ids)
