"""Policies behind one interface: deterministic mock (tests), vLLM chat model, replay (debug)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import urllib.request
from pathlib import Path

from .tools import tool_manual

JSON_BLOCK = re.compile(r"\{.*\}", re.S)
THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.S)


def strip_think(text):
    """Reasoning models (Qwen3) emit <think>...</think>; the answer is what follows."""
    return THINK_BLOCK.sub("", text or "").strip()


class ResponseCache:
    """Transcript-keyed cache of model responses.

    Experimental invariant: identical (model, prompt_version, transcript) MUST produce an identical
    response, whatever graph condition is active. Without this, continuous batching on the server
    makes temperature-0 decoding non-reproducible and conditions diverge before any
    structure-dependent observation appears.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.mem, self.lock = {}, threading.Lock()
        self.hits = self.misses = 0
        if self.path and self.path.exists():
            for line in open(self.path):
                try:
                    r = json.loads(line)
                    self.mem[r["key"]] = r["response"]
                except Exception:
                    pass

    @staticmethod
    def key(model, prompt_version, messages, temperature):
        blob = json.dumps(dict(model=model, pv=prompt_version, t=temperature, m=messages),
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, k):
        v = self.mem.get(k)
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, k, response):
        with self.lock:
            self.mem[k] = response
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(json.dumps(dict(key=k, response=response)) + "\n")


class Policy:
    name = "policy"
    prompt_version = "v1"

    def reset(self, episode, system_prompt, user_prompt):
        self.messages = [dict(role="system", content=system_prompt),
                         dict(role="user", content=user_prompt)]

    def act(self, episode):
        raise NotImplementedError

    def observe(self, tool, arguments, observation):
        self.messages.append(dict(role="assistant",
                                  content=json.dumps(dict(tool=tool, arguments=arguments))))
        self.messages.append(dict(role="user", content=f"OBSERVATION:\n{json.dumps(observation)[:6000]}"))


class MockPolicy(Policy):
    """Deterministic scripted agent: one search, one relation probe, then submit.

    Used by unit tests and as a no-model control; never calls a network service.
    """
    name = "mock"

    def act(self, episode):
        task = episode.task
        seen = [s["tool"] for s in episode.trajectory]
        inst = episode.instance
        if task == "rtloc":
            if "search_papers" not in seen:
                notes = inst.get("query_text", "review")
                return "search_papers", dict(query=notes[:400], k=10)
            if "get_neighbors" not in seen and episode.condition != "flat_agent":
                first = self._first_ids(episode)
                if first:
                    return "get_neighbors", dict(node_id=first[0], k=5)
            ranked = list(dict.fromkeys((self._first_ids(episode) or []) +
                                        [f"p{i}" for i in range(1, 21)]))
            return "submit_answer", dict(payload=dict(ranked_paragraph_ids=ranked[:10],
                                                      evidence_note_ids=[], confidence=0.3))
        if "search_papers" not in seen:
            return "search_papers", dict(query=inst.get("query_text", inst.get("masked_text", ""))[:400], k=10)
        ranked = [i.split(":", 1)[1] for i in self._first_ids(episode) if i.startswith("paper:")]
        return "submit_answer", dict(payload=dict(ranked_paper_ids=ranked[:5] or ["unknown"] * 5,
                                                  evidence_node_ids=[], evidence_edges=[],
                                                  rationale="mock", confidence=0.2))

    @staticmethod
    def _first_ids(episode):
        for s in episode.trajectory:
            if s["tool"] == "search_papers" and s["observation_ids"]:
                return list(s["observation_ids"])
        return []


class VLLMPolicy(Policy):
    """OpenAI-compatible chat completion against a local vLLM server (temperature 0)."""

    def __init__(self, model, base_url=None, max_tokens=512, temperature=0.0, timeout=180,
                 cache: "ResponseCache" = None):
        self.cache = cache
        self.model = model
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout
        self.name = f"vllm:{model}"
        self.last_raw = None

    def _chat(self, messages):
        k = None
        if self.cache is not None:
            k = ResponseCache.key(self.model, self.prompt_version, messages, self.temperature)
            hit = self.cache.get(k)
            if hit is not None:
                return hit
        out = self._chat_remote(messages)
        if self.cache is not None:
            self.cache.put(k, out)
        return out

    def _chat_remote(self, messages):
        payload = dict(model=self.model, messages=messages, temperature=self.temperature,
                       max_tokens=self.max_tokens)
        if "qwen3" in self.model.lower():
            # Qwen3 is a hybrid reasoning model: without this it spends the whole budget thinking
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(payload).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def act(self, episode):
        calls_left, tokens_left = episode.budget_left()
        self.messages.append(dict(role="user", content=(
            f"Budget: {calls_left} tool calls and about {tokens_left} observation tokens remain. "
            "Reply with ONE JSON object: {\"tool\": ..., \"arguments\": {...}}.")))
        raw = self._chat(self.messages)
        self.last_raw = raw
        self.messages.pop()
        try:
            return self._parse(raw)
        except Exception:
            # one retry, charged to the same budget (spec 3.2)
            self.messages.append(dict(role="user", content=(
                "Your last reply was not a single valid JSON object. Reply with exactly one JSON "
                "object of the form {\"tool\": \"...\", \"arguments\": {...}} and nothing else.")))
            raw = self._chat(self.messages)
            self.last_raw = raw
            self.messages.pop()
            return self._parse(raw)

    @staticmethod
    def _parse(raw):
        raw = strip_think(raw)
        m = JSON_BLOCK.search(raw)
        if not m:
            raise ValueError("no JSON found")
        d = json.loads(m.group(0))
        return d["tool"], d.get("arguments", {})


class ReplayPolicy(Policy):
    """Replays a stored trajectory (debugging; no model calls)."""

    name = "replay"

    def __init__(self, trajectory):
        self.steps = list(trajectory)
        self.i = 0

    def act(self, episode):
        if self.i >= len(self.steps):
            return "submit_answer", dict(payload=dict(ranked_paper_ids=["unknown"],
                                                      ranked_paragraph_ids=["p1"]))
        s = self.steps[self.i]
        self.i += 1
        return s["tool"], s["arguments"]


def build_system_prompt(task):
    return (
        "You are a research assistant working inside a historical snapshot of the scientific "
        "literature and its peer-review record. You can only observe what existed at the given "
        "cutoff time.\n\n"
        f"Available tools:\n{tool_manual(task)}\n\n"
        "Rules:\n"
        "- Reply with exactly one JSON object per turn: {\"tool\": \"<name>\", \"arguments\": {...}}.\n"
        "- Search broadly first, then verify candidates before answering.\n"
        "- Use only evidence you actually observed; never invent identifiers or titles.\n"
        "- Return stable IDs exactly as they appear in observations.\n"
        "- Tool errors and unavailable relations are normal environment responses; adapt and continue.\n"
        "- Uncertainty is acceptable; give your best ranked answer.\n"
        "- Stop as soon as you have enough evidence by calling submit_answer.")


class ControlledGraphPolicy(Policy):
    """Controlled graph exposure (spec follow-up, Setting 2).

    Every condition executes the SAME research pipeline with the SAME number of calls and the same
    result sizes: text search -> inspect seeds -> relation expansion -> path inspection -> rerank.
    Only the relations behind the graph tools differ, so the structural contrast is actually active.
    The model is used once, at the end, to rank the candidate pool it was shown.
    """

    name = "controlled"

    def __init__(self, llm: "VLLMPolicy", k_search=10, n_seeds=2, k_neighbors=10):
        self.llm, self.k_search, self.n_seeds, self.k_neighbors = llm, k_search, n_seeds, k_neighbors

    def reset(self, episode, system_prompt, user_prompt):
        self.messages = [dict(role="system", content=system_prompt),
                         dict(role="user", content=user_prompt)]
        self.llm.reset(episode, system_prompt, user_prompt)
        self.plan = None

    def _seeds(self, episode):
        for s in episode.trajectory:
            if s["tool"] == "search_papers" and s["observation_ids"]:
                return list(s["observation_ids"])
        return []

    def act(self, episode):
        inst = episode.instance
        done = [s["tool"] for s in episode.trajectory]
        q = (inst.get("query_text") or inst.get("masked_text") or "review")[:400]
        if not done:
            return "search_papers", dict(query=q, k=self.k_search)
        seeds = self._seeds(episode)
        if inst["task"] == "literature_evidence":
            step = len(done)
            if step == 1 and seeds:
                return "get_paper", dict(paper_id=seeds[0].replace("paper:", ""))
            if step == 2 and seeds:
                return "get_citations", dict(paper_id=seeds[0].replace("paper:", ""),
                                             direction="out", k=self.k_neighbors)
            if step == 3 and len(seeds) > 1:
                return "get_citations", dict(paper_id=seeds[1].replace("paper:", ""),
                                             direction="in", k=self.k_neighbors)
            if step == 4 and seeds:
                return "get_neighbors", dict(node_id=seeds[0], k=self.k_neighbors)
        else:
            step = len(done)
            if step == 1:
                return "get_reviews", {}
            if step == 2 and seeds:
                return "get_neighbors", dict(node_id=seeds[0], k=self.k_neighbors)
            if step == 3 and len(seeds) > 1:
                return "get_neighbors", dict(node_id=seeds[1], k=self.k_neighbors)
            if step == 4 and seeds:
                return "find_paths", dict(source_ids=seeds[:2], target_id=seeds[-1], max_hops=3)
        return self.llm.act(episode)      # final step: model ranks what it has seen

    def observe(self, tool, arguments, observation):
        self.llm.observe(tool, arguments, observation)
        self.messages = self.llm.messages
