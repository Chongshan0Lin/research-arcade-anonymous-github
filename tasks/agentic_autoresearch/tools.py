"""Tool declarations shared by both tasks, plus argument validation.

The environment owns state and validation; the model only proposes (tool, arguments).
Relation tools return a structured RELATIONS_UNAVAILABLE observation under `flat_agent`
instead of silently falling back to text search.
"""
from __future__ import annotations

RELATION_TOOLS = {"get_citations", "get_neighbors", "find_paths"}

TOOL_SPECS = {
    "search_papers": dict(
        description="Lexical search over papers (Task A) or paragraphs of the paper under review "
                    "(Task B) that are available at the instance cutoff.",
        arguments=dict(query="string", k="integer (default 10, max 20)")),
    "get_paper": dict(
        description="Metadata (title, abstract, date) of one paper available at the cutoff.",
        arguments=dict(paper_id="string")),
    "get_paragraphs": dict(
        description="Paragraph text with stable ids; Task A: of a paper, Task B: of the paper under review.",
        arguments=dict(paper_id="string (optional for Task B)", ids_or_range="list of ids or [start,end]")),
    "get_reviews": dict(
        description="Official reviews available at the cutoff.",
        arguments=dict(paper_id="string (optional)")),
    "get_thread": dict(
        description="Reply structure and text of the review discussion available at the cutoff.",
        arguments=dict(paper_id="string (optional)")),
    "get_citations": dict(
        description="Typed citation neighbours of a paper.",
        arguments=dict(paper_id="string", direction="'out'|'in'", k="integer (default 10, max 20)")),
    "get_neighbors": dict(
        description="Typed neighbours of any node in the active graph.",
        arguments=dict(node_id="string", relation_types="list of strings (optional)", k="integer")),
    "find_paths": dict(
        description="Paths between nodes in the active graph (<= max_hops).",
        arguments=dict(source_ids="list of strings", target_id="string", max_hops="integer (<=4)")),
    "inspect_history": dict(
        description="Version/review timeline metadata available at the cutoff.",
        arguments=dict(paper_id="string (optional)")),
    "submit_answer": dict(
        description="Final answer; ends the episode.",
        arguments=dict(payload="task-specific object")),
}

TASK_TOOLS = {
    "literature_evidence": ["search_papers", "get_paper", "get_paragraphs", "get_citations",
                            "get_neighbors", "find_paths", "inspect_history", "submit_answer"],
    "rtloc": ["search_papers", "get_paragraphs", "get_reviews", "get_thread", "get_neighbors",
              "find_paths", "inspect_history", "submit_answer"],
}


class ToolError(Exception):
    """Raised for malformed calls; the environment turns it into an error observation."""


def validate_call(task, tool, arguments):
    if tool not in TOOL_SPECS:
        raise ToolError(f"unknown tool '{tool}'. Available: {', '.join(TASK_TOOLS[task])}")
    if tool not in TASK_TOOLS[task]:
        raise ToolError(f"tool '{tool}' is not available for this task")
    if not isinstance(arguments, dict):
        raise ToolError("arguments must be a JSON object")
    a = dict(arguments)
    if tool == "search_papers":
        if not isinstance(a.get("query"), str) or not a["query"].strip():
            raise ToolError("search_papers requires a non-empty string 'query'")
        a["k"] = min(int(a.get("k", 10) or 10), 20)
    if tool in ("get_paper", "get_citations") and not isinstance(a.get("paper_id"), str):
        raise ToolError(f"{tool} requires string 'paper_id'")
    if tool == "get_citations":
        a["direction"] = a.get("direction", "out")
        if a["direction"] not in ("out", "in"):
            raise ToolError("get_citations 'direction' must be 'out' or 'in'")
        a["k"] = min(int(a.get("k", 10) or 10), 20)
    if tool == "get_neighbors":
        if not isinstance(a.get("node_id"), str):
            raise ToolError("get_neighbors requires string 'node_id'")
        rt = a.get("relation_types")
        if rt is not None and not isinstance(rt, list):
            raise ToolError("get_neighbors 'relation_types' must be a list of strings")
        a["k"] = min(int(a.get("k", 10) or 10), 20)
    if tool == "find_paths":
        if not isinstance(a.get("source_ids"), list) or not a["source_ids"]:
            raise ToolError("find_paths requires a non-empty list 'source_ids'")
        if not isinstance(a.get("target_id"), str):
            raise ToolError("find_paths requires string 'target_id'")
        a["max_hops"] = min(int(a.get("max_hops", 3) or 3), 4)
    if tool == "submit_answer" and not isinstance(a.get("payload", a), dict):
        raise ToolError("submit_answer requires an object payload")
    return a


def tool_manual(task):
    lines = []
    for t in TASK_TOOLS[task]:
        s = TOOL_SPECS[t]
        args = ", ".join(f"{k}: {v}" for k, v in s["arguments"].items())
        lines.append(f"- {t}({args}) -- {s['description']}")
    return "\n".join(lines)
