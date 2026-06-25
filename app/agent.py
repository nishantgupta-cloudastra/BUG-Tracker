"""The triage agent: an agentic-RAG tool loop running on whichever LLM provider
is active (Anthropic -> OpenAI-compatible -> local Ollama).

Tools exposed to the model:
  * search_existing_issues(query, k) -> hybrid (dense+BM25/RRF) retrieval. The
    model can call this several times (query expansion) to hunt duplicates/context.
  * submit_triage(...)               -> structured output; ends the loop. On the
    final iteration it is forced, so even weaker local models emit valid output.
"""
from __future__ import annotations

from typing import Optional

from .llm import get_active_provider, invalidate
from .llm.base import ProviderError, ToolDef, tool_result_msg, user_msg
from .models import RetrievedContext, TriageResponse, TriagedIssue
from .vectorstore import get_store

MAX_ITERS = 6

SYSTEM_PROMPT = """You are an AI Bug Triage & Release Operator for a software engineering team.
You turn messy, raw feedback (from Slack, email, forms, GitHub) into a clean, prioritized, \
de-duplicated engineering issue.

Workflow:
1. Use the `search_existing_issues` tool to look for DUPLICATES and relevant context. \
Search more than once if useful — try the user's words, then rephrase with likely component \
or error terms (query expansion). Only cite a duplicate whose id actually appeared in a search result.
2. Then call `submit_triage` EXACTLY ONCE with your final structured triage.

Priority rubric:
- P0: outage / data loss / security / billing-incorrect-charges — affects many users, no workaround.
- P1: major feature broken, common path, workaround is painful.
- P2: real bug but limited scope or easy workaround.
- P3: minor / cosmetic / nice-to-have / feature request.

Be specific in reproduction_steps. If the report lacks detail, infer the most likely steps and \
say so in suggested_next_action. Keep the title short and searchable.
Always finish by calling submit_triage."""

SEARCH_TOOL = ToolDef(
    name="search_existing_issues",
    description="Hybrid semantic + keyword search over existing issues and product docs. "
    "Use to find duplicate issues and grounding context before triaging.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "k": {"type": "integer", "description": "Number of results (default 5)."},
        },
        "required": ["query"],
    },
)

SUBMIT_TOOL = ToolDef(
    name="submit_triage",
    description="Submit the final structured triage for this report. Call exactly once.",
    strict=True,
    parameters={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "type": {"type": "string", "enum": ["bug", "feature_request", "question", "task"]},
            "priority": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
            "priority_reason": {"type": "string"},
            "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
            "components": {"type": "array", "items": {"type": "string"}},
            "labels": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
            "reproduction_steps": {"type": "array", "items": {"type": "string"}},
            "expected_behavior": {"type": "string"},
            "actual_behavior": {"type": "string"},
            "duplicate_candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "title", "confidence", "reason"],
                },
            },
            "suggested_next_action": {"type": "string"},
        },
        "required": [
            "title", "type", "priority", "priority_reason", "severity", "components",
            "labels", "summary", "reproduction_steps", "expected_behavior",
            "actual_behavior", "duplicate_candidates", "suggested_next_action",
        ],
    },
)


def _run_search(query: str, k: int = 5) -> tuple[str, list[RetrievedContext]]:
    hits = get_store().search(query, k=k)
    contexts: list[RetrievedContext] = []
    lines = []
    for h in hits:
        snippet = h.doc.text[:240].replace("\n", " ")
        contexts.append(
            RetrievedContext(
                id=h.doc.id, title=h.doc.title, snippet=snippet,
                score=h.score, source=h.doc.source, url=h.doc.url,
            )
        )
        lines.append(f"- id={h.doc.id} | {h.doc.title} ({h.doc.source}) | score={h.score}\n    {snippet}")
    return ("\n".join(lines) if lines else "(no results)"), contexts


def _run_triage(provider, raw_text: str, source: str, reporter: Optional[str]) -> TriageResponse:
    seed_text, seed_ctx = _run_search(raw_text, k=5)
    all_contexts: dict[str, RetrievedContext] = {c.id: c for c in seed_ctx}
    agent_searches: list[str] = []

    user_text = (
        f"Source: {source}\nReporter: {reporter or 'unknown'}\n\n"
        f"--- RAW REPORT ---\n{raw_text}\n\n"
        f"--- INITIAL SEARCH RESULTS (for grounding; search more if useful) ---\n{seed_text}"
    )
    messages = [user_msg(user_text)]

    triaged: Optional[TriagedIssue] = None
    for i in range(MAX_ITERS):
        # On the final allowed call, force submit_triage so we always get output.
        force = "submit_triage" if i == MAX_ITERS - 1 else None
        turn = provider.chat(
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=[SEARCH_TOOL, SUBMIT_TOOL],
            max_tokens=8000,
            force_tool=force,
        )
        messages.append(turn.assistant_message)
        if not turn.tool_calls:
            break

        for tc in turn.tool_calls:
            if tc.name == "submit_triage":
                triaged = TriagedIssue(**tc.arguments)
                messages.append(tool_result_msg(tc.id, "Triage recorded."))
            elif tc.name == "search_existing_issues":
                q = tc.arguments.get("query", "")
                k = int(tc.arguments.get("k", 5) or 5)
                agent_searches.append(q)
                text, ctx = _run_search(q, k=k)
                for c in ctx:
                    all_contexts.setdefault(c.id, c)
                messages.append(tool_result_msg(tc.id, text))
            else:
                messages.append(tool_result_msg(tc.id, f"Unknown tool: {tc.name}"))
        if triaged is not None:
            break

    if triaged is None:
        raise RuntimeError("The model did not produce a triage (submit_triage was never called).")

    retrieved = sorted(all_contexts.values(), key=lambda c: -c.score)[:6]
    return TriageResponse(triage=triaged, retrieved=retrieved, agent_searches=agent_searches)


def triage(raw_text: str, source: str = "other", reporter: Optional[str] = None) -> TriageResponse:
    last_err: Optional[Exception] = None
    for _ in range(2):  # primary provider, then one failover
        provider = get_active_provider()
        if provider is None:
            raise RuntimeError(
                "No LLM provider available. Set ANTHROPIC_API_KEY, or OPENAI_COMPAT_* vars, "
                "or run a local Ollama model."
            )
        try:
            return _run_triage(provider, raw_text, source, reporter)
        except ProviderError as exc:
            last_err = exc
            if not exc.recoverable:
                raise RuntimeError(str(exc))
            invalidate()  # re-select a different provider and retry
    raise RuntimeError(f"All LLM providers failed: {last_err}")


# ---- Release notes generation (a normal grounded LLM call) ----
RELEASE_SYSTEM = """You write concise, professional release notes for an engineering team.
Group the resolved issues into sections: New Features, Bug Fixes, Improvements. Omit empty \
sections. Each line should be user-facing and reference the issue number like (#123). Output \
GitHub-flavored Markdown only — start with a '## Release Notes' heading."""


def generate_release_notes(items: list[dict]) -> str:
    listing = "\n".join(
        f"- #{it['number']} {it['title']} [labels: {', '.join(it.get('labels', [])) or 'none'}]"
        for it in items
    )
    last_err: Optional[Exception] = None
    for _ in range(2):
        provider = get_active_provider()
        if provider is None:
            raise RuntimeError("No LLM provider available for release notes.")
        try:
            turn = provider.chat(
                system=RELEASE_SYSTEM,
                messages=[user_msg(f"Resolved issues:\n{listing}")],
                max_tokens=2000,
            )
            return turn.text.strip()
        except ProviderError as exc:
            last_err = exc
            if not exc.recoverable:
                raise RuntimeError(str(exc))
            invalidate()
    raise RuntimeError(f"All LLM providers failed: {last_err}")
