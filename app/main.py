"""FastAPI app: AI Bug Triage & Release Operator backend + static UI.

Genuine end-to-end: real GitHub ingestion (issues + repo markdown docs, chunked),
real Qdrant vector DB, real embeddings, agentic-RAG triage, real issue creation.
No seed data — the knowledge base is whatever you ingest.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import agent
from .chunking import chunk_text
from .config import settings
from .integrations import github_client, plane_client, slack_client
from .llm import describe as describe_llm
from .llm import get_active_provider
from .models import (
    CreateIssueRequest,
    CreateIssueResponse,
    IngestRequest,
    IngestResponse,
    ReleaseNotesRequest,
    ReleaseNotesResponse,
    TriageRequest,
    TriageResponse,
    TriagedIssue,
)
from .vectorstore import Doc, get_store

app = FastAPI(title="AI Bug Triage & Release Operator")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.on_event("startup")
def _startup() -> None:
    s = get_store()  # opens Qdrant, probes embedding dim, loads existing corpus
    print(f"[startup] vector store ready: {s.stats()} | embeddings={settings.effective_embedding_provider}")


@app.on_event("shutdown")
def _shutdown() -> None:
    get_store().close()


# ---------------- API ----------------
@app.get("/api/status")
def status() -> dict:
    return {
        "llm": describe_llm(),  # active provider + per-tier availability (fallback chain)
        "github_write": settings.github_can_write,
        "slack": settings.has_slack,
        "plane_live": settings.plane_is_live,
        "embedding_provider": settings.effective_embedding_provider,
        "embeddings_real": settings.embeddings_are_real,
        "vector_db": f"qdrant ({settings.qdrant_mode})",
        "store": get_store().stats(),
    }


@app.get("/api/slack/channels")
def slack_channels() -> dict:
    if not slack_client.is_configured():
        raise HTTPException(400, "Slack is not configured. Set SLACK_BOT_TOKEN.")
    try:
        return {"channels": slack_client.list_channels()}
    except Exception as exc:
        raise HTTPException(400, f"Slack error: {exc}")


@app.get("/api/slack/messages")
def slack_messages(channel_id: str, limit: int = 20) -> dict:
    if not slack_client.is_configured():
        raise HTTPException(400, "Slack is not configured. Set SLACK_BOT_TOKEN.")
    try:
        return {"messages": slack_client.fetch_messages(channel_id, limit=limit)}
    except Exception as exc:
        raise HTTPException(400, f"Slack error: {exc}")


@app.post("/api/triage", response_model=TriageResponse)
def triage(req: TriageRequest) -> TriageResponse:
    if get_active_provider() is None:
        raise HTTPException(
            400,
            "No LLM provider available. Set ANTHROPIC_API_KEY, or OPENAI_COMPAT_* vars, "
            "or run a local Ollama model.",
        )
    try:
        return agent.triage(req.raw_text, source=req.source, reporter=req.reporter)
    except Exception as exc:
        raise HTTPException(500, f"Triage failed: {exc}")


def _issue_body(t: TriagedIssue) -> str:
    lines = [t.summary, ""]
    if t.reproduction_steps:
        lines.append("### Steps to reproduce")
        lines += [f"{i}. {s}" for i, s in enumerate(t.reproduction_steps, 1)]
        lines.append("")
    if t.expected_behavior:
        lines += ["### Expected", t.expected_behavior, ""]
    if t.actual_behavior:
        lines += ["### Actual", t.actual_behavior, ""]
    lines += [
        "### Triage",
        f"- **Type:** {t.type}",
        f"- **Priority:** {t.priority} — {t.priority_reason}",
        f"- **Severity:** {t.severity}",
        f"- **Components:** {', '.join(t.components) or 'n/a'}",
    ]
    if t.duplicate_candidates:
        lines += ["", "### Possible duplicates"]
        for d in t.duplicate_candidates:
            lines.append(f"- `{d.id}` {d.title} — {d.confidence} ({d.reason})")
    if t.suggested_next_action:
        lines += ["", "### Suggested next action", t.suggested_next_action]
    lines += ["", "_Filed by AI Bug Triage & Release Operator._"]
    return "\n".join(lines)


@app.post("/api/create-issue", response_model=CreateIssueResponse)
def create_issue(req: CreateIssueRequest) -> CreateIssueResponse:
    t = req.triage
    body = _issue_body(t)
    try:
        if req.target == "github":
            if not req.repo_url:
                raise HTTPException(400, "repo_url is required to create a GitHub issue.")
            owner, repo = github_client.parse_repo(req.repo_url)
            res = github_client.create_issue(owner, repo, t.title, body, t.labels)
            return CreateIssueResponse(
                ok=True, target="github", id=res["id"], url=res["url"],
                message=f"Created GitHub issue #{res['id']}",
            )
        res = plane_client.create_issue(t.title, body, priority=t.priority, labels=t.labels)
        return CreateIssueResponse(
            ok=True, target="plane", id=res["id"], url=res["url"], message="Created Plane issue",
        )
    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Create issue failed: {exc}")


@app.post("/api/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """Ingest a repo's real issues AND markdown docs into the vector DB (chunked)."""
    store = get_store()
    try:
        owner, repo = github_client.parse_repo(req.repo_url)
        slug = f"{owner}/{repo}"
        docs: list[Doc] = []

        # 1) Issues (each issue body chunked if long; parent linkage in metadata).
        issues = github_client.list_issues(owner, repo, state=req.state, limit=req.limit)
        for it in issues:
            chunks = chunk_text(it["body"]) or [""]
            for ci, chunk in enumerate(chunks):
                suffix = "" if len(chunks) == 1 else f"::chunk{ci}"
                docs.append(
                    Doc(
                        id=f"{slug}#{it['number']}{suffix}",
                        title=it["title"],
                        text=chunk,
                        source="github_issue",
                        url=it["url"],
                        metadata={
                            "state": it["state"], "repo": slug, "number": it["number"],
                            "labels": it["labels"], "chunk": ci,
                        },
                    )
                )

        # 2) Repo markdown docs (README, docs/**) — real document loading + chunking.
        branch = github_client.get_default_branch(owner, repo)
        for path in github_client.list_markdown_files(owner, repo, branch=branch, limit=20):
            try:
                text = github_client.get_file_text(owner, repo, path, branch=branch)
            except Exception:
                continue
            for ci, chunk in enumerate(chunk_text(text)):
                docs.append(
                    Doc(
                        id=f"{slug}:{path}::chunk{ci}",
                        title=f"{path}",
                        text=chunk,
                        source="doc",
                        url=github_client.file_html_url(owner, repo, path, branch),
                        metadata={"repo": slug, "path": path, "chunk": ci},
                    )
                )
    except Exception as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 404:
            raise HTTPException(
                400,
                f"GitHub returned 404 for '{req.repo_url}'. This almost always means the repo is "
                "PRIVATE and your GITHUB_TOKEN can't access it (GitHub hides private repos as 404). "
                "Check: (1) the owner/repo path is exact; (2) if it's an organization repo, the "
                "fine-grained token's Resource owner is that ORG (not your personal account) with "
                "Contents: Read + Issues: Read/write, and the org allows/approved the token; or use a "
                "classic token with the 'repo' scope. Test: "
                "curl -H 'Authorization: Bearer <TOKEN>' https://api.github.com/repos/<owner>/<repo>",
            )
        if status_code in (401, 403):
            raise HTTPException(400, f"GitHub auth error ({status_code}) — your GITHUB_TOKEN is invalid or lacks scope.")
        raise HTTPException(500, f"GitHub ingest failed: {exc}")

    # Embedding happens here — surface rate-limit / provider errors cleanly (not as a 500).
    try:
        added = store.add(docs)
    except Exception as exc:
        raise HTTPException(400, f"Embedding failed during ingest: {exc}")
    return IngestResponse(
        ingested=added,
        total_in_store=store.stats()["total"],
        message=f"Ingested {added} new chunks from {slug} "
        f"({len(issues)} issues + repo docs).",
    )


@app.post("/api/release-notes", response_model=ReleaseNotesResponse)
def release_notes(req: ReleaseNotesRequest) -> ReleaseNotesResponse:
    if get_active_provider() is None:
        raise HTTPException(400, "No LLM provider available for release notes.")
    store = get_store()
    owner, repo = github_client.parse_repo(req.repo_url)
    docs = store.closed_issues(repo=f"{owner}/{repo}")
    # de-dup by issue number (chunking can produce multiple docs per issue)
    seen, items = set(), []
    for d in docs:
        num = d.metadata.get("number", d.id)
        if num in seen:
            continue
        seen.add(num)
        items.append({"number": num, "title": d.title, "labels": d.metadata.get("labels", []), "url": d.url})
        if len(items) >= req.limit:
            break
    if not items:
        return ReleaseNotesResponse(
            markdown="_No resolved (closed) issues found for this repo. Ingest it first "
            "with state=all or state=closed._",
            issue_count=0,
        )
    md = agent.generate_release_notes(items)
    return ReleaseNotesResponse(markdown=md, issue_count=len(items))


# ---------------- Static UI ----------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
