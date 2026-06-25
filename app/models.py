"""Pydantic request/response schemas for the API surface."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---- Triage input ----
class TriageRequest(BaseModel):
    raw_text: str = Field(..., description="The messy, raw bug report / feedback.")
    source: Literal["slack", "email", "github", "form", "other"] = "other"
    reporter: Optional[str] = None


# ---- Triage output (mirrors the submit_triage tool schema) ----
class DuplicateCandidate(BaseModel):
    id: str
    title: str
    confidence: Literal["high", "medium", "low"]
    reason: str


class TriagedIssue(BaseModel):
    title: str
    type: Literal["bug", "feature_request", "question", "task"]
    priority: Literal["P0", "P1", "P2", "P3"]
    priority_reason: str
    severity: Literal["critical", "high", "medium", "low"]
    components: list[str] = []
    labels: list[str] = []
    summary: str
    reproduction_steps: list[str] = []
    expected_behavior: str = ""
    actual_behavior: str = ""
    duplicate_candidates: list[DuplicateCandidate] = []
    suggested_next_action: str = ""


class RetrievedContext(BaseModel):
    id: str
    title: str
    snippet: str
    score: float
    source: str
    url: Optional[str] = None


class TriageResponse(BaseModel):
    triage: TriagedIssue
    retrieved: list[RetrievedContext]
    agent_searches: list[str] = []  # queries Claude issued itself (agentic RAG trace)


# ---- Issue creation ----
class CreateIssueRequest(BaseModel):
    triage: TriagedIssue
    target: Literal["github", "plane"] = "plane"
    repo_url: Optional[str] = None  # required for github


class CreateIssueResponse(BaseModel):
    ok: bool
    target: str
    id: str
    url: Optional[str] = None
    message: str = ""


# ---- Ingestion ----
class IngestRequest(BaseModel):
    repo_url: str
    state: Literal["open", "closed", "all"] = "all"
    limit: int = 50


class IngestResponse(BaseModel):
    ingested: int
    total_in_store: int
    message: str


# ---- Release notes ----
class ReleaseNotesRequest(BaseModel):
    repo_url: str
    limit: int = 30


class ReleaseNotesResponse(BaseModel):
    markdown: str
    issue_count: int
