import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4
import hashlib
import hmac

from fastapi import APIRouter, BackgroundTasks, FastAPI, Header, HTTPException, Request
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy.orm import Session

from switchboard.core.llm import get_llm
from switchboard.usecases.lead_qualification.db import make_engine, seed
from switchboard.usecases.lead_qualification.graph import build_graph
from switchboard.usecases.lead_qualification.tools import load_tools


class LeadIn(BaseModel):
    lead_text: str


class Decision(BaseModel):
    action: Literal["approve", "reject", "edit"]
    tool_calls: list[dict] | None = None  # only for "edit": the planned_writes, changed


class RunOut(BaseModel):
    thread_id: str
    status: Literal["running", "awaiting_review", "completed"]
    score: dict | None = None
    planned_writes: list[dict] | None = None
    summary: str | None = None


router = APIRouter()


def _config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "lead_qualification",
        "tags": ["api"],
        "metadata": {"thread_id": thread_id},
    }


def _run_out(graph, thread_id: str) -> RunOut:
    state = graph.get_state(_config(thread_id))
    if not state.values:
        raise HTTPException(404, f"No run with thread_id {thread_id!r}.")
    if state.interrupts:
        return RunOut(thread_id=thread_id, status="awaiting_review", score=state.values.get("score"),
                      planned_writes=state.interrupts[0].value["planned_writes"])
    if state.next:  # started (e.g. by a webhook) but not paused or finished yet
        return RunOut(thread_id=thread_id, status="running", score=state.values.get("score"))
    return RunOut(thread_id=thread_id, status="completed", score=state.values.get("score"),
                  summary=state.values.get("summary"))


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/runs", response_model=RunOut)
def start_run(lead: LeadIn, request: Request) -> RunOut:
    graph = request.app.state.graph
    thread_id = str(uuid4())
    graph.invoke({"lead_text": lead.lead_text}, _config(thread_id))
    return _run_out(graph, thread_id)


@router.get("/runs/{thread_id}", response_model=RunOut)
def get_run(thread_id: str, request: Request) -> RunOut:
    return _run_out(request.app.state.graph, thread_id)


@router.post("/runs/{thread_id}/decision", response_model=RunOut)
def decide(thread_id: str, decision: Decision, request: Request) -> RunOut:
    graph = request.app.state.graph
    if _run_out(graph, thread_id).status != "awaiting_review":
        raise HTTPException(409, "This run is not waiting for a review.")
    if decision.action == "edit" and decision.tool_calls is None:
        raise HTTPException(422, "An edit needs tool_calls.")
    graph.invoke(Command(resume=decision.model_dump()), _config(thread_id))
    return _run_out(graph, thread_id)

class LeadWebhook(BaseModel):
    event_id: str  # the sender's id for this event; a retry of the same event reuses it
    name: str
    email: str
    company: str | None = None
    message: str


def _verify_signature(secret: str | None, body: bytes, signature: str | None) -> None:
    """The sender signs the raw body with the shared secret (HMAC-SHA256); anyone else can't."""
    if not secret:
        raise HTTPException(503, "Webhooks are disabled: WEBHOOK_SECRET is not set.")
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not signature or not hmac.compare_digest(signature, expected):
        raise HTTPException(401, "Invalid signature.")


@router.post("/webhooks/lead", status_code=202, response_model=RunOut)
async def lead_webhook(
    payload: LeadWebhook,
    request: Request,
    background: BackgroundTasks,
    x_signature: str | None = Header(default=None),
) -> RunOut:
    """A form or CRM pushes a new lead. Answer at once; the agent runs after the response is sent."""
    _verify_signature(request.app.state.webhook_secret, await request.body(), x_signature)
    graph = request.app.state.graph
    thread_id = f"webhook-{payload.event_id}"
    if graph.get_state(_config(thread_id)).values:  # a retry of an event we already accepted
        return _run_out(graph, thread_id)

    company = f", {payload.company}" if payload.company else ""
    lead_text = f"From: {payload.name} <{payload.email}>{company}\n\n{payload.message}"
    background.add_task(graph.invoke, {"lead_text": lead_text}, _config(thread_id))
    return RunOut(thread_id=thread_id, status="running")

def make_checkpointer(path: str) -> SqliteSaver:
    """Paused runs are saved in a SQLite file, so they survive a server restart."""
    # FastAPI runs each request in a worker thread; SqliteSaver locks around its own writes.
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)

def _default_graph():
    engine = make_engine(f"sqlite:///{Path('crm.db').resolve()}")
    with Session(engine) as session:
        seed(session)
    tools = load_tools(engine, os.getenv("TOOLS", "local"))
    return build_graph(get_llm(), tools, checkpointer=make_checkpointer(os.getenv("CHECKPOINT_DB", "checkpoints.db")))


def create_app(graph=None, webhook_secret: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.graph = graph or _default_graph()
        app.state.webhook_secret = webhook_secret or os.getenv("WEBHOOK_SECRET")
        yield

    app = FastAPI(title="Switchboard", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()