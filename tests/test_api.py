"""The REST API, with the scripted fake LLM: no API keys, no network."""
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy.orm import Session

from switchboard.usecases.lead_qualification.api import create_app, make_checkpointer
from switchboard.usecases.lead_qualification.db import make_engine, seed
from switchboard.usecases.lead_qualification.graph import build_graph
from switchboard.usecases.lead_qualification.tools import make_tools
from test_graph import FakeLLM, ada_agent_replies, make_score

DONE = AIMessage("Created Ada and a follow-up task.")


@pytest.fixture
def tools(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'crm.db'}")
    with Session(engine) as s:
        seed(s)
    yield {t.name: t for t in make_tools(engine)}
    engine.dispose()


def make_client(tools, replies, checkpoint_path):
    """A fresh app, as if the server had just started. Same checkpoint file = same saved runs."""
    graph = build_graph(FakeLLM(make_score(), replies), list(tools.values()), make_checkpointer(str(checkpoint_path)))
    return TestClient(create_app(graph))


@pytest.fixture
def client(tools, tmp_path):
    with make_client(tools, ada_agent_replies(), tmp_path / "checkpoints.db") as c:
        yield c


def test_run_pauses_before_writing(client, tools):
    run = client.post("/runs", json={"lead_text": "Ada"}).json()
    assert run["status"] == "awaiting_review"
    assert [w["name"] for w in run["planned_writes"]] == ["create_contact", "create_task"]
    assert tools["search_contacts"].invoke({"query": "ada@initech.example"})["count"] == 0
    assert client.get(f"/runs/{run['thread_id']}").json()["status"] == "awaiting_review"


def test_approve_completes_the_run_and_writes(client, tools):
    thread_id = client.post("/runs", json={"lead_text": "Ada"}).json()["thread_id"]
    done = client.post(f"/runs/{thread_id}/decision", json={"action": "approve"}).json()
    assert done["status"] == "completed"
    assert done["summary"] == "Test summary."
    assert tools["search_contacts"].invoke({"query": "ada@initech.example"})["count"] == 1


def test_reject_writes_nothing(client, tools):
    thread_id = client.post("/runs", json={"lead_text": "Ada"}).json()["thread_id"]
    done = client.post(f"/runs/{thread_id}/decision", json={"action": "reject"}).json()
    assert done["status"] == "completed"
    assert tools["search_contacts"].invoke({"query": "ada@initech.example"})["count"] == 0


def test_bad_requests(client):
    thread_id = client.post("/runs", json={"lead_text": "Ada"}).json()["thread_id"]
    assert client.get("/runs/nope").status_code == 404
    assert client.post("/runs/nope/decision", json={"action": "approve"}).status_code == 404
    assert client.post(f"/runs/{thread_id}/decision", json={"action": "maybe"}).status_code == 422
    assert client.post(f"/runs/{thread_id}/decision", json={"action": "edit"}).status_code == 422
    assert client.post(f"/runs/{thread_id}/decision", json={"action": "approve"}).status_code == 200
    assert client.post(f"/runs/{thread_id}/decision", json={"action": "approve"}).status_code == 409


def test_paused_run_survives_a_restart(tools, tmp_path):
    checkpoints = tmp_path / "checkpoints.db"
    with make_client(tools, ada_agent_replies(), checkpoints) as before:
        thread_id = before.post("/runs", json={"lead_text": "Ada"}).json()["thread_id"]

    with make_client(tools, [DONE], checkpoints) as after:  # new app, new graph, nothing in memory
        assert after.get(f"/runs/{thread_id}").json()["status"] == "awaiting_review"
        done = after.post(f"/runs/{thread_id}/decision", json={"action": "approve"}).json()
    assert done["status"] == "completed"
    assert tools["search_contacts"].invoke({"query": "ada@initech.example"})["count"] == 1


SECRET = "test-secret"
WEBHOOK = {"event_id": "evt_1", "name": "Ada Park", "email": "ada@initech.example",
           "company": "Initech", "message": "We want to automate our lead routing."}


def post_webhook(client, payload=WEBHOOK, secret=SECRET):
    body = json.dumps(payload).encode()
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return client.post("/webhooks/lead", content=body,
                       headers={"Content-Type": "application/json", "X-Signature": signature})


@pytest.fixture
def webhook_client(tools, tmp_path):
    graph = build_graph(FakeLLM(make_score(), ada_agent_replies()), list(tools.values()),
                        make_checkpointer(str(tmp_path / "checkpoints.db")))
    with TestClient(create_app(graph, webhook_secret=SECRET)) as c:
        yield c


def test_webhook_accepts_then_runs_in_the_background(webhook_client):
    accepted = post_webhook(webhook_client)
    assert accepted.status_code == 202
    assert accepted.json() == {**accepted.json(), "thread_id": "webhook-evt_1", "status": "running"}
    # TestClient runs background tasks before returning, so the run has reached the review by now.
    run = webhook_client.get("/runs/webhook-evt_1").json()
    assert run["status"] == "awaiting_review"
    assert [w["name"] for w in run["planned_writes"]] == ["create_contact", "create_task"]


def test_webhook_retry_does_not_start_a_second_run(webhook_client):
    post_webhook(webhook_client)
    retry = post_webhook(webhook_client)  # a second run would need agent replies the fake LLM doesn't have
    assert retry.status_code == 202
    assert retry.json()["status"] == "awaiting_review"


def test_webhook_rejects_a_bad_signature(webhook_client):
    assert post_webhook(webhook_client, secret="wrong").status_code == 401
    assert webhook_client.post("/webhooks/lead", json=WEBHOOK).status_code == 401  # no signature at all
    assert webhook_client.get("/runs/webhook-evt_1").status_code == 404