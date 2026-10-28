import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy.orm import Session

from switchboard.usecases.lead_qualification.db import make_engine, seed
from switchboard.usecases.lead_qualification.graph import build_graph, route_after_agent
from switchboard.usecases.lead_qualification.state import LeadScore
from switchboard.usecases.lead_qualification.tools import make_tools

ADA = "Hello, Ada Park here, Head of Operations at Initech (ada@initech.example)."
CONFIG = {"configurable": {"thread_id": "test"}}


def make_score(action="pursue"):
    return LeadScore(
        name="Ada Park", email="ada@initech.example", title="Head of Operations",
        company_name="Initech", company_domain="initech.example",
        fit_score=70, intent="medium", reasons=["test"], action=action,
    )


def call(name, args, id):
    return {"name": name, "args": args, "id": id}


class FakeLLM:
    """Scripted stand-in for a chat model: one score, a queue of agent replies, a fixed summary."""

    def __init__(self, score, agent_replies):
        self.score = score
        self.agent_replies = list(agent_replies)

    def with_structured_output(self, schema, include_raw=False):
        return RunnableLambda(lambda _: {"raw": None, "parsed": self.score, "parsing_error": None})

    def bind_tools(self, tools):
        return RunnableLambda(lambda _: self.agent_replies.pop(0))

    def invoke(self, messages):
        return AIMessage("Test summary.")


def ada_agent_replies():
    """Agent looks Ada up, then plans: create contact + task."""
    return [
        AIMessage("", tool_calls=[
            call("lookup_company", {"domain": "initech.example"}, "1"),
            call("search_contacts", {"query": "ada@initech.example"}, "2"),
        ]),
        AIMessage("", tool_calls=[
            call("create_contact", {"name": "Ada Park", "email": "ada@initech.example",
                                    "company_domain": "initech.example"}, "3"),
            call("create_task", {"contact_email": "ada@initech.example", "description": "Intro call"}, "4"),
        ]),
    ]


@pytest.fixture
def engine(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with Session(engine) as s:
        seed(s)
    return engine


@pytest.fixture
def ada_run(engine):
    """Run Ada until the graph pauses for review. Returns (app, tools by name)."""
    tools = make_tools(engine)
    app = build_graph(FakeLLM(make_score(), ada_agent_replies()), tools, checkpointer=InMemorySaver())
    app.invoke({"lead_text": ADA}, CONFIG)
    return app, {t.name: t for t in tools}


def ada_in_crm(tools) -> bool:
    return tools["search_contacts"].invoke({"query": "ada@initech.example"})["count"] == 1


def test_route_with_no_tool_calls_goes_to_summarize():
    assert route_after_agent({"messages": [AIMessage("Not a fit.")]}) == "summarize"


def test_route_with_only_reads_goes_to_tools():
    msg = AIMessage("", tool_calls=[call("search_contacts", {"query": "x"}, "1")])
    assert route_after_agent({"messages": [msg]}) == "tools"


def test_route_with_any_write_goes_to_human_review():
    msg = AIMessage("", tool_calls=[
        call("search_contacts", {"query": "x"}, "1"),
        call("create_contact", {"name": "A", "email": "a@b.example"}, "2"),
    ])
    assert route_after_agent({"messages": [msg]}) == "human_review"


def test_graph_pauses_before_writing(ada_run):
    app, tools = ada_run
    state = app.get_state(CONFIG)
    planned = [c["name"] for c in state.interrupts[0].value["planned_writes"]]
    assert planned == ["create_contact", "create_task"]
    assert state.next == ("human_review",)
    assert not ada_in_crm(tools)


def test_approve_runs_writes_in_order(ada_run):
    app, tools = ada_run
    result = app.invoke(Command(resume={"action": "approve"}), CONFIG)

    assert result["summary"] == "Test summary."
    assert ada_in_crm(tools)
    writes = [m for m in result["messages"] if m.type == "tool"][-2:]
    assert [m.name for m in writes] == ["create_contact", "create_task"]
    assert all("error" not in m.content for m in writes)


def test_reject_writes_nothing(ada_run):
    app, tools = ada_run
    result = app.invoke(Command(resume={"action": "reject"}), CONFIG)

    assert not ada_in_crm(tools)
    assert "Rejected by human reviewer" in result["messages"][-1].content


def test_edit_changes_args_and_drops_calls(ada_run):
    app, tools = ada_run
    plan = app.get_state(CONFIG).interrupts[0].value["planned_writes"]
    edited_contact = {**plan[0], "args": {**plan[0]["args"], "title": "COO"}}
    app.invoke(Command(resume={"action": "edit", "tool_calls": [edited_contact]}), CONFIG)

    contact = tools["search_contacts"].invoke({"query": "ada@initech.example"})["contacts"][0]
    assert contact["title"] == "COO"
    final = app.get_state(CONFIG).values["messages"]
    assert [m.name for m in final if m.type == "tool"][-1] == "create_contact"


def test_rejected_new_lead_writes_nothing(engine):
    llm = FakeLLM(make_score("reject"), [AIMessage("Not a fit: student, no business need.")])
    tools = make_tools(engine)
    result = build_graph(llm, tools).invoke({"lead_text": "hey im a student"})

    assert not any(m.type == "tool" for m in result["messages"])
    assert result["summary"] == "Test summary."