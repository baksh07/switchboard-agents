import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from sqlalchemy.orm import Session

from switchboard.usecases.lead_qualification.db import make_engine, seed
from switchboard.usecases.lead_qualification.graph import build_graph, route_after_agent
from switchboard.usecases.lead_qualification.state import LeadScore
from switchboard.usecases.lead_qualification.tools import make_tools

ADA = "Hello, Ada Park here, Head of Operations at Initech (ada@initech.example)."


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


@pytest.fixture
def engine(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with Session(engine) as s:
        seed(s)
    return engine


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


def test_good_new_lead_is_looked_up_then_created_with_a_task(engine):
    llm = FakeLLM(make_score(), [
        AIMessage("", tool_calls=[
            call("lookup_company", {"domain": "initech.example"}, "1"),
            call("search_contacts", {"query": "ada@initech.example"}, "2"),
        ]),
        AIMessage("", tool_calls=[
            call("create_contact", {"name": "Ada Park", "email": "ada@initech.example",
                                    "company_domain": "initech.example", "company_name": "Initech"}, "3"),
            call("create_task", {"contact_email": "ada@initech.example", "description": "Intro call"}, "4"),
        ]),
    ])
    tools = make_tools(engine)
    result = build_graph(llm, tools).invoke({"lead_text": ADA})

    assert result["summary"] == "Test summary."
    found = {t.name: t for t in tools}["search_contacts"].invoke({"query": "ada@initech.example"})
    assert found["count"] == 1
    tool_results = [m for m in result["messages"] if m.type == "tool"]
    assert [m.name for m in tool_results] == ["lookup_company", "search_contacts", "create_contact", "create_task"]
    assert "error" not in tool_results[-1].content


def test_rejected_new_lead_writes_nothing(engine):
    llm = FakeLLM(make_score("reject"), [AIMessage("Not a fit: student, no business need.")])
    tools = make_tools(engine)
    result = build_graph(llm, tools).invoke({"lead_text": "hey im a student"})

    assert not any(m.type == "tool" for m in result["messages"])
    assert {t.name: t for t in tools}["search_contacts"].invoke({"query": "ada@initech.example"})["count"] == 0