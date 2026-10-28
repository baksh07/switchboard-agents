import argparse
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from sqlalchemy.orm import Session

from switchboard.core.llm import get_llm
from switchboard.usecases.lead_qualification.db import make_engine, seed
from switchboard.usecases.lead_qualification.prompts import AGENT_SYSTEM_PROMPT
from switchboard.usecases.lead_qualification.scoring import SAMPLE_LEADS, score_lead
from switchboard.usecases.lead_qualification.state import LeadState
from switchboard.usecases.lead_qualification.tools import make_tools


def route_after_agent(state: LeadState) -> Literal["tools", "__end__"]:
    """
    if the agent asked for tools run them; otherwise it is done.
    """
    if state["messages"][-1].tool_calls:
        return "tools"
    return END


def build_graph(llm: BaseChatModel, tools: list[BaseTool]):
    agent_llm = llm.bind_tools(tools)

    def extract_and_score(state: LeadState) -> dict:
        score = score_lead(state["lead_text"], llm)
        brief = (
            f"Inbound lead:\n{state['lead_text']}\n\n"
            f"First-pass score (unverified):\n{score.model_dump_json(indent=2)}"
        )
        return {"score": score.model_dump(), "messages": [HumanMessage(brief)]}

    def agent(state: LeadState) -> dict:
        reply = agent_llm.invoke([SystemMessage(AGENT_SYSTEM_PROMPT),
                                  *state["messages"]])
        return {"messages" : [reply]}

    graph = StateGraph(LeadState)
    graph.add_node("extract_and_score", extract_and_score)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "extract_and_score")
    graph.add_edge("extract_and_score", "agent")
    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_edge("tools", "agent")

    return graph.compile()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the lead-qualification agent")
    parser.add_argument("--model", default=None)
    parser.add_argument("--lead", default=None, help="Lead Text Default: run the sample leads.")
    args = parser.parse_args()


    engine = make_engine()
    with Session(engine) as session:
        seed(session=session)
    app = build_graph(get_llm(args.model), make_tools(engine))

    for lead in [args.lead] if args.lead else SAMPLE_LEADS:
        print("=" * 80+f"\nLEAD: {lead}\n")
        for update in app.stream({"lead_text": lead}, stream_mode="updates"):
            for node, data in update.items():
                print(f"--> {node}")
                for msg in data.get("messages", []):
                    for call in getattr(msg, "tool_calls", []):
                        print(f"    calls {call['name']}({call['args']})")
                    if msg.type == 'tool':
                        print(f"    result {msg.content[:120]}")
                    if msg.type == "ai" and msg.content:
                        print(f"    says {msg.content}")