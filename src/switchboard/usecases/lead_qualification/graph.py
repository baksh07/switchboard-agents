"""The lead-qualification agent as a LangGraph StateGraph."""

import json
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from switchboard.usecases.lead_qualification.prompts import AGENT_SYSTEM_PROMPT, SUMMARY_SYSTEM_PROMPT
from switchboard.usecases.lead_qualification.scoring import score_lead
from switchboard.usecases.lead_qualification.state import LeadState
from switchboard.usecases.lead_qualification.tools import WRITE_TOOLS


def route_after_agent(state: LeadState) -> Literal["tools", "human_review", "summarize"]:
    """Read-only tool calls run straight away; any CRM write goes to review; no calls means done."""
    calls = state["messages"][-1].tool_calls
    if not calls:
        return "summarize"
    if any(call["name"] in WRITE_TOOLS for call in calls):
        return "human_review"
    return "tools"


MAX_WRITE_ROUNDS = 3


def route_after_execute(state: LeadState) -> Literal["agent", "summarize"]:
    """After writes: let the agent add missing follow-ups, unless the reviewer said no or we hit the cap."""
    if not state.get("approved") or state.get("write_rounds", 0) >= MAX_WRITE_ROUNDS:
        return "summarize"
    return "agent"


def build_graph(llm: BaseChatModel, tools: list[BaseTool], checkpointer: BaseCheckpointSaver | None = None):
    agent_llm = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    def extract_and_score(state: LeadState) -> dict:
        score = score_lead(state["lead_text"], llm)
        brief = (
            f"Inbound lead:\n{state['lead_text']}\n\n"
            f"First-pass score (unverified):\n{score.model_dump_json(indent=2)}"
        )
        return {"score": score.model_dump(), "messages": [HumanMessage(brief)]}

    def agent(state: LeadState) -> dict:
        reply = agent_llm.invoke([SystemMessage(AGENT_SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply]}

    def human_review(state: LeadState) -> dict:
        """Pause and show the planned writes; resume with approve, reject, or edit (new tool_calls)."""
        plan = state["messages"][-1]
        decision = interrupt({"planned_writes": plan.tool_calls})

        if decision["action"] == "approve":
            return {"approved": True}
        if decision["action"] == "edit":
            # Same id, so add_messages replaces the agent's message instead of appending a new one.
            edited = AIMessage(content=plan.content, tool_calls=decision["tool_calls"], id=plan.id)
            return {"approved": True, "messages": [edited]}
        return {"approved": False}

    def execute(state: LeadState) -> dict:
        """Run the approved writes one by one, in order (a task needs its contact to exist first)."""
        results = []
        for call in state["messages"][-1].tool_calls:
            if not state.get("approved"):
                output = {"error": "Rejected by human reviewer. Nothing was written."}
            else:
                try:
                    output = tools_by_name[call["name"]].invoke(call["args"])
                except Exception as exc:  # bad arguments etc.: report it instead of crashing the run
                    output = {"error": f"{type(exc).__name__}: {exc}"}
            # results.append(ToolMessage(json.dumps(output), tool_call_id=call["id"], name=call["name"]))
            content = output if isinstance(output, str) else json.dumps(output) # MCP tools already return text
            results.append(ToolMessage(content, tool_call_id=call["id"], name=call["name"]))
        return {"messages": results, "write_rounds": state.get("write_rounds", 0) + 1}

    def summarize(state: LeadState) -> dict:
        # Tell the model exactly which writes succeeded, so it can't claim changes that never happened.
        writes = [
            f"- {m.name}: {m.content}"
            for m in state["messages"]
            if m.type == "tool" and m.name in WRITE_TOOLS and '"error"' not in m.content
        ]
        facts = "Confirmed CRM changes:\n" + "\n".join(writes) if writes else "No CRM changes were made."
        if state.get("approved") is False:
            facts += "\nA human reviewer rejected the last planned changes, so those were not written."
        reply = llm.invoke([SystemMessage(SUMMARY_SYSTEM_PROMPT), *state["messages"], HumanMessage(facts)])
        return {"summary": reply.content}

    graph = StateGraph(LeadState)
    graph.add_node("extract_and_score", extract_and_score)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("human_review", human_review)
    graph.add_node("execute", execute)
    graph.add_node("summarize", summarize)

    graph.add_edge(START, "extract_and_score")
    graph.add_edge("extract_and_score", "agent")
    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_edge("tools", "agent")
    graph.add_edge("human_review", "execute")
    graph.add_conditional_edges("execute", route_after_execute)
    graph.add_edge("summarize", END)

    return graph.compile(checkpointer=checkpointer)