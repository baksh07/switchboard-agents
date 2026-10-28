"""Terminal demo: run leads through the agent and approve, reject or edit its CRM writes."""

import argparse
import json
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy.orm import Session

from switchboard.core.llm import get_llm
from switchboard.usecases.lead_qualification.db import make_engine, seed
from switchboard.usecases.lead_qualification.graph import build_graph
from switchboard.usecases.lead_qualification.scoring import SAMPLE_LEADS
from switchboard.usecases.lead_qualification.tools import make_tools


def print_update(update: dict) -> None:
    for node, data in update.items():
        if node == "__interrupt__":
            continue
        print(f"--> {node}")
        for msg in (data or {}).get("messages", []):
            for call in getattr(msg, "tool_calls", []):
                print(f"    calls {call['name']}({call['args']})")
            if msg.type == "tool":
                print(f"    result {msg.content[:150]}")


def edit_plan(plan: list[dict]) -> list[dict]:
    edited = []
    for call in plan:
        while True:
            raw = input(f"  {call['name']} args [Enter = keep, 'drop' = remove, or new JSON]: ").strip()
            if raw == "":
                edited.append(call)
            elif raw == "drop":
                pass
            else:
                try:
                    edited.append({**call, "args": json.loads(raw)})
                except json.JSONDecodeError as exc:
                    print(f"  Not valid JSON ({exc}), try again.")
                    continue
            break
    return edited


def ask_human(plan: list[dict]) -> dict:
    print("\n  PLANNED CRM WRITES (nothing has been written yet):")
    for i, call in enumerate(plan, 1):
        print(f"    {i}. {call['name']} {json.dumps(call['args'])}")
    while True:
        choice = input("  [a]pprove / [r]eject / [e]dit? ").strip().lower()
        if choice in ("a", "approve"):
            return {"action": "approve"}
        if choice in ("r", "reject"):
            return {"action": "reject"}
        if choice in ("e", "edit"):
            return {"action": "edit", "tool_calls": edit_plan(plan)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lead-qualification agent with human approval.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--lead", default=None, help="Lead text. Default: run the sample leads.")
    parser.add_argument("--auto-approve", action="store_true", help="Approve every plan without asking.")
    parser.add_argument("--mermaid", action="store_true", help="Print the graph diagram and exit.")
    args = parser.parse_args()

    engine = make_engine()
    with Session(engine) as session:
        seed(session)
    app = build_graph(get_llm(args.model), make_tools(engine), checkpointer=InMemorySaver())

    if args.mermaid:
        print(app.get_graph().draw_mermaid())
        return

    for lead in [args.lead] if args.lead else SAMPLE_LEADS:
        print("=" * 80, f"\nLEAD: {lead}\n")
        config = {"configurable": {"thread_id": str(uuid4())}}
        run_input = {"lead_text": lead}

        while True:
            for update in app.stream(run_input, config, stream_mode="updates"):
                print_update(update)
            state = app.get_state(config)
            if not state.interrupts:
                break
            plan = state.interrupts[0].value["planned_writes"]
            decision = {"action": "approve"} if args.auto_approve else ask_human(plan)
            run_input = Command(resume=decision)

        print("\nSUMMARY:", state.values["summary"], "\n")


if __name__ == "__main__":
    main()