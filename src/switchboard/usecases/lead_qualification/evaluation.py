"""How to run and score one eval example for the lead-qualification use case (used by core/evals.py)."""

import tempfile
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy.orm import Session

from switchboard.usecases.lead_qualification.db import make_engine, seed
from switchboard.usecases.lead_qualification.graph import build_graph
from switchboard.usecases.lead_qualification.tools import load_tools


def classify_writes(writes: list[dict]) -> dict:
    """Turn a list of planned tool calls into the action/task shape used in the dataset."""
    names = [w["name"] for w in writes]
    statuses = [w["args"].get("status") for w in writes if w["name"] == "update_contact"]
    if "create_contact" in names:
        action = "create"
    elif "disqualified" in statuses:
        action = "disqualify"
    elif "update_contact" in names:
        action = "update"
    else:
        action = "none"
    return {"action": action, "task": "create_task" in names}


def run_example(llm: BaseChatModel, example: dict, config: dict, tools: str = "local") -> dict:
    """Run one lead on a fresh seeded CRM, auto-approving every plan; return what the agent planned."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = make_engine(f"sqlite:///{Path(tmp) / 'crm.db'}")
        with Session(engine) as session:
            seed(session)
        app = build_graph(llm, load_tools(engine, tools), checkpointer=InMemorySaver())

        planned = []
        app.invoke(example["input"], config)
        while interrupts := app.get_state(config).interrupts:
            planned += interrupts[0].value["planned_writes"]
            app.invoke(Command(resume={"action": "approve"}), config)

        summary = app.get_state(config).values.get("summary", "")
        engine.dispose()  # release the SQLite file so Windows can delete the temp dir

    writes = [{"name": w["name"], "args": w["args"]} for w in planned]
    return {"writes": writes, **classify_writes(writes), "summary": summary}


def score(example: dict, outcome: dict) -> dict[str, bool]:
    expected = example["expected"]
    action_ok = outcome["action"] == expected["action"]
    return {
        "action": action_ok,
        "full_plan": action_ok and outcome["task"] == expected["task"],
    }