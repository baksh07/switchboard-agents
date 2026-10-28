# Switchboard

**Model-agnostic LangGraph agent stack: swap LLMs in one line, tools over MCP, human approval, evals and tracing built in.**

A switchboard routes connections. Here it routes any model to any tool, with a human operator approving every action that changes data.

The demo use case is **inbound lead qualification against a mock CRM**: an agent reads a sales lead, checks the facts with CRM and enrichment tools, proposes CRM changes, and waits for a human to approve, reject or edit them before anything is written.

> **Status:** work in progress. The agent, tools, provider switching and human-in-the-loop work and are tested.
> Tracing, evals, MCP, the REST API, Docker and CI are next (see [Roadmap](#roadmap)).

---

## What it does

Given a lead like:

> *Hello, Ada Park here, Head of Operations at Initech (ada@initech.example). Just exploring options for next year, no rush.*

the agent:

1. **Extracts and scores** the lead into a typed `LeadScore` (name, email, company, fit score 0-100, intent, reasons), using structured output.
2. **Checks the facts** with read-only tools: `lookup_company` (CRM data or mock enrichment) and `search_contacts`.
3. **Plans CRM writes** following explicit rules: create a new contact plus a follow-up task, update an existing contact, or change nothing.
4. **Pauses for human review.** Nothing is written until a person approves, rejects or edits the plan.
5. **Executes** the approved writes in order and **summarizes** what happened for the sales team.

A real run, from the terminal demo (output shortened):

```text
--> extract_and_score
--> agent
    calls lookup_company({'domain': 'initech.example'})
    calls search_contacts({'query': 'ada@initech.example'})
--> tools
    result {"found": true, "source": "enrichment", "name": "Initech", "industry": "Software", "employee_count": 300}
    result {"count": 0, "contacts": []}
--> agent
    calls create_contact({'name': 'Ada Park', 'email': 'ada@initech.example', 'title': 'Head of Operations', ...})
    calls create_task({'contact_email': 'ada@initech.example', 'description': 'Follow up with Ada Park ...'})

  PLANNED CRM WRITES (nothing has been written yet):
    1. create_contact {"name": "Ada Park", "email": "ada@initech.example", ...}
    2. create_task {"contact_email": "ada@initech.example", ...}
  [a]pprove / [r]eject / [e]dit? a
--> human_review
--> execute
    result {"created": {"id": 5, "name": "Ada Park", ..., "company": "Initech"}}
    result {"created_task": {"id": 2, "contact": "ada@initech.example", ...}}
--> summarize
```

---

## Architecture

```mermaid
graph TD
    START([start]) --> extract_and_score
    extract_and_score --> agent
    agent -. only reads .-> tools
    tools --> agent
    agent -. any write .-> human_review
    agent -. no tool calls .-> summarize
    human_review -->|"interrupt(): approve / reject / edit"| execute
    execute --> summarize
    summarize --> END([end])
```

| Node | What it does |
|---|---|
| `extract_and_score` | Structured output into `LeadScore`, with one retry on invalid output. Treated as a *first impression*, not a decision. |
| `agent` | LLM with all CRM tools bound. Loops with `tools` until it has the facts it needs. |
| `tools` | LangGraph `ToolNode`. Runs **read-only** tools immediately. |
| `human_review` | `interrupt()` shows the planned writes; the run resumes with `approve`, `reject` or `edit`. |
| `execute` | Runs the approved writes **one by one, in order**, and reports errors back as messages. |
| `summarize` | 2-4 sentence summary for the sales team, given the confirmed CRM changes as facts. |

State is a typed `LeadState` (`TypedDict`). `messages` uses the `add_messages` reducer, and a checkpointer saves the state after every node, which is what makes pausing and resuming possible.

### Generic core, pluggable use cases

```
src/switchboard/
├─ core/                         reusable, no provider- or use-case-specific code
│  ├─ llm.py                     get_llm("provider:model")
│  └─ structured.py              structured output with retry and clean failure
└─ usecases/lead_qualification/  the demo use case
   ├─ db.py                      SQLAlchemy models + seed data for the mock CRM
   ├─ tools.py                   5 CRM tools, built by make_tools(engine)
   ├─ state.py                   LeadScore (Pydantic) and LeadState (graph state)
   ├─ prompts.py                 scoring, agent and summary prompts
   ├─ scoring.py                 lead -> LeadScore
   ├─ graph.py                   the StateGraph
   └─ cli.py                     terminal demo with human approval
```

The rule: **`core/` never imports from a use case.** A second use case adds its own folder (state, tools, prompts, graph) and reuses the core.

---

## Quickstart

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/baksh07/switchboard-agents.git
cd switchboard-agents
uv sync
cp .env.example .env        # then add OPENAI_API_KEY and/or GROQ_API_KEY
```

Run the tests (no API keys needed, the LLM is faked):

```bash
uv run pytest
```

Run the agent on three sample leads, with you as the approver:

```bash
uv run python -m switchboard.usecases.lead_qualification.cli --model openai:gpt-4o-mini
```

Other options:

```bash
# your own lead
uv run python -m switchboard.usecases.lead_qualification.cli --lead "Hi, I'm ..."

# approve every plan automatically
uv run python -m switchboard.usecases.lead_qualification.cli --auto-approve

# print the graph as a Mermaid diagram
uv run python -m switchboard.usecases.lead_qualification.cli --mermaid
```

The mock CRM lives in `crm.db` (SQLite) and is created and seeded on first run. Delete the file to reset it.

### Reviewing a plan

When the agent plans CRM writes, the run pauses:

| Input | Effect |
|---|---|
| `a` | approve: run the plan as proposed |
| `r` | reject: write nothing; the summary says a reviewer rejected it |
| `e` | edit: for each planned call, press Enter to keep it, type `drop` to remove it, or paste new JSON arguments |

---

## Switching models

Every model is chosen with one `provider:model` string, from `--model` or `MODEL` in `.env`:

```python
from switchboard.core.llm import get_llm

llm = get_llm("openai:gpt-4o-mini")
llm = get_llm("groq:openai/gpt-oss-120b")
```

`get_llm` wraps LangChain's `init_chat_model`, so the graph only ever sees a `BaseChatModel` and no provider-specific code. It defaults to `temperature=0` and `max_retries=6` (agent runs make several calls and easily hit free-tier rate limits).

Tested so far: OpenAI `gpt-4o-mini` and Groq `openai/gpt-oss-120b`. Adding a provider means installing its LangChain package (e.g. `uv add langchain-anthropic`); no code changes.

---

## Design decisions

Most of these came from bugs found while running the agent against real models.

**Reads run freely, writes need approval, and the graph enforces it.**
The LLM can request any tool, but `route_after_agent` (plain Python, unit-tested) sends any turn containing a write tool to `human_review`. Safety does not depend on the model following the prompt.

**Writes run in order in their own node.**
`ToolNode` runs tool calls in parallel. In an early version, `create_task` ran before `create_contact` had finished, failed with "no such contact", and the agent's recovery attempts created a duplicate task. `execute` now runs writes sequentially.

**The first-pass score is a hint; the decision comes after the tools.**
Scoring the raw email alone made one model claim *"company size fits"* when the email never mentioned size. Now the score is labelled unverified, and the agent decides only after `lookup_company`.

**If code can check a fact, code tells the model.**
The summary model once reported a contact as created when nothing had been written, and once invented a reason for a rejected plan. `summarize` now passes the confirmed writes (or "a reviewer rejected the plan") as explicit facts.

**Tools return errors as data.**
`{"error": "Contact already exists. Use update_contact."}` lets the agent correct itself instead of crashing the run. Inputs are validated by type hints (e.g. `status: Literal["new", "qualified", "disqualified"]`), emails are normalized, and the database enforces unique emails as a final guard.

**Tools are built by a factory: `make_tools(engine)`.**
No global database: the CLI passes `crm.db`, tests pass a temporary SQLite file. The same idea (passing dependencies in) is used for the LLM in `build_graph(llm, tools)`, which is why the whole graph can be tested with a scripted fake model.

**Editing a plan replaces the agent's message.**
An edited plan is a new `AIMessage` with the *same id*, so `add_messages` replaces the original and `execute` needs no special case for edits.

**Structured output retries once, showing the model its error, then fails cleanly** with a dedicated `StructuredOutputError`.

---

## Testing

```bash
uv run pytest
```

- **Tools:** every tool's success and error paths against a temporary SQLite database.
- **LLM factory:** provider selection, defaults and input validation (no network calls).
- **Structured output:** first-try success, retry with the error shown to the model, clean failure.
- **Graph:** routing rules, plus full runs with a scripted fake LLM: the graph pauses *before* any write, and approve, reject and edit each produce the right CRM state.

No API keys are needed to run the tests.

---

## Roadmap

- [x] LangGraph agent with typed state, conditional routing and a checkpointer
- [x] Model-agnostic LLM layer (OpenAI and Groq tested)
- [x] CRM tools with unit tests (SQLAlchemy, SQLite)
- [x] Structured output with retry
- [x] Human-in-the-loop approval with `interrupt()` (approve / reject / edit)
- [ ] LangSmith tracing
- [ ] Eval harness: accuracy, latency, tokens and cost per model, with a results table
- [ ] MCP server for the CRM tools (`--tools mcp|local`)
- [ ] FastAPI: start runs, approve via HTTP, webhook receiver
- [ ] Docker Compose
- [ ] GitHub Actions CI (lint + tests)
- [ ] Optional: RAG knowledge-base tool, Amazon Bedrock provider

## Limitations

- The CRM and the company enrichment are mocks with fake `.example` data.
- The checkpointer is in-memory, so a paused run is lost when the process exits. A persistent checkpointer comes with the API.
- The agent gets one write plan per lead. If a model proposes only part of its plan (for example, an update without the follow-up task), the rest is not executed. The upcoming evals will measure how often this happens per model.
- No evaluation numbers yet. Model comparisons will be added once they are measured.

---

Built with help from [Claude Code](https://claude.com/claude-code).
