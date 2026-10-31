"""The CRM tools behave the same whether called in-process or through the MCP server (stdio subprocess)."""

import json

import pytest
from sqlalchemy.orm import Session

from switchboard.usecases.lead_qualification.db import make_engine, seed
from switchboard.usecases.lead_qualification.tools import load_tools


@pytest.fixture(scope="module")
def mcp_tools(tmp_path_factory):
    engine = make_engine(f"sqlite:///{tmp_path_factory.mktemp('mcp') / 'crm.db'}")
    with Session(engine) as s:
        seed(s)
    return {t.name: t for t in load_tools(engine, "mcp")}


def test_mcp_server_exposes_the_same_tools_and_schemas(mcp_tools, tmp_path):
    local = {t.name: t for t in load_tools(make_engine(f"sqlite:///{tmp_path / 'x.db'}"), "local")}
    assert set(mcp_tools) == set(local)
    status = mcp_tools["update_contact"].args_schema["properties"]["status"]
    assert status["anyOf"][0]["enum"] == ["new", "qualified", "disqualified"]


def test_mcp_read_and_write_round_trip(mcp_tools):
    created = json.loads(
        mcp_tools["create_contact"].invoke(
            {"name": "Ada Park", "email": "ada@initech.example", "company_domain": "initech.example"}
        )
    )
    assert created["created"]["company"] == "Initech"
    found = json.loads(mcp_tools["search_contacts"].invoke({"query": "ada@initech.example"}))
    assert found["count"] == 1


def test_mcp_tool_errors_come_back_as_text(mcp_tools):
    duplicate = mcp_tools["create_contact"].invoke({"name": "Priya", "email": "priya@acmerobotics.example"})
    assert "already exists" in json.loads(duplicate)["error"]
    invalid = mcp_tools["update_contact"].invoke({"email": "jonas@acmerobotics.example", "status": "maybe"})
    assert "validation error" in invalid
