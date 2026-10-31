import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from switchboard.usecases.lead_qualification.db import make_engine, seed
from switchboard.usecases.lead_qualification.tools import make_tools


@pytest.fixture
def tools(tmp_path):
    """
    Fresh seeded SQLITE file per test so tests never affect each other or crm.db
    """
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with Session(engine) as s:
        seed(s)

    return {t.name: t for t in make_tools(engine)}


def test_search_by_email_finds_existing_contact(tools):
    result = tools["search_contacts"].invoke({"query": "PRIYA@acmerobotics.example"})
    assert result["count"] == 1
    assert result["contacts"][0]["name"] == "Priya Sharma"


def test_search_by_domain_finds_all_company_contacts(tools):
    result = tools["search_contacts"].invoke({"query": "acmerobotics.example"})
    assert result["count"] == 2


def test_search_unknown_returns_empty(tools):
    assert tools["search_contacts"].invoke({"query": "nobody@nowhere.example"})["count"] == 0


def test_lookup_company_from_crm_includes_open_deals(tools):
    result = tools["lookup_company"].invoke({"domain": "acmerobotics.example"})
    assert result["source"] == "crm"
    assert result["open_deals"] == [{"name": "Acme - 50 seat pilot", "amount": 25_000}]


def test_lookup_company_falls_back_to_enrichment(tools):
    result = tools["lookup_company"].invoke({"domain": "initech.example"})
    assert result["source"] == "enrichment"
    assert result["employee_count"] == 300


def test_lookup_unknown_company(tools):
    assert tools["lookup_company"].invoke({"domain": "unknown.example"})["found"] is False


def test_create_contact_with_new_company(tools):
    result = tools["create_contact"].invoke(
        {
            "name": "Ada Park",
            "email": "ada@initech.example",
            "company_domain": "initech.example",
            "company_name": "Initech",
        }
    )
    assert result["created"]["company"] == "Initech"
    assert tools["search_contacts"].invoke({"query": "ada@initech.example"})["count"] == 1


def test_create_contact_rejects_duplicate_email(tools):
    result = tools["create_contact"].invoke({"name": "Priya", "email": "priya@acmerobotics.example"})
    assert "already exists" in result["error"]


def test_update_contact_changes_status(tools):
    result = tools["update_contact"].invoke({"email": "jonas@acmerobotics.example", "status": "qualified"})
    assert result["updated"]["status"] == "qualified"


def test_update_unknown_contact_returns_error(tools):
    result = tools["update_contact"].invoke({"email": "ghost@x.example", "status": "qualified"})
    assert "error" in result


def test_update_rejects_invalid_status(tools):
    with pytest.raises(ValidationError):
        tools["update_contact"].invoke({"email": "jonas@acmerobotics.example", "status": "maybe"})


def test_create_task_for_existing_contact(tools):
    result = tools["create_task"].invoke(
        {"contact_email": "mei.lin@globex.example", "description": "Book demo", "due_in_days": 2}
    )
    assert result["created_task"]["contact"] == "mei.lin@globex.example"


def test_create_task_for_unknown_contact_returns_error(tools):
    result = tools["create_task"].invoke({"contact_email": "ghost@x.example", "description": "x"})
    assert "error" in result
