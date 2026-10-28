import pytest
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


from switchboard.core.llm import get_llm

@pytest.fixture(autouse=True)
def fake_keys(monkeypatch):
    """Clients only need a key to exist when built; no request is sent in these tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


def test_openai_spec_builds_openai_client():
    llm = get_llm("openai:gpt-4o-mini")
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o-mini"

def test_groq_spec_builds_groq_client():
    llm = get_llm("groq:openai/gpt-oss-120b")
    assert isinstance(llm, ChatGroq)
    assert llm.model_name == "openai/gpt-oss-120b"


def test_temperature_defaults_to_zero_but_can_be_overridden():
    assert get_llm("openai:gpt-4o-mini").temperature == 0
    assert get_llm("openai:gpt-4o-mini", temperature=0.7).temperature == 0.7


def test_falls_back_to_model_env_var(monkeypatch):
    monkeypatch.setenv("MODEL", "groq:openai/gpt-oss-120b")
    assert isinstance(get_llm(), ChatGroq)


@pytest.mark.parametrize("bad", ["gpt-4o-mini", "openai:", ":gpt-4o-mini"])
def test_rejects_spec_without_provider(bad):
    with pytest.raises(ValueError, match="provider:model"):
        get_llm(bad)