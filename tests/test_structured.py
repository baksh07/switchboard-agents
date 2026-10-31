import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from switchboard.core.structured import StructuredOutputError, invoke_structured


class Answer(BaseModel):
    value: int


class FakeStructuredLLM:
    """
    Stands in for a chat model: returns pre-scripted results and records every call.
    """

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def with_structured_output(self, schema, include_raw=False):
        def respond(messages):
            self.calls.append(messages)
            return self.results.pop(0)

        return RunnableLambda(respond)


def ok(value):
    return {"raw": None, "parsed": Answer(value=value), "parsing_error": None}


def bad(error="value: not an int"):
    return {"raw": None, "parsed": None, "parsing_error": ValueError(error)}


def test_returns_parsed_output_on_first_try():
    llm = FakeStructuredLLM([ok(42)])
    assert invoke_structured(llm, Answer, [HumanMessage("hi")]) == Answer(value=42)
    assert len(llm.calls) == 1


def test_retries_once_and_shows_the_error_to_the_model():
    llm = FakeStructuredLLM([bad("value: not an int"), ok(7)])
    assert invoke_structured(llm, Answer, [HumanMessage("hi")]).value == 7
    retry_messages = llm.calls[1]
    assert "value: not an int" in retry_messages[-1].content


def test_raises_clean_error_after_retries_run_out():
    llm = FakeStructuredLLM([bad(), bad()])
    with pytest.raises(StructuredOutputError, match="after 2 attempts"):
        invoke_structured(llm, Answer, [HumanMessage("hi")])
