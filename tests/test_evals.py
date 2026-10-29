from types import SimpleNamespace

import pytest

from switchboard.core.evals import run_eval, summarize
from switchboard.usecases.lead_qualification.evaluation import classify_writes


def w(name, **args):
    return {"name": name, "args": args}


@pytest.mark.parametrize(
    "writes, expected",
    [
        ([], {"action": "none", "task": False}),
        ([w("create_contact", email="a@x.example"), w("create_task", contact_email="a@x.example")],
         {"action": "create", "task": True}),
        ([w("update_contact", email="a@x.example", status="qualified")], {"action": "update", "task": False}),
        ([w("update_contact", email="a@x.example", status="disqualified")], {"action": "disqualify", "task": False}),
    ],
)
def test_classify_writes(writes, expected):
    assert classify_writes(writes) == expected


@pytest.fixture(autouse=True)
def fake_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


def fake_usecase(results):
    """A use case whose run_example returns scripted outcomes (or raises) by example id."""

    def run_example(llm, example, config):
        result = results[example["id"]]
        if isinstance(result, Exception):
            raise result
        return result

    def score(example, outcome):
        return {"correct": outcome["answer"] == example["expected"]}

    return SimpleNamespace(run_example=run_example, score=score)


def test_run_eval_scores_each_example_and_records_errors():
    examples = [{"id": "a", "expected": 1}, {"id": "b", "expected": 2}, {"id": "c", "expected": 3}]
    usecase = fake_usecase({"a": {"answer": 1}, "b": {"answer": 99}, "c": RuntimeError("boom")})

    rows = run_eval(usecase, "fake", examples, ["openai:gpt-4o-mini"])

    assert [r["scores"] for r in rows] == [{"correct": True}, {"correct": False}, {}]
    assert rows[2]["error"] == "RuntimeError: boom"


def test_summarize_counts_errors_as_failures():
    rows = [
        {"model": "m", "scores": {"correct": True}, "error": None},
        {"model": "m", "scores": {"correct": False}, "error": None},
        {"model": "m", "scores": {}, "error": "RuntimeError: boom"},
        {"model": "m", "scores": {"correct": True}, "error": None},
    ]
    assert summarize(rows) == {"m": {"examples": 4, "correct": 0.5, "errors": 1}}