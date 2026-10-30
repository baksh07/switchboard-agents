from types import SimpleNamespace

import pytest

from switchboard.core.evals import cost_usd, run_eval, summarize
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

    def run_example(llm, example, config, tools="local"):
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


def row(correct, error=None, latency=1.0, tokens=(100, 10), cost=0.001):
    return {
        "model": "m", "scores": {} if error else {"correct": correct}, "error": error,
        "latency_s": latency, "input_tokens": tokens[0], "output_tokens": tokens[1], "cost_usd": cost,
    }


def test_summarize_counts_errors_as_failures():
    rows = [row(True), row(False), row(None, error="RuntimeError: boom"), row(True)]
    summary = summarize(rows)["m"]
    assert summary["correct"] == 0.5
    assert summary["errors"] == 1


def test_summarize_uses_median_latency_and_mean_tokens_and_cost():
    rows = [row(True, latency=1.0, tokens=(100, 10)), row(True, latency=2.0, tokens=(300, 30)),
            row(True, latency=60.0, tokens=(200, 20))]
    summary = summarize(rows)["m"]
    assert summary["median_latency_s"] == 2.0  # one slow outlier (e.g. a rate-limit wait) doesn't skew it
    assert summary["avg_tokens_in"] == 200
    assert summary["avg_tokens_out"] == 20
    assert summary["cost_per_lead_usd"] == pytest.approx(0.001)


def test_cost_uses_price_table_and_is_none_for_unknown_models():
    assert cost_usd("openai:gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert cost_usd("someone:unknown-model", 1000, 1000) is None