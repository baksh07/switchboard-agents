"""Generic eval harness: run any use case's dataset against several models and score every example.

A use case plugs in by providing `switchboard/usecases/<name>/evaluation.py` with:
    run_example(llm, example, config, tools) -> outcome dict   (tools: "local" or "mcp")
    score(example, outcome) -> {metric_name: bool}
and a dataset at `evals/datasets/<name>.jsonl` (one JSON object per line with an "id").
"""

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path
from types import ModuleType
from uuid import uuid4

from langchain_core.callbacks import UsageMetadataCallbackHandler

from switchboard.core.llm import get_llm

DATASETS_DIR = Path("evals/datasets")

# USD per 1M tokens (input, output). Check the provider's pricing page before publishing numbers.
# Checked 2026-10-29: developers.openai.com/api/docs/pricing, console.groq.com/docs/models
PRICES_PER_MILLION = {
    "openai:gpt-4o-mini": (0.15, 0.60),
    "groq:openai/gpt-oss-120b": (0.15, 0.60),
}


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def token_totals(usage: UsageMetadataCallbackHandler) -> tuple[int, int]:
    """Sum input and output tokens over every LLM call the callback saw."""
    calls = usage.usage_metadata.values()
    return sum(u["input_tokens"] for u in calls), sum(u["output_tokens"] for u in calls)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model not in PRICES_PER_MILLION:
        return None
    price_in, price_out = PRICES_PER_MILLION[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def run_eval(
    usecase: ModuleType, name: str, examples: list[dict], models: list[str], tools: str = "local"
) -> list[dict]:
    """Run every example on every model. A crash in one example is recorded as an error, not raised."""
    rows = []
    for model in models:
        llm = get_llm(model)
        for example in examples:
            usage = UsageMetadataCallbackHandler()
            config = {
                "configurable": {"thread_id": str(uuid4())},
                "run_name": f"eval:{name}",
                "tags": ["eval", model, f"tools:{tools}"],
                "metadata": {"example_id": example["id"], "model": model, "tools": tools},
                "callbacks": [usage],
            }
            start = time.perf_counter()
            try:
                outcome = usecase.run_example(llm, example, config, tools=tools)
                scores, error = usecase.score(example, outcome), None
            except Exception as exc:
                outcome, scores, error = {}, {}, f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - start
            input_tokens, output_tokens = token_totals(usage)

            rows.append({
                "model": model,
                "id": example["id"],
                "scores": scores,
                "latency_s": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd(model, input_tokens, output_tokens),
                "outcome": outcome,
                "error": error,
            })
            status = error or " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in scores.items())
            print(f"[{model}] {example['id']}: {status}  ({latency:.1f}s, {input_tokens}+{output_tokens} tokens)")
    return rows


def summarize(rows: list[dict]) -> dict[str, dict]:
    """Per model: share of examples passing each metric (errors count as failures), plus cost and speed."""
    metrics = sorted({m for row in rows for m in row["scores"]})
    summary = {}
    for model in dict.fromkeys(row["model"] for row in rows):
        model_rows = [r for r in rows if r["model"] == model]
        costs = [r["cost_usd"] for r in model_rows]
        summary[model] = {
            "examples": len(model_rows),
            **{m: sum(r["scores"].get(m, False) for r in model_rows) / len(model_rows) for m in metrics},
            "errors": sum(r["error"] is not None for r in model_rows),
            "median_latency_s": statistics.median(r["latency_s"] for r in model_rows),
            "avg_tokens_in": statistics.mean(r["input_tokens"] for r in model_rows),
            "avg_tokens_out": statistics.mean(r["output_tokens"] for r in model_rows),
            "cost_per_lead_usd": None if None in costs else statistics.mean(costs),
        }
    return summary


def format_cell(column: str, value) -> str:
    if value is None:
        return "n/a"
    if column == "median_latency_s":
        return f"{value:.1f}s"
    if column.startswith("avg_tokens"):
        return f"{value:,.0f}"
    if column == "cost_per_lead_usd":
        return f"${value:.5f}"
    if isinstance(value, float):
        return f"{value:.0%}"
    return str(value)


def print_table(summary: dict[str, dict]) -> None:
    columns = list(next(iter(summary.values())))
    print("\n| model | " + " | ".join(columns) + " |")
    print("|---" * (len(columns) + 1) + "|")
    for model, stats in summary.items():
        print(f"| {model} | " + " | ".join(format_cell(c, v) for c, v in stats.items()) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a use case's eval dataset across models.")
    parser.add_argument("--usecase", default="lead_qualification")
    parser.add_argument("--models", nargs="+", required=True, help='e.g. openai:gpt-4o-mini groq:openai/gpt-oss-120b')
    parser.add_argument("--dataset", type=Path, default=None, help="Default: evals/datasets/<usecase>.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="Write every row (with outcomes) to this JSONL file.")
    parser.add_argument("--tools", choices=["local", "mcp"], default="local", help="Call tools in-process or via MCP.")
    args = parser.parse_args()

    usecase = importlib.import_module(f"switchboard.usecases.{args.usecase}.evaluation")
    examples = load_dataset(args.dataset or DATASETS_DIR / f"{args.usecase}.jsonl")
    rows = run_eval(usecase, args.usecase, examples, args.models, tools=args.tools)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print_table(summarize(rows))


if __name__ == "__main__":
    main()