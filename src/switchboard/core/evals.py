
"""Generic eval harness: run any use case's dataset against several models and score every example.

A use case plugs in by providing `switchboard/usecases/<name>/evaluation.py` with:
    run_example(llm, example, config) -> outcome dict
    score(example, outcome) -> {metric_name: bool}
and a dataset at `evals/datasets/<name>.jsonl` (one JSON object per line with an "id").
"""

import argparse
import importlib
import json
from pathlib import Path
from types import ModuleType
from uuid import uuid4

from switchboard.core.llm import get_llm

DATASETS_DIR = Path("evals/datasets")


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_eval(usecase: ModuleType, name: str, examples: list[dict], models: list[str]) -> list[dict]:
    """Run every example on every model. A crash in one example is recorded as an error, not raised."""
    rows = []
    for model in models:
        llm = get_llm(model)
        for example in examples:
            config = {
                "configurable": {"thread_id": str(uuid4())},
                "run_name": f"eval:{name}",
                "tags": ["eval", model],
                "metadata": {"example_id": example["id"], "model": model},
            }
            try:
                outcome = usecase.run_example(llm, example, config)
                scores, error = usecase.score(example, outcome), None
            except Exception as exc:
                outcome, scores, error = {}, {}, f"{type(exc).__name__}: {exc}"
            rows.append({"model": model, "id": example["id"], "scores": scores, "outcome": outcome, "error": error})
            status = error or " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in scores.items())
            print(f"[{model}] {example['id']}: {status}")
    return rows


def summarize(rows: list[dict]) -> dict[str, dict]:
    """Per model: share of examples passing each metric. Errored examples count as failures."""
    metrics = sorted({m for row in rows for m in row["scores"]})
    summary = {}
    for model in dict.fromkeys(row["model"] for row in rows):
        model_rows = [r for r in rows if r["model"] == model]
        summary[model] = {
            "examples": len(model_rows),
            **{m: sum(r["scores"].get(m, False) for r in model_rows) / len(model_rows) for m in metrics},
            "errors": sum(r["error"] is not None for r in model_rows),
        }
    return summary


def print_table(summary: dict[str, dict]) -> None:
    columns = list(next(iter(summary.values())))
    print("\n| model | " + " | ".join(columns) + " |")
    print("|---" * (len(columns) + 1) + "|")
    for model, stats in summary.items():
        cells = [f"{v:.0%}" if isinstance(v, float) else str(v) for v in stats.values()]
        print(f"| {model} | " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a use case's eval dataset across models.")
    parser.add_argument("--usecase", default="lead_qualification")
    parser.add_argument("--models", nargs="+", required=True, help='e.g. openai:gpt-4o-mini groq:openai/gpt-oss-120b')
    parser.add_argument("--dataset", type=Path, default=None, help="Default: evals/datasets/<usecase>.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="Write every row (with outcomes) to this JSONL file.")
    args = parser.parse_args()

    usecase = importlib.import_module(f"switchboard.usecases.{args.usecase}.evaluation")
    examples = load_dataset(args.dataset or DATASETS_DIR / f"{args.usecase}.jsonl")
    rows = run_eval(usecase, args.usecase, examples, args.models)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print_table(summarize(rows))


if __name__ == "__main__":
    main()