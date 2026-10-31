"""How much of each lead's input is tool definitions? Measured with the provider's own token counts."""

import argparse
import statistics
import tempfile
from pathlib import Path
from uuid import uuid4

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage

from switchboard.core.evals import load_dataset, token_totals
from switchboard.core.llm import get_llm
from switchboard.usecases.lead_qualification.db import make_engine
from switchboard.usecases.lead_qualification.evaluation import run_example
from switchboard.usecases.lead_qualification.tools import make_tools


def tool_definition_tokens(llm, tools) -> int:
    """Send the same tiny prompt with and without the tools bound; the difference is what the definitions cost."""
    probe = [HumanMessage("Hi")]
    plain = llm.invoke(probe).usage_metadata["input_tokens"]
    with_tools = llm.bind_tools(tools).invoke(probe).usage_metadata["input_tokens"]
    return with_tools - plain


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the input-token share of the tool definitions.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", type=Path, default=Path("evals/datasets/lead_qualification.jsonl"))
    args = parser.parse_args()

    llm = get_llm(args.model)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = make_engine(f"sqlite:///{Path(tmp) / 'crm.db'}")
        per_call = tool_definition_tokens(llm, make_tools(engine))
        engine.dispose()
    print(f"Tool definitions: {per_call} input tokens per agent call\n")

    shares = []
    for example in load_dataset(args.dataset):
        usage = UsageMetadataCallbackHandler()
        config = {"configurable": {"thread_id": str(uuid4())}, "callbacks": [usage]}
        outcome = run_example(llm, example, config)
        input_tokens, _ = token_totals(usage)
        tool_tokens = per_call * outcome["agent_calls"]
        shares.append(tool_tokens / input_tokens)
        print(
            f"{example['id']}: {outcome['agent_calls']} agent calls, "
            f"{tool_tokens:,} of {input_tokens:,} input tokens are tool definitions ({tool_tokens / input_tokens:.0%})"
        )

    print(f"\nMedian share: {statistics.median(shares):.0%}")


if __name__ == "__main__":
    main()
