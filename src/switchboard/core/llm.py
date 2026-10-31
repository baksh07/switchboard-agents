import argparse
import os
import time

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

load_dotenv()

DEFAULT_MODEL = "openai:gpt-4o-mini"


def get_llm(model: str | None = None, **kwargs) -> BaseChatModel:
    spec = model or os.getenv("MODEL") or DEFAULT_MODEL
    provider, sep, name = spec.partition(":")
    if not sep or not provider or not name:
        raise ValueError(f"Model must look like 'provider:model', got {spec!r}")
    kwargs.setdefault("temperature", 0)
    kwargs.setdefault("max_retries", 6)
    return init_chat_model(name, model_provider=provider, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send one primpt to a model.")
    parser.add_argument("--model", default=None, help='e.g. "openai/gpt-oss-120b"')
    parser.add_argument("prompt", nargs="?", default="In one sentence: what does a CRM do?")
    args = parser.parse_args()

    llm = get_llm(args.model)
    start = time.perf_counter()
    reply = llm.invoke(args.prompt)
    elapsed = time.perf_counter() - start

    print(reply.content)
    print(f"\n[{type(llm).__name__}] {elapsed:.2f}s  tokens: {reply.usage_metadata}")
