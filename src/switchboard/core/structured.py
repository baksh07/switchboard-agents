from typing import TypeVar
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)

class StructuredOutputError(Exception):
    """
    The model didnot return valide data for the schema even after retrying
    """

def invoke_structured(
    llm: BaseChatModel,
    schema: type[T],
    messages: list[BaseMessage],
    retries: int = 1,
) -> T:
    """Return a validated `schema` instance; on invalid output, retry with the error shown to the model."""
    structured_llm = llm.with_structured_output(schema, include_raw=True)
    error = None
    for _ in range(retries + 1):
        result = structured_llm.invoke(messages)
        if result["parsed"] is not None:
            return result["parsed"]
        error = result["parsing_error"]
        messages = [
            *messages,
            HumanMessage(f"Your previous answer was invalid: {error}\nAnswer again, following the schema exactly."),
        ]
    raise StructuredOutputError(f"{schema.__name__}: no valid output after {retries + 1} attempts: {error}")