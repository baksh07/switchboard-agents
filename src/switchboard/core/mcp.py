import asyncio

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient


def _as_sync(tool: BaseTool) -> BaseTool:
    """MCP tools are async-only; wrap one so .invoke() works. Each call runs on its own short event loop."""

    def call(**kwargs):
        result = asyncio.run(tool.ainvoke(kwargs))
        if isinstance(result, list):  # MCP returns content blocks; keep just the text, like a local tool's output
            return "\n".join(block["text"] for block in result if block.get("type") == "text")
        return result

    return StructuredTool.from_function(
        func=call, name=tool.name, description=tool.description, args_schema=tool.args_schema
    )

def load_mcp_tools(connections: dict)-> list[BaseTool]:
    client = MultiServerMCPClient(connections=connections)
    return [_as_sync(t) for t in asyncio.run(client.get_tools())]