"""MCP server exposing the mock-CRM tools over stdio.

Any MCP client can use these tools: this project's agent (--tools mcp), Claude Desktop, Claude Code, ...
The CRM database comes from CRM_DATABASE_URL (default: sqlite:///crm.db in the current directory).
"""


import os 
import sys
from langchain_mcp_adapters.tools import to_fastmcp
from mcp.server.fastmcp import FastMCP

from switchboard.usecases.lead_qualification.db import make_engine
from switchboard.usecases.lead_qualification.tools import make_tools

engine =  make_engine(os.getenv("CRM_DATABASE_URL", "sqlite:///crm.db"))
mcp = FastMCP(
    "crm", tools=[to_fastmcp(t) for t in make_tools(engine)], log_level="WARNING",
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8000")),
    )

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        # mcp.settings.host = os.getenv("MCP_HOST", "127.0.0.1")
        # mcp.settings.port = int(os.getenv("MCP_PORT", "8000"))
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
