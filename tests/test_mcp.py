import asyncio

from mcp import Client

from src.mcp_server import mcp


def test_mcp_discovery_and_validation():
    async def check():
        async with Client(mcp) as client:
            tools = await client.list_tools()
            assert {"ask_agent", "ask_agent_fast", "rag_answer", "search_documents"} <= {
                t.name for t in tools.tools
            }
            response = await client.call_tool("search_documents", {"question": "q", "k": -1})
            assert response.is_error

    asyncio.run(check())
