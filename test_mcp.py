import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


async def main():
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "mcp_server/server.py"],
        cwd="."
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            print("\nMCP connection established.")

            tools = await session.list_tools()

            print("\nAvailable MCP tools:")

            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            print("\nTesting stock analysis tool...")

            result = await session.call_tool(
                "get_stock_technical_analysis",
                {"ticker": "RELIANCE.NS"}
            )

            for item in result.content:
                if hasattr(item, "text"):
                    print(item.text)


if __name__ == "__main__":
    asyncio.run(main())