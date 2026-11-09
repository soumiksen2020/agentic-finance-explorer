import sys
from pathlib import Path

from crewai import Agent, LLM
from crewai.mcp import MCPServerStdio

project_root = Path(__file__).resolve().parent
mcp_server_path = project_root / "mcp_server" / "server.py"

mcp_stock_server = MCPServerStdio(
    command=sys.executable,
    args=[str(mcp_server_path)],
    cache_tools_list=True
)

agent = Agent(
    role="MCP Integration Tester",
    goal="Test stock analysis through MCP",
    backstory="You test MCP tool connectivity.",
    llm=LLM(model="groq/openai/gpt-oss-120b"),
    mcps=[mcp_stock_server],
    verbose=False
)

tools = agent.get_mcp_tools(agent.mcps)

print("\nDiscovered MCP tools:")
for tool in tools:
    print(f"- {tool.name}: {tool.description}")

print("\nTesting MCP tool execution:")

result = tools[0].run(ticker="RELIANCE.NS")
print(result)