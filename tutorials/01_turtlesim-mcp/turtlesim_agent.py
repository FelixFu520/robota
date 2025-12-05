import os
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

model = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

async def main():
    mcp_script = os.path.join(os.path.dirname(__file__), "..", "ros", "start_turtlesim_mcp.sh")
    mcp_script = os.path.abspath(mcp_script)
    print(f"mcp_script: {mcp_script}")
    client = MultiServerMCPClient(  
        {
            "turtlesim": {
                "transport": "streamable_http",  # HTTP-based remote server
                # Ensure you start your weather server on port 8000
                "url": "http://localhost:8000/mcp",
            }
        }
    )

    tools = await client.get_tools()
    agent = create_agent(model, tools)
    user_input = input("Hi! I am the ROS2 agent. How can i help you? :) \n \n Enter:")
    while user_input != "exit":
        msg1 = {"messages": [("user", user_input)]}
        res1 = await agent.ainvoke(msg1)
        for m in res1['messages']:
            m.pretty_print()
        user_input = input("Enter: ")

if __name__ == "__main__":
    asyncio.run(main())