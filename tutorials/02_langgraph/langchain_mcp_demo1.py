import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import os


async def main():
    client = MultiServerMCPClient(  
        {
            # "math": {
            #     "transport": "stdio",  # Local subprocess communication
            #     "command": "python",
            #     # Absolute path to your math_server.py file
            #     "args": ["/home/drobotics/projects/robota/tests/langgraph/langchain_mcp_demo1_mathserver.py"],
            # },
            "weather": {
                "transport": "streamable_http",  # HTTP-based remote server
                # Ensure you start your weather server on port 8000
                "url": "http://localhost:8001/mcp",
            }
        }
    )

    tools = await client.get_tools()  
    model = ChatOpenAI(
        model="doubao-seed-1-6-251015",  # Specify a model available on OpenRouter
        api_key=os.environ.get("ARK_API_KEY"),
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    agent = create_agent(
        model,
        tools  
    )
    # math_response = await agent.ainvoke(
    #     {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    # )
    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
    )
    
    # print("Math response:", math_response)
    # for m in math_response["messages"]:
    #     m.pretty_print()
    #     print('*'*100)
    # print('='*100)
    print("Weather response:", weather_response)
    for m in weather_response["messages"]:
        m.pretty_print()
        print('*'*100)


if __name__ == "__main__":
    asyncio.run(main())