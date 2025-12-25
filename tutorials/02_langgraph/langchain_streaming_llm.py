from langchain.agents import create_agent
import os
from langchain_openai import ChatOpenAI
from langchain.tools import tool
model = ChatOpenAI(
    model="doubao-seed-1-6-251015",  # Specify a model available on OpenRouter
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"

agent = create_agent(
    model=model,
    tools=[get_weather],
)
for token, metadata in agent.stream(  
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",
):
    print(f"node: {metadata['langgraph_node']}")
    print(f"content: {token.content_blocks}")
    print("\n")