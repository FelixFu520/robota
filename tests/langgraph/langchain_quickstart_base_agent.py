import os
import json
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

llm = init_chat_model(
    model="doubao-seed-1-6-251015",
    model_provider="openai",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    temperature=0
)

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",

)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print('-'*100)
for m in result["messages"]:
    m.pretty_print()
    print('*'*100)

print('-'*100)
for m in result["messages"]:
    print(m)
    print('*'*100)