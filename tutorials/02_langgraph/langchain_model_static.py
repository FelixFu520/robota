import os
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import ToolMessage

model = ChatOpenAI(
    model="doubao-seed-1-6-251015",  # Specify a model available on OpenRouter
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

response = model.invoke("What is the weather in Tokyo?")
print(response)