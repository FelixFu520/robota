import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


basic_model = ChatOpenAI(
    model="doubao-seed-1-6-lite-251015",  # Specify a model available on OpenRouter
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)
advanced_model = ChatOpenAI(
    model="doubao-seed-1-6-251015",  # Specify a model available on OpenRouter
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 1:
        # Use an advanced model for longer conversations
        print("Using advanced model")
        model = advanced_model
    else:
        print("Using basic model")
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,  # Default model
    tools=[],
    middleware=[dynamic_model_selection]
)

response = agent.invoke({"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]})
print(response)

response = agent.invoke({"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}, {"role": "user", "content": "What is the weather in Beijing?"}]})
print(response)


