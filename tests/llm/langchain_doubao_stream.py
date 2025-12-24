import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool


model = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

agent = create_agent(
    model,
)
print("Agent")
print('*'*100)
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "介绍一下你自己"}]},  
    stream_mode="messages",
    extra_body={
        "thinking": {
            "type": "disabled"  # 不使用深度思考能力
            # "type": "enabled" # 使用深度思考能力
            # "type": "auto" # 模型自行判断是否使用深度思考能力
        }
    }
):
    print(token.text, end="|", flush=True)
print('*'*100)

print("Model")
print('*'*100)
for chunk in model.stream("介绍一下你自己", extra_body={
        "thinking": {
            "type": "disabled"  # 不使用深度思考能力
            # "type": "enabled" # 使用深度思考能力
            # "type": "auto" # 模型自行判断是否使用深度思考能力
        }
    }):
    print(chunk.text, end="|", flush=True)
print('*'*100)
