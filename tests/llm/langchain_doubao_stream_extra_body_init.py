import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool
import time


class ChatOpenAIWithExtraBody:
    """包装 ChatOpenAI，自动添加默认的 extra_body"""
    
    def __init__(self, model: ChatOpenAI, default_extra_body: dict):
        self._model = model
        self._default_extra_body = default_extra_body
    
    def bind(self, **kwargs):
        # 如果调用时没有提供 extra_body，则使用默认值
        if 'extra_body' not in kwargs:
            kwargs['extra_body'] = self._default_extra_body.copy()
        # 如果提供了 extra_body，需要合并默认值
        elif 'extra_body' in kwargs:
            # 合并默认值和提供的值
            merged_extra_body = self._default_extra_body.copy()
            merged_extra_body.update(kwargs['extra_body'])
            kwargs['extra_body'] = merged_extra_body
        bound_model = self._model.bind(**kwargs)
        # 返回一个包装后的绑定模型，确保继续使用默认值
        return ChatOpenAIWithExtraBody(bound_model, self._default_extra_body)
    
    def stream(self, *args, **kwargs):
        # 如果调用时没有提供 extra_body，则使用默认值
        if 'extra_body' not in kwargs:
            kwargs['extra_body'] = self._default_extra_body.copy()
        else:
            # 合并默认值和提供的值
            merged_extra_body = self._default_extra_body.copy()
            merged_extra_body.update(kwargs['extra_body'])
            kwargs['extra_body'] = merged_extra_body
        return self._model.stream(*args, **kwargs)
    
    def invoke(self, *args, **kwargs):
        # 如果调用时没有提供 extra_body，则使用默认值
        if 'extra_body' not in kwargs:
            kwargs['extra_body'] = self._default_extra_body.copy()
        else:
            # 合并默认值和提供的值
            merged_extra_body = self._default_extra_body.copy()
            merged_extra_body.update(kwargs['extra_body'])
            kwargs['extra_body'] = merged_extra_body
        return self._model.invoke(*args, **kwargs)
    
    async def astream(self, *args, **kwargs):
        # 如果调用时没有提供 extra_body，则使用默认值
        if 'extra_body' not in kwargs:
            kwargs['extra_body'] = self._default_extra_body.copy()
        else:
            # 合并默认值和提供的值
            merged_extra_body = self._default_extra_body.copy()
            merged_extra_body.update(kwargs['extra_body'])
            kwargs['extra_body'] = merged_extra_body
        return await self._model.astream(*args, **kwargs)
    
    async def ainvoke(self, *args, **kwargs):
        # 如果调用时没有提供 extra_body，则使用默认值
        if 'extra_body' not in kwargs:
            kwargs['extra_body'] = self._default_extra_body.copy()
        else:
            # 合并默认值和提供的值
            merged_extra_body = self._default_extra_body.copy()
            merged_extra_body.update(kwargs['extra_body'])
            kwargs['extra_body'] = merged_extra_body
        return await self._model.ainvoke(*args, **kwargs)
    
    def __getattr__(self, name):
        # 转发其他属性到原始模型
        return getattr(self._model, name)


base_model = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

default_extra_body = {
    "thinking": {
        "type": "disabled"  # 不使用深度思考能力
        # "type": "enabled" # 使用深度思考能力
        # "type": "auto" # 模型自行判断是否使用深度思考能力
    }
}

# 先在初始化时绑定 extra_body，然后包装
bound_model = base_model.bind(extra_body=default_extra_body)
model = ChatOpenAIWithExtraBody(
    bound_model,
    default_extra_body=default_extra_body,
)

agent = create_agent(
    model,
)
print("Agent")
print('*'*100)
time_start = time.time()
Time_flag = True
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "介绍一下你自己"}]},  
    stream_mode="messages",
):
    time_end = time.time()
    if Time_flag:
        print(f"Time: {time_end - time_start:.2f} seconds", end="", flush=True)
        Time_flag = False
    print("|", end="", flush=True)
    print(token.text, end="", flush=True)
print('*'*100)

print("Agent")
print('*'*100)
time_start = time.time()
Time_flag = True
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "介绍一下你自己"}]},  
    stream_mode="messages",
):
    time_end = time.time()
    if Time_flag:
        print(f"Time: {time_end - time_start:.2f} seconds", end="", flush=True)
        Time_flag = False
    print("|", end="", flush=True)
    print(token.text, end="", flush=True)
print('*'*100)

print("Model")
print('*'*100)
time_start = time.time()
Time_flag = True
for chunk in model.stream("介绍一下你自己"):
    time_end = time.time()
    if Time_flag:
        print(f"Time: {time_end - time_start:.2f} seconds", end="", flush=True)
        Time_flag = False
    print("|", end="", flush=True)
    print(chunk.text, end="",flush=True)
print('*'*100)

print("Model")
print('*'*100)
time_start = time.time()
Time_flag = True
for chunk in model.stream("介绍一下你自己"):
    time_end = time.time()
    if Time_flag:
        print(f"Time: {time_end - time_start:.2f} seconds", end="", flush=True)
        Time_flag = False
    print("|", end="", flush=True)
    print(chunk.text, end="",flush=True)
print('*'*100)