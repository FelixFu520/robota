from openai import OpenAI
import os


class OpenAIWithExtraBody:
    """包装 OpenAI 客户端，自动添加默认的 extra_body"""
    
    def __init__(self, client: OpenAI, default_extra_body: dict):
        self._client = client
        self._default_extra_body = default_extra_body
    
    @property
    def chat(self):
        return ChatCompletionsWrapper(self._client.chat, self._default_extra_body)
    
    def __getattr__(self, name):
        # 转发其他属性到原始客户端
        return getattr(self._client, name)


class ChatCompletionsWrapper:
    """包装 chat，自动添加默认的 extra_body"""
    
    def __init__(self, chat, default_extra_body: dict):
        self._chat = chat
        self._default_extra_body = default_extra_body
        # 预包装 completions 对象
        self._completions = CompletionsWrapper(chat.completions, default_extra_body)
    
    @property
    def completions(self):
        return self._completions
    
    def __getattr__(self, name):
        # 转发其他属性到原始 chat
        return getattr(self._chat, name)


class CompletionsWrapper:
    """包装 completions，自动添加默认的 extra_body"""
    
    def __init__(self, completions, default_extra_body: dict):
        self._completions = completions
        self._default_extra_body = default_extra_body
    
    def create(self, *args, **kwargs):
        # 如果调用时没有提供 extra_body，则使用默认值
        if 'extra_body' not in kwargs:
            kwargs['extra_body'] = self._default_extra_body
        return self._completions.create(*args, **kwargs)
    
    def __getattr__(self, name):
        # 转发其他属性到原始 completions
        return getattr(self._completions, name)


base_client = OpenAI(
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

client = OpenAIWithExtraBody(
    base_client,
    default_extra_body={
        "thinking": {
            "type": "disabled"  # 不使用深度思考能力
            # "type": "enabled" # 使用深度思考能力
            # "type": "auto" # 模型自行判断是否使用深度思考能力
        }
    },
)


response = client.chat.completions.create(
    messages=[
        # 把用户提示词传进来content
        {'role': 'user', 'content': "介绍一下你自己"},
    ],
    model='doubao-seed-1-6-251015',  # 调用的模型
    stream=True,  # True 是流逝返回，False是非流逝返回
)

# stream=False的时候，打开这个，启用非流式返回
# print(response.choices[0].message.content)

# stream=True的时候，启用流示返回
print('*'*100)
for chunk in response:
    # print("|", end="|", flush=True)
    print(chunk.choices[0].delta.content, end="", flush=True)
print('*'*100)