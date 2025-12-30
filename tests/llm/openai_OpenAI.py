from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)


response = client.chat.completions.create(
    messages=[
        # 把用户提示词传进来content
        {'role': 'user', 'content': "介绍一下你自己"},
    ],
    model='doubao-seed-1-6-251015',  # 调用的模型
    stream=True,  # True 是流逝返回，False是非流逝返回
    extra_body={
        "thinking": {
            "type": "disabled"  # 不使用深度思考能力
            # "type": "enabled" # 使用深度思考能力
            # "type": "auto" # 模型自行判断是否使用深度思考能力
        }
    },
)

# stream=False的时候，打开这个，启用非流式返回
# print(response.choices[0].message.content)

# stream=True的时候，启用流示返回
print('*'*100)
for chunk in response:
    print("|", end="|", flush=True)
    print(chunk.choices[0].delta.content, end="", flush=True)
print('*'*100)