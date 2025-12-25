import os
from langchain_openai import ChatOpenAI
 

__all__ = [
    'DOUBAO_SEED_1_6_251015',
    'DOUBAO_SEED_1_6_LITE_251015',
    'DOUBAO_SEED_1_6_251015_NOTHINKING',
    'DOUBAO_SEED_1_6_LITE_251015_NOTHINKING',
]


DOUBAO_SEED_1_6_251015 = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)


DOUBAO_SEED_1_6_LITE_251015 = ChatOpenAI(
    model="doubao-seed-1-6-lite-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)


# Use model_kwargs to pass extra_body (this ensures it's preserved even when using tools)
# Note: Using model_kwargs instead of .bind() because .bind() loses extra_body when
# create_agent() calls bind_tools() internally.
default_extra_body = {
    "thinking": {"type": "disabled"}
}

DOUBAO_SEED_1_6_251015_NOTHINKING = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    streaming=True,  # 启用流式输出
    model_kwargs={"extra_body": default_extra_body},
)

DOUBAO_SEED_1_6_LITE_251015_NOTHINKING = ChatOpenAI(
    model="doubao-seed-1-6-lite-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    model_kwargs={"extra_body": default_extra_body},
)