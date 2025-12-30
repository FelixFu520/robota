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


# Use extra_body as a direct parameter instead of model_kwargs to avoid warnings
# Note: extra_body is a supported parameter of ChatOpenAI and should be passed directly
default_extra_body = {
    "thinking": {"type": "disabled"}
}

DOUBAO_SEED_1_6_251015_NOTHINKING = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    streaming=True,  # 启用流式输出
    extra_body=default_extra_body,
)

DOUBAO_SEED_1_6_LITE_251015_NOTHINKING = ChatOpenAI(
    model="doubao-seed-1-6-lite-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    extra_body=default_extra_body,
)