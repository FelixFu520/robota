import os
from langchain_openai import ChatOpenAI
 

__all__ = [
    'DOUBAO_SEED_1_6_251015',
    'DOUBAO_SEED_1_6_LITE_251015',
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