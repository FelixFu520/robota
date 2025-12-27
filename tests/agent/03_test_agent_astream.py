import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from typing import Union

 
@tool
def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    计算两数之和
    
    Args:
        a: 第一个数
        b: 第二个数

    Returns:
        两数之和

    Examples:
        >>> add(1, 2)
        3
        >>> add(-1, 1)
        0
        >>> add(0, 0)
        0
    """
    return a + b

model = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    streaming=True,
    extra_body={"thinking": {"type": "disabled"}},
)

agent = create_agent(
            model=model,
            # tools=[add],
            system_prompt="你是个智能助手，可以和用户聊天",
)


async def test_astream():
    print('test_astream:')
    messages = {"messages": [{"role": "user", "content": "你好，介绍你自己"}]}
    async for chunk in agent.astream(messages):
        print(chunk)
        # print(token.text, end="", flush=True)
    print('\n', flush=True)


if __name__ == "__main__":
    asyncio.run(test_astream())