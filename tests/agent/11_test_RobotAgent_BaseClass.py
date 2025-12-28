from typing import Union
import asyncio
from langchain.tools import tool
from robota.agent.base import RobotAgent
from robota.model import DOUBAO_SEED_1_6_251015_NOTHINKING


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

def test_invoke(agent: RobotAgent):
    print("test_invoke:")
    messages = {"messages": [{"role": "user", "content": "你好，介绍你自己"}]}
    response = agent.invoke(messages)
    # print(response)
    print(response["messages"][-1].content, flush=True)

async def test_ainvoke(agent: RobotAgent):
    print("test_ainvoke:")
    messages = {"messages": [{"role": "user", "content": "你好，介绍你自己"}]}
    response = await agent.ainvoke(messages)
    # print(response)
    print(response["messages"][-1].content, flush=True)

def test_stream(agent: RobotAgent):
    print('test_stream:')
    messages = {"messages": [{"role": "user", "content": "你好，介绍你自己"}]}
    response = agent.stream(messages)
    for token, metadata in response:
        # print(chunk)
        print(token.text, end="", flush=True)
    print('\n')

async def test_astream(agent: RobotAgent):
    print('test_astream:')
    messages = {"messages": [{"role": "user", "content": "你好，介绍你自己"}]}
    async for token, metadata in agent.astream(messages):
        # print(chunk)
        print(token.text, end="", flush=True)
    print('\n', flush=True)

async def test_astream_with_tools(agent: RobotAgent):
    print('test_astream_with_tools:')
    messages = {"messages": [{"role": "user", "content": "你好，介绍你自己"}]}
    async for event in agent.astream_with_tools(messages["messages"][0]["content"]):
        if event.get("type") == "token":
            print(event["content"], end="", flush=True)
    print('\n', flush=True)

    async for event in agent.astream_with_tools("你好，帮我计算 1 + 2 等于多少"):
        if event.get("type") == "token":
            print(event["content"], end="", flush=True)
    print('\n', flush=True)


if __name__ == "__main__":
    agent = RobotAgent(
        model=DOUBAO_SEED_1_6_251015_NOTHINKING,
        tools=[add],
        system_prompt="你是个智能助手，可以和用户聊天"
    )
    assert agent is not None

    # 同步调用
    test_invoke(agent)

    # 异步调用
    asyncio.run(test_ainvoke(agent))

    # 同步流式输出
    test_stream(agent)

    # 异步流式输出，加上工具后，不流式了，TODO: 找到原因
    asyncio.run(test_astream(agent))

    # 异步流式输出，支持工具调用
    asyncio.run(test_astream_with_tools(agent))

