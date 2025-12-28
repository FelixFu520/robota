"""
Test script for Turtlesim Agent.

This script tests the agent locally without requiring langgraph dev.
Run this to test the agent before deploying with langgraph dev.
"""
import asyncio
import os
import time
# 在导入其他模块之前先导入 logging，确保日志配置生效
from robota.utils import logging  # noqa: F401
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool

from robota.agent.turtlesim_text import TurtlesimAgentText
from robota.mcp.math import mcp_math_path


def _wrap_async_tool_as_sync(async_tool: BaseTool) -> BaseTool:
    """
    将异步工具包装为同步工具，以便在同步的 stream() 方法中使用。
    
    Args:
        async_tool: 异步的 BaseTool 实例（通常是 StructuredTool）
        
    Returns:
        同步的 BaseTool 实例
    """
    # 检查工具是否支持异步调用
    if not hasattr(async_tool, 'ainvoke'):
        # 如果没有 ainvoke 方法，可能是同步工具，直接返回
        return async_tool
    
    # 创建同步包装函数
    def sync_wrapper(**kwargs):
        """同步包装器，使用 asyncio.run 执行异步工具"""
        # 在同步上下文中，使用 asyncio.run 创建新的事件循环
        # 这会在新的事件循环中运行异步工具
        return asyncio.run(async_tool.ainvoke(kwargs))
    
    # 使用 StructuredTool.from_function 创建同步工具
    # 获取原始工具的 args_schema
    args_schema = None
    if hasattr(async_tool, 'args_schema'):
        args_schema = async_tool.args_schema
    
    sync_tool = StructuredTool.from_function(
        func=sync_wrapper,
        name=async_tool.name,
        description=async_tool.description,
        args_schema=args_schema,
    )
    
    return sync_tool


async def _get_tools():
    mcp_client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": [mcp_math_path],
        },
    })
    async_tools = await mcp_client.get_tools()
    # 将所有异步工具转换为同步工具
    sync_tools = [_wrap_async_tool_as_sync(tool) for tool in async_tools]
    return sync_tools

def is_tool_message(token):
    """
    检查 token 是否是工具返回的消息（ToolMessage）
    
    Args:
        token: 从 agent.stream 返回的 token 对象
        
    Returns:
        bool: 如果是 ToolMessage 返回 True，否则返回 False
    """
    return (
        isinstance(token, ToolMessage) or
        (hasattr(token, "tool_call_id") and token.tool_call_id) or
        (hasattr(token, "__class__") and "ToolMessage" in token.__class__.__name__)
    )

async def test_agent():
    tools = await _get_tools()
    agent = TurtlesimAgentText(tools=tools)

    print("\n" + "="*60)
    print("🤖 Turtlesim Agent is ready!")
    print("="*60)
    print("\nYou can now interact with the agent.")
    print("Type 'exit' to quit.\n")
    
    # Interactive loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\n🤖 Agent: ", end="", flush=True)
            time_start = time.time()
            is_first_token = True
            
            # 使用同步流式输出方法，参考 01_test_RobotAgent_stream.py
            for token, metadata in agent.stream({"messages": [{"role": "user", "content": user_input}]}):
                if is_first_token:
                    print(f"Time: {time.time() - time_start:.2f} seconds ", end="", flush=True)
                    is_first_token = False
                
                # 过滤掉工具返回的消息（ToolMessage），只输出 LLM 生成的文本消息
                if is_tool_message(token):
                    continue
                
                # 提取文本内容（如果有）
                if hasattr(token, "text") and token.text:
                    print(token.text, end="", flush=True)
            
            print('\n' + '-' * 60 + '\n')
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    asyncio.run(test_agent())
