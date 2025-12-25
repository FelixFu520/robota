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

from robota.agent.turtlesim import TurtlesimAgent
from robota.mcp.math import mcp_math_path

async def _get_tools():
    mcp_client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": [mcp_math_path],
        },
    })
    return await mcp_client.get_tools()

async def test_agent():
    tools = await _get_tools()
    agent = TurtlesimAgent(tools=tools)

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
            
            # 使用统一的流式输出方法，支持工具调用和普通对话
            async for event in agent.astream_with_tools(user_input):
                if is_first_token:
                    print(f"Time: {time.time() - time_start:.2f} seconds ", end="", flush=True)
                    is_first_token = False
                
                event_type = event["type"]
                
                if event_type == "token":
                    # 实时打印每个token
                    print(event["content"], end="", flush=True)
                
                elif event_type == "tool_call_start":
                    # 显示工具调用信息
                    print(f"\n  🔧 [调用工具: {event['tool_name']}]")
                    print(f"     参数: {event['tool_args']}", flush=True)
                
                elif event_type == "tool_call_end":
                    # 显示工具返回结果
                    print(f"  ✅ [工具返回: {event['tool_result']}]\n🤖 Agent: ", end="", flush=True)
                
                elif event_type == "error":
                    print(f"\n  ❌ 错误: {event['error']}", flush=True)
                
                elif event_type == "done":
                    print("\n")
                    break
            
            print("-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(test_agent())
