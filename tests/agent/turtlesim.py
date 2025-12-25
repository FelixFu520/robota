"""
Test script for Turtlesim Agent.

This script tests the agent locally without requiring langgraph dev.
Run this to test the agent before deploying with langgraph dev.
"""
import asyncio
import os
import time
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
            
            print("\n🤖 Agent thinking...")
            print("\n" + "-"*60)
            print("Agent Response:")
            print("-"*60)
            time_start = time.time()
            Time_flag = True
            
            # 检测是否需要工具调用来决定流式策略
            needs_tools = any(keyword in user_input.lower() for keyword in ['计算', '加', '乘', '数学', 'calculate', 'add', 'multiply'])
            
            if needs_tools:
                # 对于工具调用，使用代理的异步流式输出
                async for token, metadata in agent.agent.astream(
                    {"messages": [{"role": "user", "content": user_input}]},
                    stream_mode="messages",
                ):
                    time_end = time.time()
                    if Time_flag:
                        print(f"Time: {time_end - time_start:.2f} seconds ", end="", flush=True)
                        Time_flag = False
                    
                    # Extract text content from token
                    if hasattr(token, 'content_blocks') and token.content_blocks:
                        for block in token.content_blocks:
                            if block.get("type") == "text" and block.get("text"):
                                print(block["text"], end="", flush=True)
                            elif block.get("type") == "tool_call_chunk":
                                if block.get("name"):
                                    print(f"\n[🔧 Calling tool: {block['name']}] ", end="", flush=True)
                    elif hasattr(token, 'text') and token.text:
                        print(token.text, end="", flush=True)
                    elif hasattr(token, 'content') and token.content:
                        print(token.content, end="", flush=True)
            else:
                # 对于普通对话，直接使用模型的流式输出获得更好的体验
                for chunk in agent.model.stream(f"你是一个能够控制Turtlesim机器人的ROS 2助手。{user_input}"):
                    time_end = time.time()
                    if Time_flag:
                        print(f"Time: {time_end - time_start:.2f} seconds ", end="", flush=True)
                        Time_flag = False
                    
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
            
            print("\n" + "-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(test_agent())
