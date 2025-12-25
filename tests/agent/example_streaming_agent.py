#!/usr/bin/env python3
"""
流式 Agent 使用示例
演示如何实时打印 Agent 输出并支持工具调用
"""
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from robota.mcp.math import mcp_math_path
from robota.agent.turtlesim import TurtlesimAgent


async def main():
    # 初始化 MCP 工具
    print("🔧 初始化 MCP 工具...")
    mcp_client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": [mcp_math_path],
        },
    })
    tools = await mcp_client.get_tools()
    print(f"✅ 成功加载 {len(tools)} 个工具\n")
    
    # 创建 Agent
    agent = TurtlesimAgent(tools=tools)
    
    # 示例1: 普通对话（流式输出）
    print("="*60)
    print("示例1: 普通对话")
    print("="*60)
    user_input = "简单介绍你自己"
    print(f"👤 用户: {user_input}\n🤖 Agent: ", end="", flush=True)
    
    async for event in agent.astream_with_tools(user_input):
        if event["type"] == "token":
            print(event["content"], end="", flush=True)
        elif event["type"] == "done":
            print("\n")
            break
    
    # 示例2: 调用工具（流式输出 + 工具调用）
    print("\n" + "="*60)
    print("示例2: 调用工具")
    print("="*60)
    user_input = "帮我计算 123 + 456 等于多少"
    print(f"👤 用户: {user_input}\n🤖 Agent: ", end="", flush=True)
    
    async for event in agent.astream_with_tools(user_input):
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
    
    print("="*60)
    print("✅ 演示完成！")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())

