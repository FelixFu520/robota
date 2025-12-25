#!/usr/bin/env python3
"""
测试流式打印是否实时工作
"""
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from robota.agent.turtlesim import TurtlesimAgent
from robota.mcp.math import mcp_math_path

async def test_streaming():
    """测试实时流式输出"""
    print("🔧 初始化MCP工具...")
    mcp_client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": [mcp_math_path],
        },
    })
    tools = await mcp_client.get_tools()
    
    print("🤖 创建代理...")
    agent = TurtlesimAgent(tools=tools)
    
    print("🧮 测试流式输出: 计算1+1并介绍自己")
    print("=" * 50)
    
    try:
        # 使用底层的agent.astream直接访问
        async for token, metadata in agent.agent.astream(
            {"messages": [{"role": "user", "content": "计算1+1，然后介绍一下你自己"}]},
            stream_mode="messages",
        ):
            # 实时打印每个token
            if hasattr(token, 'content_blocks') and token.content_blocks:
                for block in token.content_blocks:
                    if block.get("type") == "text" and block.get("text"):
                        print(block["text"], end="", flush=True)
                    elif block.get("type") == "tool_call_chunk":
                        if block.get("name"):
                            print(f"\n[调用工具: {block['name']}]", flush=True)
            elif hasattr(token, 'text') and token.text:
                print(token.text, end="", flush=True)
        
        print("\n" + "=" * 50)
        print("✅ 流式输出测试完成!")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_streaming())
    if success:
        print("🎉 流式输出工作正常!")
    else:
        print("💥 流式输出有问题.")
