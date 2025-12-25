#!/usr/bin/env python3
"""
测试修复后的流式输出
"""
import asyncio
import time
from langchain_mcp_adapters.client import MultiServerMCPClient
from robota.agent.turtlesim import TurtlesimAgent
from robota.mcp.math import mcp_math_path

async def test_streaming_fix():
    """测试修复后的流式输出"""
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
    
    print("\n" + "="*60)
    print("🧪 测试修复后的流式输出")
    print("="*60)
    
    print("请求: 介绍一下你自己")
    print("-" * 40)
    
    start_time = time.time()
    token_count = 0
    
    try:
        # 使用修复后的流式输出
        async for token, metadata in agent.agent.astream(
            {"messages": [{"role": "user", "content": "介绍一下你自己"}]},
            stream_mode="messages",
        ):
            current_time = time.time()
            elapsed = current_time - start_time
            token_count += 1
            
            # 实时打印每个token
            if hasattr(token, 'content_blocks') and token.content_blocks:
                for block in token.content_blocks:
                    if block.get("type") == "text" and block.get("text"):
                        text = block["text"]
                        print(f"[{elapsed:.3f}s] {text}", end="", flush=True)
            elif hasattr(token, 'text') and token.text:
                text = token.text
                print(f"[{elapsed:.3f}s] {text}", end="", flush=True)
            elif hasattr(token, 'content') and token.content:
                text = token.content
                print(f"[{elapsed:.3f}s] {text}", end="", flush=True)
        
        print(f"\n\n✅ 总共收到 {token_count} 个tokens")
        
        # 测试工具调用的流式输出
        print("\n" + "="*60)
        print("🧪 测试工具调用的流式输出")
        print("="*60)
        
        print("请求: 计算1+1")
        print("-" * 40)
        
        start_time = time.time()
        token_count = 0
        
        async for token, metadata in agent.agent.astream(
            {"messages": [{"role": "user", "content": "计算1+1"}]},
            stream_mode="messages",
        ):
            current_time = time.time()
            elapsed = current_time - start_time
            token_count += 1
            
            # 实时打印每个token
            if hasattr(token, 'content_blocks') and token.content_blocks:
                for block in token.content_blocks:
                    if block.get("type") == "text" and block.get("text"):
                        text = block["text"]
                        print(f"[{elapsed:.3f}s] {text}", end="", flush=True)
                    elif block.get("type") == "tool_call_chunk":
                        if block.get("name"):
                            print(f"\n[{elapsed:.3f}s] [调用工具: {block['name']}]", flush=True)
            elif hasattr(token, 'text') and token.text:
                text = token.text
                print(f"[{elapsed:.3f}s] {text}", end="", flush=True)
            elif hasattr(token, 'content') and token.content:
                text = token.content
                print(f"[{elapsed:.3f}s] {text}", end="", flush=True)
        
        print(f"\n\n✅ 总共收到 {token_count} 个tokens")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_streaming_fix())
    if success:
        print("\n🎉 流式输出修复成功!")
    else:
        print("\n💥 流式输出仍有问题.")
