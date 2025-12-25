#!/usr/bin/env python3
"""
测试直接使用模型的流式输出（不通过代理）
"""
import asyncio
import time
from langchain_mcp_adapters.client import MultiServerMCPClient
from robota.mcp.math import mcp_math_path
from robota.model import DOUBAO_SEED_1_6_251015_NOTHINKING

async def test_direct_streaming():
    """测试直接使用模型的流式输出"""
    print("🔧 初始化MCP工具...")
    mcp_client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": [mcp_math_path],
        },
    })
    tools = await mcp_client.get_tools()
    
    print("🤖 创建模型...")
    model = DOUBAO_SEED_1_6_251015_NOTHINKING
    
    print("\n" + "="*60)
    print("🧪 测试直接模型流式输出（不使用工具）")
    print("="*60)
    
    print("请求: 详细介绍一下你自己")
    print("-" * 40)
    
    start_time = time.time()
    token_count = 0
    
    try:
        # # 直接使用模型，不绑定工具
        # for chunk in model.stream("详细介绍一下你自己，包括你的功能和特点"):
        #     current_time = time.time()
        #     elapsed = current_time - start_time
        #     token_count += 1
            
        #     if chunk.content:
        #         print(f"[{elapsed:.3f}s] {chunk.content}", end="", flush=True)
        
        # print(f"\n\n✅ 总共收到 {token_count} 个tokens")
        
        # 测试绑定工具后的流式输出
        print("\n" + "="*60)
        print("🧪 测试绑定工具后的流式输出")
        print("="*60)
        
        print("请求: 详细介绍一下你自己")
        print("-" * 40)
        
        model_with_tools = model.bind_tools(tools)
        
        start_time = time.time()
        token_count = 0
        
        for chunk in model_with_tools.stream("介绍一下你自己"):
            current_time = time.time()
            elapsed = current_time - start_time
            token_count += 1
            
            if chunk.content:
                print(f"[{elapsed:.3f}s] {chunk.content}", end="", flush=True)
        print("-"*40)
        print(tools)
        for chunk in model_with_tools.stream("你都有那些工具，可以计算1+1等于多少"):
            current_time = time.time()
            elapsed = current_time - start_time
            token_count += 1
            
            if chunk.content:
                print(f"[{elapsed:.3f}s] {chunk.content}", end="", flush=True)
        print("-"*40)
        print(f"\n\n✅ 总共收到 {token_count} 个tokens")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_direct_streaming())
    if success:
        print("\n🎉 直接模型流式输出测试完成!")
    else:
        print("\n💥 直接模型流式输出有问题.")
