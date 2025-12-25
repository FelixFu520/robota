#!/usr/bin/env python3
"""
测试 Agent 的流式输出和工具调用
"""
import asyncio
import time
from langchain_mcp_adapters.client import MultiServerMCPClient
from robota.mcp.math import mcp_math_path
from robota.agent.turtlesim import TurtlesimAgent
from robota.model import DOUBAO_SEED_1_6_251015_NOTHINKING


async def _get_tools():
    """获取 MCP 工具"""
    print("🔧 初始化 MCP 工具...")
    mcp_client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": [mcp_math_path],
        },
    })
    tools = await mcp_client.get_tools()
    print(f"✅ 成功加载 {len(tools)} 个工具")
    return tools


async def test_model_streaming():
    """测试1: 验证模型本身是否支持流式输出"""
    print("\n" + "="*60)
    print("🔍 测试1: 验证模型流式输出能力")
    print("="*60)
    
    model = DOUBAO_SEED_1_6_251015_NOTHINKING
    print(f"模型配置: streaming={getattr(model, 'streaming', 'unknown')}")
    
    print("\n介绍你自己，10字内")    
    print("-" * 60)
    
    start_time = time.time()
    chunk_count = 0
    first_chunk_time = None
    
    # 直接使用模型流式输出
    async for chunk in model.astream("介绍你自己，10字内"):
        chunk_count += 1
        current_time = time.time()
        
        if first_chunk_time is None:
            first_chunk_time = current_time
            print(f"\n⏱️ 首个token到达时间: {first_chunk_time - start_time:.3f}秒\n")
        
        # 打印内容
        if hasattr(chunk, 'content') and chunk.content:
            print(chunk.content, end="", flush=True)
            time.sleep(0.01)  # 添加微小延迟来观察流式效果
    
    total_time = time.time() - start_time
    print(f"\n\n✅ 模型流式输出测试完成!")
    print(f"   - 总耗时: {total_time:.3f}秒")
    print(f"   - 收到chunks: {chunk_count}个")
    print(f"   - 平均间隔: {total_time/max(chunk_count, 1):.3f}秒/chunk")
    
    return chunk_count > 1  # 只要>1个chunk即可认定在流式


async def test_agent_streaming():
    """测试2: Agent流式输出"""
    print("\n" + "="*60)
    print("🤖 测试2: Agent流式输出 (不使用工具)")
    print("="*60)
    
    tools = await _get_tools()
    agent = TurtlesimAgent(tools=tools)
    
    print("\n请求: 简单介绍你自己")
    print("-" * 60)
    
    start_time = time.time()
    token_count = 0
    first_token_time = None
    
    # 使用新的 astream_with_tools 方法获取逐token流
    async for event in agent.astream_with_tools("介绍你自己，详细点, 说下你有那些工具"):
        print("|", end="|", flush=True)
        event_type = event.get("type")
        current_time = time.time()
        
        if event_type == "token":
            token_count += 1
            
            if first_token_time is None:
                first_token_time = current_time
                print(f"\n⏱️ 首个token到达时间: {first_token_time - start_time:.3f}秒\n")
            
            print(event["content"], end="", flush=True)
            time.sleep(0.01)
        
        elif event_type == "done":
            break
    
    total_time = time.time() - start_time
    print(f"\n\n✅ Agent流式测试完成!")
    print(f"   - 总耗时: {total_time:.3f}秒")
    print(f"   - 收到tokens: {token_count}个")


async def test_agent_with_tools():
    """测试3: Agent调用工具"""
    print("\n" + "="*60)
    print("🛠️ 测试3: Agent调用工具流式输出")
    print("="*60)
    
    tools = await _get_tools()
    agent = TurtlesimAgent(tools=tools)
    
    print("\n请求: 计算1+1等于多少")
    print("-" * 60)
    
    start_time = time.time()
    token_count = 0
    first_token_time = None
    
    # 使用新的 astream_with_tools 方法
    async for event in agent.astream_with_tools("计算1+1等于多少, 2+2等于多少， 4+4等于多少， 4-2等于多少"):
        print("|", end="|", flush=True)

        event_type = event.get("type")
        current_time = time.time()
        elapsed = current_time - start_time
        
        if event_type == "token":
            token_count += 1
            
            if first_token_time is None:
                first_token_time = current_time
                print(f"\n⏱️ 首个token到达时间: {elapsed:.3f}秒\n")
            
            print(event["content"], end="", flush=True)
            time.sleep(0.01)
        
        elif event_type == "tool_call_start":
            print(f"\n\n🔧 [{elapsed:.3f}s] 调用工具: {event['tool_name']}", flush=True)
            print(f"   参数: {event['tool_args']}", flush=True)
        
        elif event_type == "tool_call_end":
            print(f"\n✅ 工具返回: {event['tool_result']}\n", flush=True)
        
        elif event_type == "error":
            print(f"\n❌ 错误: {event['error']}", flush=True)
        
        elif event_type == "done":
            break
    
    total_time = time.time() - start_time
    print(f"\n✅ 工具调用测试完成! 总耗时: {total_time:.3f}秒, Tokens: {token_count}个")


async def main():
    """主测试函数"""
    print("="*60)
    print("🚀 开始流式输出诊断测试")
    print("="*60)
    
    # # 测试1: 验证模型流式能力
    # is_streaming = await test_model_streaming()
    
    # if is_streaming:
    #     print("\n✅ 模型支持流式输出")
    # else:
    #     print("\n⚠️ 模型可能没有返回多chunk（请确认 streaming=True 配置）")
    
    # 测试2: Agent流式输出
    await test_agent_streaming()
    
    # 测试3: Agent工具调用
    await test_agent_with_tools()
    
    print("\n" + "="*60)
    print("🎉 所有测试完成!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
