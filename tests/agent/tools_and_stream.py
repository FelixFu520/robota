#!/usr/bin/env python3
"""
测试 Agent 的流式输出和工具调用

本模块提供了三个测试用例，用于验证 Agent 系统的流式输出功能：

1. test_model_streaming(): 
   测试底层模型本身的流式输出能力，不经过 Agent 层

2. test_agent_streaming(): 
   测试 Agent 在处理简单请求时的流式输出（不使用工具）

3. test_agent_with_tools(): 
   测试 Agent 在调用工具时的完整流程，包括工具调用和流式响应

这些测试可以帮助诊断流式输出功能是否正常工作，以及各个层级的实现是否正确。
"""
import asyncio
import time
from langchain_mcp_adapters.client import MultiServerMCPClient
from robota.mcp.math import mcp_math_path
from robota.agent.turtlesim import TurtlesimAgent
from robota.model import DOUBAO_SEED_1_6_251015_NOTHINKING


async def _get_tools():
    """
    获取 MCP (Model Context Protocol) 工具
    
    通过 MultiServerMCPClient 连接到数学计算工具服务器，
    并返回所有可用的工具列表。
    
    Returns:
        list: MCP 工具列表，每个工具包含名称、描述、参数等信息
    """
    print("🔧 初始化 MCP 工具...")
    # 创建多服务器 MCP 客户端，配置数学工具服务器
    # transport="stdio" 表示使用标准输入输出进行通信
    mcp_client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",  # 通信方式：标准输入输出
            "command": "python",   # 启动命令
            "args": [mcp_math_path],  # 工具服务器脚本路径
        },
    })
    # 异步获取所有可用工具
    tools = await mcp_client.get_tools()
    print(f"✅ 成功加载 {len(tools)} 个工具")
    return tools


async def test_model_streaming():
    """
    测试1: 验证模型本身是否支持流式输出
    
    直接测试底层模型的流式输出能力，不经过 Agent 层。
    通过统计接收到的 chunk 数量来判断是否真正实现了流式输出。
    
    Returns:
        bool: 如果收到超过1个chunk，返回True，表示支持流式输出
    """
    print("\n" + "="*60)
    print("🔍 测试1: 验证模型流式输出能力")
    print("="*60)
    
    # 初始化模型实例
    model = DOUBAO_SEED_1_6_251015_NOTHINKING
    # 检查模型的流式配置属性
    print(f"模型配置: streaming={getattr(model, 'streaming', 'unknown')}")
    
    print("\n介绍你自己")    
    print("-" * 60)
    
    # 记录测试开始时间和统计信息
    start_time = time.time()
    chunk_count = 0  # 接收到的数据块计数
    first_chunk_time = None  # 首个chunk到达的时间戳
    
    # 直接使用模型的异步流式接口，逐块接收响应
    async for chunk in model.astream("介绍你自己"):
        chunk_count += 1
        current_time = time.time()
        
        # 记录首个chunk的到达时间，用于计算首token延迟
        if first_chunk_time is None:
            first_chunk_time = current_time
            print(f"\n⏱️ 首个token到达时间: {first_chunk_time - start_time:.3f}秒\n")
        
        # 打印chunk内容（如果存在）
        # end="" 表示不换行，flush=True 立即输出到终端
        if hasattr(chunk, 'content') and chunk.content:
            print(chunk.content, end="", flush=True)
            time.sleep(0.01)  # 添加微小延迟来观察流式效果
    
    # 计算并输出测试统计信息
    total_time = time.time() - start_time
    print(f"\n\n✅ 模型流式输出测试完成!")
    print(f"   - 总耗时: {total_time:.3f}秒")
    print(f"   - 收到chunks: {chunk_count}个")
    print(f"   - 平均间隔: {total_time/max(chunk_count, 1):.3f}秒/chunk")
    
    # 只要收到超过1个chunk即可认定在流式输出
    return chunk_count > 1


async def test_agent_streaming():
    """
    测试2: Agent流式输出（不使用工具）
    
    测试 Agent 在处理不需要工具调用的请求时的流式输出能力。
    验证 Agent 层是否正确传递和转发模型的流式响应。
    """
    print("\n" + "="*60)
    print("🤖 测试2: Agent流式输出 (不使用工具)")
    print("="*60)
    
    # 获取工具列表并初始化 Agent
    tools = await _get_tools()
    agent = TurtlesimAgent(tools=tools)
    
    print("\n请求: 简单介绍你自己")
    print("-" * 60)
    
    # 记录测试开始时间和统计信息
    start_time = time.time()
    token_count = 0  # 接收到的token计数
    first_token_time = None  # 首个token到达的时间戳
    
    # 使用 Agent 的异步流式方法，获取事件流
    # astream_with_tools 会返回不同类型的事件：token、tool_call_start、tool_call_end、done等
    async for event in agent.astream_with_tools("介绍你自己，详细点, 说下你有那些工具"):
        # 打印分隔符，用于观察事件流
        print("|", end="|", flush=True)
        event_type = event.get("type")  # 获取事件类型
        current_time = time.time()
        
        # 处理token事件：模型生成的文本内容
        if event_type == "token":
            token_count += 1
            
            # 记录首个token的到达时间
            if first_token_time is None:
                first_token_time = current_time
                print(f"\n⏱️ 首个token到达时间: {first_token_time - start_time:.3f}秒\n")
            
            # 实时打印token内容
            print(event["content"], end="", flush=True)
            time.sleep(0.01)  # 微小延迟，便于观察流式效果
        
        # 收到完成事件，退出循环
        elif event_type == "done":
            break
    
    # 输出测试统计信息
    total_time = time.time() - start_time
    print(f"\n\n✅ Agent流式测试完成!")
    print(f"   - 总耗时: {total_time:.3f}秒")
    print(f"   - 收到tokens: {token_count}个")


async def test_agent_with_tools():
    """
    测试3: Agent调用工具时的流式输出
    
    测试 Agent 在需要调用工具（如数学计算）时的完整流程：
    1. 接收用户请求
    2. 决定调用工具
    3. 执行工具调用
    4. 基于工具结果生成流式响应
    
    验证工具调用和流式输出的混合场景是否正常工作。
    """
    print("\n" + "="*60)
    print("🛠️ 测试3: Agent调用工具流式输出")
    print("="*60)
    
    # 获取工具列表并初始化 Agent
    tools = await _get_tools()
    agent = TurtlesimAgent(tools=tools)
    
    print("\n请求: 计算1+1等于多少")
    print("-" * 60)
    
    # 记录测试开始时间和统计信息
    start_time = time.time()
    token_count = 0  # 接收到的token计数
    first_token_time = None  # 首个token到达的时间戳
    
    # 使用 Agent 的异步流式方法，处理包含工具调用的请求
    async for event in agent.astream_with_tools("计算1+1等于多少, 2+2等于多少， 4+4等于多少， 4-2等于多少"):
        # 打印分隔符，用于观察事件流
        print("|", end="|", flush=True)

        event_type = event.get("type")  # 获取事件类型
        current_time = time.time()
        elapsed = current_time - start_time  # 从开始到当前经过的时间
        
        # 处理token事件：模型生成的文本内容
        if event_type == "token":
            token_count += 1
            
            # 记录首个token的到达时间
            if first_token_time is None:
                first_token_time = current_time
                print(f"\n⏱️ 首个token到达时间: {elapsed:.3f}秒\n")
            
            # 实时打印token内容
            print(event["content"], end="", flush=True)
            time.sleep(0.01)  # 微小延迟，便于观察流式效果
        
        # 处理工具调用开始事件
        elif event_type == "tool_call_start":
            print(f"\n\n🔧 [{elapsed:.3f}s] 调用工具: {event['tool_name']}", flush=True)
            print(f"   参数: {event['tool_args']}", flush=True)
        
        # 处理工具调用结束事件，显示工具返回结果
        elif event_type == "tool_call_end":
            print(f"\n✅ 工具返回: {event['tool_result']}\n", flush=True)
        
        # 处理错误事件
        elif event_type == "error":
            print(f"\n❌ 错误: {event['error']}", flush=True)
        
        # 收到完成事件，退出循环
        elif event_type == "done":
            break
    
    # 输出测试统计信息
    total_time = time.time() - start_time
    print(f"\n✅ 工具调用测试完成! 总耗时: {total_time:.3f}秒, Tokens: {token_count}个")


async def main():
    """
    主测试函数
    
    按顺序执行所有测试用例：
    1. 测试模型本身的流式输出能力（当前已注释）
    2. 测试 Agent 的流式输出（不使用工具）
    3. 测试 Agent 调用工具时的流式输出
    
    通过这三个测试可以全面诊断流式输出功能是否正常工作。
    """
    print("="*60)
    print("🚀 开始流式输出诊断测试")
    print("="*60)
    
    # 测试1: 验证模型流式能力（当前已注释，可取消注释以启用）
    # 此测试直接验证底层模型的流式输出，不经过 Agent 层
    is_streaming = await test_model_streaming()
    
    if is_streaming:
        print("\n✅ 模型支持流式输出")
    else:
        print("\n⚠️ 模型可能没有返回多chunk（请确认 streaming=True 配置）")
    
    # 测试2: Agent流式输出（不使用工具）
    # 验证 Agent 层是否正确传递模型的流式响应
    await test_agent_streaming()
    
    # 测试3: Agent工具调用
    # 验证 Agent 在调用工具时仍能保持流式输出
    await test_agent_with_tools()
    
    print("\n" + "="*60)
    print("🎉 所有测试完成!")
    print("="*60)


if __name__ == "__main__":
    # 运行主测试函数
    # 使用 asyncio.run() 来执行异步主函数
    asyncio.run(main())
