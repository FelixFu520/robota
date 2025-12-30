"""
Test script for Turtlesim Agent Voice Mode.

This script tests the voice interaction mode of the agent.
Run this to test the agent's voice interaction capabilities.

语音交互模式:
- 通过语音输入与 Agent 交互,Agent 会语音回复
- 使用队列机制进行异步通信,支持实时流式处理
- ASR识别结果自动发送给Agent,Agent回复自动转换为语音播放
"""
import asyncio
import argparse
# 在导入其他模块之前先导入 logging，确保日志配置生效
from robota.utils import logging  # noqa: F401
from robota.utils.logging import default_logger as logger
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool, StructuredTool

from robota.agent.turtlesim_voice import TurtlesimAgentVoice
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
    """获取MCP工具并转换为同步工具"""
    mcp_client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": [mcp_math_path],
        },
        "mcp-ros-server":{
            "transport": "streamable_http",
            "url": "http://127.0.0.1:9000/mcp",
        }
    })
    async_tools = await mcp_client.get_tools()
    # 将所有异步工具转换为同步工具
    sync_tools = [_wrap_async_tool_as_sync(tool) for tool in async_tools]
    return sync_tools

async def test_voice_mode(agent: TurtlesimAgentVoice):
    """
    测试语音交互模式.
    
    使用新的 start() 方法启动完整的语音交互流程:
    - ASR自动识别语音并放入队列
    - Agent自动处理队列中的文本并生成回复
    - TTS自动将Agent回复转换为语音并播放
    
    Args:
        agent: TurtlesimAgentVoice 实例
    """
    print("\n" + "="*60)
    print("🎤 Turtlesim Agent (语音模式) is ready!")
    print("="*60)
    print("\n语音交互模式已启动,使用静音检测模式")
    print(f"静音超时时间: {agent.silence_timeout_ms}ms")
    print("按 Ctrl+C 退出\n")
    
    try:
        # 启动语音Agent,会自动处理整个语音交互流程
        await agent.start()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()

async def test_agent():
    """
    主测试函数.
    """
    parser = argparse.ArgumentParser(description='Turtlesim Agent 语音测试工具')
    parser.add_argument(
        '--silence-timeout', 
        type=int,
        default=600,
        help='ASR静音超时时间(毫秒),默认: 600ms'
    )
    parser.add_argument(
        '--no-aec', 
        action='store_true',
        help='禁用回声消除'
    )
    
    args = parser.parse_args()
    
    # 获取工具
    tools = await _get_tools()
    
    # 创建 Agent
    agent = TurtlesimAgentVoice(
        tools=tools,
        silence_timeout_ms=args.silence_timeout,
        enable_aec=not args.no_aec
    )
    
    try:
        # 启动语音交互测试
        await test_voice_mode(agent)
    finally:
        # 清理资源 (stop() 方法会在 start() 的 finally 中自动调用,这里确保资源清理)
        await agent.stop()
        logger.info("测试完成,资源已清理")


if __name__ == "__main__":
    asyncio.run(test_agent())
    """
    使用示例:
    
    # 使用默认设置启动语音交互
    python tests/agent/03_test_turtlesim_voice.py
    """
