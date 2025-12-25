"""
Test script for Turtlesim Agent.

This script tests the agent locally without requiring langgraph dev.
Run this to test the agent before deploying with langgraph dev.

支持文本交互和语音交互两种模式:
- 文本模式: 通过命令行输入文本与 Agent 交互
- 语音模式: 通过语音输入与 Agent 交互,Agent 会语音回复
"""
import asyncio
import os
import time
import argparse
from typing import Optional
# 在导入其他模块之前先导入 logging，确保日志配置生效
from robota.utils import logging  # noqa: F401
from robota.utils.logging import default_logger as logger
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

async def test_text_mode(agent: TurtlesimAgent):
    """
    测试文本交互模式.
    
    Args:
        agent: TurtlesimAgent 实例
    """
    print("\n" + "="*60)
    print("🤖 Turtlesim Agent (文本模式) is ready!")
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
            import traceback
            traceback.print_exc()

async def test_voice_mode(agent: TurtlesimAgent, duration: Optional[int] = None):
    """
    测试语音交互模式.
    
    Args:
        agent: TurtlesimAgent 实例
        duration: 每次录音的时长(秒),None 表示使用静音检测模式(检测到0.8秒静音后自动结束)
    """
    print("\n" + "="*60)
    print("🎤 Turtlesim Agent (语音模式) is ready!")
    print("="*60)
    if duration is None:
        print(f"\n语音交互模式已启动,使用静音检测模式(检测到0.8秒静音后自动结束)")
    else:
        print(f"\n语音交互模式已启动,每次录音时长: {duration} 秒")
    print("按 Ctrl+C 退出\n")
    
    # 启动音频流
    agent.start_audio_streams()
    
    try:
        interaction_count = 0
        while True:
            try:
                interaction_count += 1
                print("\n" + "="*60)
                print(f"📝 交互 #{interaction_count}")
                print("="*60)
                
                # 完整的语音交互流程（实时打印 ASR 和 Agent 输出）
                response = await agent.voice_interact(
                    duration_seconds=duration,
                    print_realtime=True
                )
                
                if response:
                    print("\n" + "="*60)
                    print("✅ 本次交互完成")
                    print("="*60)
                else:
                    print("\n" + "="*60)
                    print("⚠️  未识别到有效语音或生成回复失败")
                    print("="*60)
                
                print("\n等待下一次交互... (按 Ctrl+C 退出)")
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)
    finally:
        agent.stop_audio_streams()

async def test_asr_only(agent: TurtlesimAgent, duration: int = 10):
    """
    仅测试 ASR 功能.
    
    Args:
        agent: TurtlesimAgent 实例
        duration: 录音时长(秒)
    """
    print("\n" + "="*60)
    print("🎤 ASR 测试模式")
    print("="*60)
    print(f"\n开始录音,时长: {duration} 秒...")
    
    agent.start_audio_streams()
    
    try:
        text = await agent.listen_and_transcribe(duration_seconds=duration)
        if text:
            print(f"\n✅ 识别结果: {text}")
        else:
            print("\n⚠️  未识别到语音")
    finally:
        agent.stop_audio_streams()

async def test_tts_only(agent: TurtlesimAgent, text: str):
    """
    仅测试 TTS 功能.
    
    Args:
        agent: TurtlesimAgent 实例
        text: 要转换的文本
    """
    print("\n" + "="*60)
    print("🔊 TTS 测试模式")
    print("="*60)
    print(f"\n正在将文本转换为语音: {text}")
    
    agent.start_audio_streams()
    
    try:
        success = await agent.speak_and_play(text)
        if success:
            print("\n✅ 语音播放完成")
        else:
            print("\n❌ 语音播放失败")
    finally:
        # 等待播放完成
        await asyncio.sleep(2)
        agent.stop_audio_streams()

async def test_agent():
    """
    主测试函数,根据命令行参数选择测试模式.
    """
    parser = argparse.ArgumentParser(description='Turtlesim Agent 测试工具')
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['text', 'voice', 'asr', 'tts'], 
        default='text',
        help='测试模式: text=文本交互, voice=语音交互, asr=仅测试ASR, tts=仅测试TTS (默认: text)'
    )
    def duration_type(value):
        """将字符串转换为int或None"""
        if value.lower() == 'none' or value == '':
            return None
        return int(value)
    
    parser.add_argument(
        '--duration', 
        type=duration_type,
        default=None,
        help='录音时长(秒),用于voice和asr模式,None或"none"表示使用静音检测模式(检测到0.8秒静音后自动结束) (默认: None)'
    )
    parser.add_argument(
        '--tts-text', 
        type=str, 
        default='你好，我是地瓜君',
        help='TTS测试文本,用于tts模式 (默认: 你好，我是地瓜君)'
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
    agent = TurtlesimAgent(
        tools=tools,
        enable_aec=not args.no_aec
    )
    
    try:
        # 根据模式执行相应的测试
        if args.mode == 'text':
            await test_text_mode(agent)
        elif args.mode == 'voice':
            await test_voice_mode(agent, duration=args.duration)
        elif args.mode == 'asr':
            await test_asr_only(agent, duration=args.duration)
        elif args.mode == 'tts':
            await test_tts_only(agent, text=args.tts_text)
    finally:
        # 清理资源
        agent.cleanup()
        logger.info("测试完成,资源已清理")


if __name__ == "__main__":
    asyncio.run(test_agent())
    """
    # 文本交互模式（默认）
    python tests/agent/turtlesim.py --mode text

    # 语音交互模式
    python tests/agent/turtlesim.py --mode voice --duration 1000

    # 仅测试 ASR
    python tests/agent/turtlesim.py --mode asr --duration 30

    # 仅测试 TTS
    python tests/agent/turtlesim.py --mode tts --tts-text "你好，我是地瓜君"

    # 禁用回声消除
    python tests/agent/turtlesim.py --mode voice --no-aec
    """
