"""
测试实时ASR + Agent + TTS的完整流程
1. 实时获取音频流并进行ASR识别
2. 将ASR结果发送给Agent处理
3. Agent流式输出时，按标点符号分割，将完整句子发送给TTS播放
"""

import os
import asyncio
import threading
from queue import Queue
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from robota.audio.asr_tts import ASRTTS
from robota.utils.logging import default_logger as logger
from typing import Union
from langchain.tools import tool


# 创建Agent模型
model = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    streaming=True,
    extra_body={"thinking": {"type": "disabled"}},
)

@tool
def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """计算两数之和"""
    return 1234567890


def is_tool_message(token):
    """检查token是否是工具返回的消息"""
    return (
        isinstance(token, ToolMessage) or
        (hasattr(token, "tool_call_id") and token.tool_call_id) or
        (hasattr(token, "__class__") and "ToolMessage" in token.__class__.__name__)
    )


agent = create_agent(
    model=model,
    tools=[add],  # 可以根据需要添加工具
    system_prompt="""
        你是个智能助手，可以和用户聊天。
        请用自然、流畅的语言回答用户的问题。
        回答要简洁明了，适合语音播报。重要：当你调用工具后，必须完全使用工具返回的结果，不要自己计算或推断。
                如果工具返回的结果与你预期的不同，也要如实报告工具返回的结果。""",
)


def run_agent_stream(messages: list, result_queue: Queue):
    """在线程中运行agent的流式输出，将结果放入队列"""
    try:
        for token, metadata in agent.stream(
            {"messages": messages},
            stream_mode="messages",
        ):
            if is_tool_message(token):
                continue
            
            content = None
            if hasattr(token, "text") and token.text:
                content = token.text
            elif hasattr(token, "content") and token.content:
                content = token.content
            
            if content:
                result_queue.put(content)
        
        result_queue.put(None)
    except Exception as e:
        logger.error(f"Agent流式输出错误: {e}")
        result_queue.put(None)


async def process_agent_output(result_queue: Queue, asr_tts: ASRTTS):
    """处理Agent的输出，按标点符号分割并发送给TTS"""
    sentence_delimiters = set(['。', '.', '，', ',', '？', '?', '！', '!', '；', ';', '\n'])
    current_sentence = ""
    full_response = ""
    
    print("🤖 Agent: ", end="", flush=True)
    
    while True:
        # 从队列获取token
        try:
            token = result_queue.get_nowait()
        except Exception:
            # 队列为空，等待一小段时间
            await asyncio.sleep(0.05)
            continue
        
        # 检查是否收到结束标记（None值表示结束）
        if token is None:
            # 处理剩余的文本
            if current_sentence.strip():
                print(f"\n📢 TTS: {current_sentence.strip()}")
                asr_tts.put_tts_text(current_sentence.strip())
            break
        
        # 处理文本token
        print(token, end="", flush=True)
        current_sentence += token
        full_response += token
        
        # 检测句子分隔符
        if token in sentence_delimiters and current_sentence.strip():
            print(f"\n📢 TTS: {current_sentence.strip()}")
            asr_tts.put_tts_text(current_sentence.strip())
            current_sentence = ""
    
    return full_response


async def monitor_asr_queue(asr_tts: ASRTTS, conversation_history: list):
    """监控ASR队列，获取识别结果并发送给Agent"""
    while True:
        try:
            # 等待ASR队列事件通知
            if asr_tts.asr_queue_event:
                await asr_tts.asr_queue_event.wait()
                asr_tts.asr_queue_event.clear()
            
            # 处理队列中的结果
            if not asr_tts.asr_queue.empty():
                result = asr_tts.asr_queue.get_nowait()
                if result and isinstance(result, dict):
                    text = result.get("text", "").strip()
                    
                    if text:
                        print(f"\n🎤 ASR识别: {text}")
                        
                        # 添加用户消息到对话历史
                        conversation_history.append({"role": "user", "content": text})
                        
                        # 在线程中运行Agent流式输出
                        agent_result_queue = Queue()
                        agent_thread = threading.Thread(
                            target=run_agent_stream,
                            args=(conversation_history.copy(), agent_result_queue),
                            daemon=True
                        )
                        agent_thread.start()
                        
                        # 处理Agent输出
                        agent_response = await process_agent_output(agent_result_queue, asr_tts)
                        
                        # 等待Agent线程完成
                        agent_thread.join(timeout=30)
                        
                        # 添加助手回复到对话历史
                        if agent_response.strip():
                            conversation_history.append({"role": "assistant", "content": agent_response.strip()})
                            print()
            
            await asyncio.sleep(0.1)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"监控ASR队列错误: {e}")
            await asyncio.sleep(0.1)


async def main():
    """测试程序主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="实时ASR + Agent + TTS测试")
    parser.add_argument("--silence-timeout", type=int, default=600,
                       help="ASR静音超时时间(毫秒)，默认: 600")
    
    args = parser.parse_args()
    
    # 创建ASRTTS实例
    asr_tts = ASRTTS()
    conversation_history = []
    
    tts_task = None
    monitor_task = None
    
    try:
        # 启动TTS处理器
        tts_task = asyncio.create_task(asr_tts.start_tts_processor())
        await asyncio.sleep(0.1)
        
        # 启动ASR队列监控任务
        monitor_task = asyncio.create_task(monitor_asr_queue(asr_tts, conversation_history))
        
        # 启动实时ASR识别
        await asr_tts.start_realtime_asr(silence_timeout_ms=args.silence_timeout)
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise
    finally:
        # 停止TTS处理器
        if asr_tts:
            asr_tts.stop_tts_processor()
        
        # 取消所有任务
        if monitor_task:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        if tts_task:
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass
        
        # 清理资源
        asr_tts.audio_device.cleanup()
        logger.info("测试结束，资源已清理")


if __name__ == "__main__":
    asyncio.run(main())

