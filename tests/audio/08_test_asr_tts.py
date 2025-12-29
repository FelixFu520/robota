"""
测试实时ASR + Agent + TTS的完整流程
1. 实时获取音频流并进行ASR识别
2. 将ASR结果发送给Agent处理
3. Agent流式输出时，按标点符号分割，将完整句子发送给TTS播放
"""

import os
import re
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
    """
    计算两数之和
    
    Args:
        a: 第一个数
        b: 第二个数

    Returns:
        两数之和

    Examples:
        >>> add(1, 2)
        3
        >>> add(-1, 1)
        0
        >>> add(0, 0)
        0
    """
    return 1234567890


def is_tool_message(token):
    """
    检查 token 是否是工具返回的消息（ToolMessage）
    
    Args:
        token: 从 agent.stream 返回的 token 对象
        
    Returns:
        bool: 如果是 ToolMessage 返回 True，否则返回 False
    """
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
    """
    在线程中运行agent的流式输出，将结果放入队列
    
    Args:
        messages: 对话历史消息列表
        result_queue: 用于传递agent输出token的队列
    """
    try:
        for token, metadata in agent.stream(
            {"messages": messages},
            stream_mode="messages",
        ):
            # 过滤掉工具返回的消息（ToolMessage），只处理 LLM 生成的文本消息
            if is_tool_message(token):
                continue
            
            # 从token中提取文本内容
            content = None
            if hasattr(token, "text") and token.text:
                content = token.text
            elif hasattr(token, "content") and token.content:
                content = token.content
            
            # 只将文本内容放入队列
            if content:
                result_queue.put(content)
        
        # 发送结束标记
        result_queue.put(None)
    except Exception as e:
        logger.error(f"Agent流式输出错误: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(None)


async def process_agent_output(result_queue: Queue, asr_tts: ASRTTS):
    """
    处理Agent的输出，按标点符号分割并发送给TTS
    
    Args:
        result_queue: Agent输出token队列
        asr_tts: ASRTTS实例
    
    Returns:
        完整的Agent回复文本
    """
    # 句子分隔符：中文和英文的句号、逗号、问号、感叹号、分号
    sentence_delimiters = set(['。', '.', '，', ',', '？', '?', '！', '!', '；', ';', '\n'])
    
    current_sentence = ""  # 当前正在收集的句子
    full_response = ""  # 完整的回复文本
    
    print("🤖 Agent: ", end="", flush=True)
    
    finished = False
    empty_count = 0  # 连续空队列计数，用于检测是否真的结束
    
    while not finished:
        # 从队列中获取token
        token = None
        if not result_queue.empty():
            try:
                token = result_queue.get_nowait()
                empty_count = 0  # 重置空计数
            except Exception:
                pass
        
        if token is not None:
            # 检查是否收到结束标记
            if token is None:
                # 处理剩余的文本
                if current_sentence.strip():
                    print(f"\n📢 TTS: {current_sentence.strip()}")
                    asr_tts.put_tts_text(current_sentence.strip())
                    full_response += current_sentence
                finished = True
                break
            
            # 打印token
            print(token, end="", flush=True)
            
            # 累积到当前句子和完整回复
            current_sentence += token
            full_response += token
            
            # 检测句子分隔符
            if token in sentence_delimiters:
                # 将当前句子发送给TTS
                if current_sentence.strip():
                    print(f"\n📢 TTS: {current_sentence.strip()}")
                    asr_tts.put_tts_text(current_sentence.strip())
                    current_sentence = ""  # 重置当前句子
        else:
            # 队列为空，等待一小段时间
            empty_count += 1
            # 如果连续多次（如100次，约5秒）都是空的，可能是Agent线程已经结束但没有发送None
            if empty_count > 100:
                logger.warning("Agent输出队列长时间为空，可能已结束")
                # 处理剩余的文本
                if current_sentence.strip():
                    print(f"\n📢 TTS: {current_sentence.strip()}")
                    asr_tts.put_tts_text(current_sentence.strip())
                    full_response += current_sentence
                break
            await asyncio.sleep(0.05)
    
    return full_response


async def monitor_asr_queue(asr_tts: ASRTTS, conversation_history: list):
    """
    监控ASR队列，获取识别结果并发送给Agent
    
    Args:
        asr_tts: ASRTTS实例
        conversation_history: 对话历史消息列表
    """
    last_processed_text = ""  # 记录最近一次处理的文本，用于去重
    last_processed_time = 0  # 记录最近一次处理的时间
    
    while True:
        try:
            # 等待ASR队列事件通知
            if asr_tts.asr_queue_event:
                try:
                    await asyncio.wait_for(asr_tts.asr_queue_event.wait(), timeout=1.0)
                    asr_tts.asr_queue_event.clear()
                except asyncio.TimeoutError:
                    # 超时是正常的，继续检查队列
                    pass
            
            # 检查队列中是否有结果（处理所有待处理的结果）
            processed_any = False
            # 先收集所有待处理的文本，选择最完整的一个
            pending_texts = []
            while not asr_tts.asr_queue.empty():
                result = asr_tts.asr_queue.get_nowait()
                if result and isinstance(result, dict):
                    text = result.get("text", "").strip()
                    if text:
                        pending_texts.append(text)
            
            # 如果有待处理的文本，选择最长的（通常是最完整的）
            if pending_texts:
                # 选择最长的文本（通常是最完整的识别结果）
                text = max(pending_texts, key=len)
                
                # 去重逻辑：检查是否与最近一次处理的文本相同或相似
                current_time = asyncio.get_event_loop().time()
                time_since_last = (current_time - last_processed_time) * 1000  # 转换为毫秒
                
                # 如果文本完全相同，跳过
                if text == last_processed_text:
                    logger.debug(f"跳过重复的ASR识别: {text}")
                    continue
                
                # 如果新文本是最近一次文本的前缀（说明是部分识别），跳过
                if last_processed_text and text in last_processed_text and len(text) < len(last_processed_text):
                    logger.debug(f"跳过部分识别结果: {text} (已有更完整的: {last_processed_text})")
                    continue
                
                # 如果最近一次文本是新文本的前缀，且时间间隔较短（<10秒），说明是扩展识别，跳过
                if last_processed_text and last_processed_text in text and time_since_last < 10000:
                    logger.debug(f"跳过扩展识别结果: {text} (已有: {last_processed_text}, 间隔: {time_since_last:.0f}ms)")
                    continue
                
                # 更智能的去重：如果新文本只是旧文本加上标点符号，视为重复
                # 移除标点符号后比较
                def remove_punctuation(s):
                    """移除标点符号"""
                    return re.sub(r'[。，、；：？！.!?,;:""''（）()【】\[\]《》]', '', s).strip()
                
                text_no_punct = remove_punctuation(text)
                last_text_no_punct = remove_punctuation(last_processed_text) if last_processed_text else ""
                
                # 如果去除标点后文本完全相同，无论时间间隔多长都视为重复（很可能是同一句话）
                if last_processed_text and text_no_punct == last_text_no_punct and text_no_punct:
                    logger.info(f"跳过仅标点差异的识别结果: {text} (已有: {last_processed_text}, 间隔: {time_since_last:.0f}ms)")
                    continue
                
                # 如果新文本只是旧文本的前缀（去除标点后），且时间间隔较短（<10秒），跳过
                if last_processed_text and text_no_punct in last_text_no_punct and len(text_no_punct) < len(last_text_no_punct) and time_since_last < 10000:
                    logger.debug(f"跳过部分识别结果（去除标点后）: {text} (已有: {last_processed_text}, 间隔: {time_since_last:.0f}ms)")
                    continue
                
                # 如果最近一次文本是新文本的前缀（去除标点后），且时间间隔较短（<10秒），跳过
                if last_processed_text and last_text_no_punct in text_no_punct and len(last_text_no_punct) < len(text_no_punct) and time_since_last < 10000:
                    logger.debug(f"跳过扩展识别结果（去除标点后）: {text} (已有: {last_processed_text}, 间隔: {time_since_last:.0f}ms)")
                    continue
                
                # 处理新的识别结果
                print(f"\n🎤 ASR识别: {text}")
                
                # 更新最近处理的文本和时间
                last_processed_text = text
                last_processed_time = current_time
                
                # 将用户消息添加到对话历史
                user_message = {"role": "user", "content": text}
                conversation_history.append(user_message)
                
                # 创建队列用于Agent输出
                agent_result_queue = Queue()
                
                # 在线程中运行Agent流式输出（传入完整的对话历史）
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
                if agent_thread.is_alive():
                    logger.warning("Agent线程未在30秒内完成")
                
                # 将Agent回复添加到对话历史
                if agent_response.strip():
                    assistant_message = {"role": "assistant", "content": agent_response.strip()}
                    conversation_history.append(assistant_message)
                    print()  # 换行
                
                processed_any = True
            
            # 如果没有处理任何结果，等待一小段时间避免CPU占用过高
            if not processed_any:
                await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"监控ASR队列错误: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(0.1)


async def main():
    """
    主函数：启动ASR、TTS和Agent处理流程
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="实时ASR + Agent + TTS测试")
    parser.add_argument("--silence-timeout", type=int, default=800,
                       help="ASR静音超时时间(毫秒)，默认: 800")
    parser.add_argument("--duration", type=int, default=None,
                       help="录音时长(秒)，None表示无限录音，默认: None")
    
    args = parser.parse_args()
    
    # 创建ASRTTS实例
    logger.info("初始化ASRTTS...")
    asr_tts = ASRTTS()
    
    # 初始化对话历史
    conversation_history = []
    
    asr_task = None
    tts_task = None
    monitor_task = None
    
    try:
        # 启动TTS处理器
        logger.info("启动TTS处理器...")
        tts_task = asyncio.create_task(asr_tts.start_tts_processor())
        await asyncio.sleep(1.0)  # 等待TTS处理器初始化
        
        # 启动ASR队列监控任务（传入对话历史）
        logger.info("启动ASR队列监控...")
        monitor_task = asyncio.create_task(monitor_asr_queue(asr_tts, conversation_history))
        
        # 启动实时ASR识别
        logger.info("启动实时ASR识别...")
        logger.info("开始录音，请说话...")
        asr_task = asyncio.create_task(
            asr_tts.start_realtime_asr(
                duration_seconds=args.duration,
                silence_timeout_ms=args.silence_timeout
            )
        )
        
        # 等待ASR任务完成
        await asr_task
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
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
        
        if asr_task:
            asr_task.cancel()
            try:
                await asr_task
            except asyncio.CancelledError:
                pass
        
        # 清理资源
        if asr_tts:
            asr_tts.audio_device.cleanup()
        
        logger.info("测试结束，资源已清理")


if __name__ == "__main__":
    asyncio.run(main())

