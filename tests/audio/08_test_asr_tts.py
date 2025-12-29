"""
测试实时ASR + Agent + TTS的完整流程
1. 实时获取音频流并进行ASR识别
2. 将ASR结果发送给Agent处理
3. Agent流式输出时，按标点符号分割，将完整句子发送给TTS播放
"""

import asyncio
import threading
from queue import Queue
from typing import Union
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from robota.agent.base import RobotAgent
from robota.audio.asr_tts import ASRTTS
from robota.model import DOUBAO_SEED_1_6_251015_NOTHINKING
from robota.utils.logging import default_logger as logger


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


class VoiceAgent:
    """语音Agent类，集成ASR、Agent和TTS功能"""
    
    def __init__(self, silence_timeout_ms: int = 600):
        """
        初始化语音Agent
        
        Args:
            silence_timeout_ms: ASR静音超时时间(毫秒)
        """
        self.silence_timeout_ms = silence_timeout_ms
        self.conversation_history = []
        
        # 初始化Agent
        self.agent = RobotAgent(
            model=DOUBAO_SEED_1_6_251015_NOTHINKING,
            tools=[add],
            system_prompt="""
                你是个智能助手，可以和用户聊天。
                请用自然、流畅的语言回答用户的问题。
                回答要简洁明了，适合语音播报。重要：当你调用工具后，必须完全使用工具返回的结果，不要自己计算或推断。
                如果工具返回的结果与你预期的不同，也要如实报告工具返回的结果。""",
        )
        
        # 初始化ASR和TTS
        self.asr_tts = ASRTTS()
        
        # 任务句柄
        self.tts_task = None
        self.monitor_task = None
    
    def _run_agent_stream(self, messages: list, result_queue: Queue):
        """在线程中运行agent的流式输出，将结果放入队列"""
        try:
            for token, metadata in self.agent.stream(
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
    
    async def _process_agent_output(self, result_queue: Queue):
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
                await asyncio.sleep(0.05)
                continue
            
            # 检查是否收到结束标记
            if token is None:
                if current_sentence.strip():
                    # print(f"\n📢 TTS: {current_sentence.strip()}")
                    self.asr_tts.put_tts_text(current_sentence.strip())
                break
            
            # 处理文本token
            print(token, end="", flush=True)
            current_sentence += token
            full_response += token
            
            # 检测句子分隔符
            if token in sentence_delimiters and current_sentence.strip():
                # print(f"\n📢 TTS: {current_sentence.strip()}")
                self.asr_tts.put_tts_text(current_sentence.strip())
                current_sentence = ""
        
        return full_response
    
    async def _monitor_asr_queue(self):
        """监控ASR队列，获取识别结果并发送给Agent"""
        while True:
            try:
                # 等待ASR队列事件通知
                if self.asr_tts.asr_queue_event:
                    await self.asr_tts.asr_queue_event.wait()
                    self.asr_tts.asr_queue_event.clear()
                
                # 处理队列中的结果
                if not self.asr_tts.asr_queue.empty():
                    result = self.asr_tts.asr_queue.get_nowait()
                    if result and isinstance(result, dict):
                        text = result.get("text", "").strip()
                        
                        if text:
                            print(f"\n🎤 ASR识别: {text}")
                            
                            # 添加用户消息到对话历史
                            self.conversation_history.append({"role": "user", "content": text})
                            
                            # 在线程中运行Agent流式输出
                            agent_result_queue = Queue()
                            agent_thread = threading.Thread(
                                target=self._run_agent_stream,
                                args=(self.conversation_history.copy(), agent_result_queue),
                                daemon=True
                            )
                            agent_thread.start()
                            
                            # 处理Agent输出
                            agent_response = await self._process_agent_output(agent_result_queue)
                            
                            # 等待Agent线程完成
                            agent_thread.join(timeout=30)
                            
                            # 添加助手回复到对话历史
                            if agent_response.strip():
                                self.conversation_history.append({"role": "assistant", "content": agent_response.strip()})
                                print()
                
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控ASR队列错误: {e}")
                await asyncio.sleep(0.1)
    
    async def start(self):
        """启动语音Agent：先启动Agent，然后启动TTS，最后启动ASR"""
        try:
            # 1. Agent已初始化，无需额外启动
            
            # 2. 启动TTS处理器
            self.tts_task = asyncio.create_task(self.asr_tts.start_tts_processor())
            await asyncio.sleep(0.1)
            
            # 3. 启动ASR队列监控任务
            self.monitor_task = asyncio.create_task(self._monitor_asr_queue())
            
            # 4. 启动实时ASR识别
            await self.asr_tts.start_realtime_asr(silence_timeout_ms=self.silence_timeout_ms)
            
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止...")
        except Exception as e:
            logger.error(f"启动失败: {e}")
            raise
    
    async def stop(self):
        """停止语音Agent并清理资源"""
        # 停止TTS处理器
        if self.asr_tts:
            self.asr_tts.stop_tts_processor()
        
        # 取消所有任务
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        if self.tts_task:
            self.tts_task.cancel()
            try:
                await self.tts_task
            except asyncio.CancelledError:
                pass
        
        # 清理资源
        if self.asr_tts:
            self.asr_tts.audio_device.cleanup()
        
        logger.info("测试结束，资源已清理")


async def main():
    """测试程序主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="实时ASR + Agent + TTS测试")
    parser.add_argument("--silence-timeout", type=int, default=600,
                       help="ASR静音超时时间(毫秒)，默认: 600")
    
    args = parser.parse_args()
    
    # 创建并启动语音Agent
    voice_agent = VoiceAgent(silence_timeout_ms=args.silence_timeout)
    
    try:
        await voice_agent.start()
    finally:
        await voice_agent.stop()


if __name__ == "__main__":
    asyncio.run(main())

