"""
测试实时ASR + Agent + TTS的完整流程
1. 实时获取音频流并进行ASR识别
2. 将ASR结果发送给Agent处理
3. Agent流式输出时，按标点符号分割，将完整句子发送给TTS播放
"""

import asyncio
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
    """语音Agent类，集成ASR、Agent和TTS功能，通过队列进行通信"""
    
    def __init__(self, silence_timeout_ms: int = 600):
        """
        初始化语音Agent
        
        Args:
            silence_timeout_ms: ASR静音超时时间(毫秒)
        """
        self.silence_timeout_ms = silence_timeout_ms
        
        # 初始化组件
        self.agent = RobotAgent(
            model=DOUBAO_SEED_1_6_251015_NOTHINKING,
            tools=[add],
            system_prompt="""
                你是个智能助手，可以和用户聊天。
                请用自然、流畅的语言回答用户的问题。
                回答要简洁明了，适合语音播报。重要：当你调用工具后，必须完全使用工具返回的结果，不要自己计算或推断。
                如果工具返回的结果与你预期的不同，也要如实报告工具返回的结果。""",
        )
        self.asr_tts = ASRTTS()
        
        # 队列：ASR -> Agent -> TTS
        self.agent_input_queue = asyncio.Queue()  # Agent输入队列（从ASR接收）
        self.agent_output_queue = asyncio.Queue()  # Agent输出队列（发送给TTS）
        
        # 任务句柄
        self.asr_task = None
        self.asr_to_agent_task = None  # ASR到Agent的桥接任务
        self.agent_task = None  # Agent处理任务
        self.agent_to_tts_task = None  # Agent到TTS的桥接任务
        self.tts_task = None
    
    async def _run_agent_stream(self, user_text: str, output_queue: asyncio.Queue):
        """在协程中运行agent的流式输出，将结果放入队列"""
        try:
            # 只传递当前用户消息，不使用历史记录
            messages = [{"role": "user", "content": user_text}]
            
            # 在线程池中运行同步的stream迭代，保持流式处理
            loop = asyncio.get_event_loop()
            
            def run_stream():
                """在线程中运行stream迭代，将结果放入队列"""
                try:
                    for token, metadata in self.agent.stream(
                        {"messages": messages},
                        stream_mode="messages",
                    ):
                        if is_tool_message(token):
                            continue
                        
                        content = getattr(token, "text", None) or getattr(token, "content", None)
                        if content:
                            # 使用线程安全的方式将token放入异步队列
                            asyncio.run_coroutine_threadsafe(
                                output_queue.put(content),
                                loop
                            )
                    
                    # 发送结束标记
                    asyncio.run_coroutine_threadsafe(
                        output_queue.put(None),
                        loop
                    )
                except Exception as e:
                    logger.error(f"Agent流式输出错误: {e}")
                    asyncio.run_coroutine_threadsafe(
                        output_queue.put(None),
                        loop
                    )
            
            # 在线程池中运行，避免阻塞事件循环
            await loop.run_in_executor(None, run_stream)
            
        except Exception as e:
            logger.error(f"Agent流式输出错误: {e}")
            await output_queue.put(None)
    
    async def _bridge_asr_to_agent(self):
        """桥接任务：从ASR队列读取，放入Agent输入队列"""
        while True:
            try:
                # 等待ASR队列事件通知
                if self.asr_tts.asr_queue_event:
                    await self.asr_tts.asr_queue_event.wait()
                    self.asr_tts.asr_queue_event.clear()
                
                # 处理ASR队列中的所有结果
                while not self.asr_tts.asr_queue.empty():
                    result = self.asr_tts.asr_queue.get_nowait()
                    if result and isinstance(result, dict):
                        text = result.get("text", "").strip()
                        if text:
                            print(f"\n🎤 ASR识别: {text}")
                            # 将ASR结果放入Agent输入队列
                            await self.agent_input_queue.put(text)
                
                await asyncio.sleep(0.01)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ASR到Agent桥接错误: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_agent(self):
        """Agent处理任务：从输入队列读取，处理后将token放入输出队列"""
        while True:
            try:
                # 从Agent输入队列获取用户输入
                user_text = await self.agent_input_queue.get()
                
                if user_text:
                    # 创建异步队列用于协程间通信
                    token_queue = asyncio.Queue()
                    
                    # 在协程中运行Agent流式输出
                    stream_task = asyncio.create_task(
                        self._run_agent_stream(user_text, token_queue)
                    )
                    
                    # 收集Agent的完整响应，并将token转发到输出队列
                    full_response = ""
                    print("🤖 Agent: ", end="", flush=True)
                    
                    while True:
                        # 从异步队列读取token
                        token = await token_queue.get()
                        
                        if token is None:
                            # 结束标记，转发到输出队列
                            await self.agent_output_queue.put(None)
                            break
                        
                        print(token, end="", flush=True)
                        full_response += token
                        # 将token转发到输出队列（供桥接任务处理）
                        await self.agent_output_queue.put(token)
                    
                    # 等待Agent协程完成
                    await stream_task
                    print()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Agent处理错误: {e}")
                await asyncio.sleep(0.1)
    
    async def _bridge_agent_to_tts(self):
        """桥接任务：从Agent输出队列读取token，按句子分割后放入TTS队列"""
        sentence_delimiters = {'。', '.', '，', ',', '？', '?', '！', '!', '；', ';', '\n'}
        current_sentence = ""
        
        while True:
            try:
                # 从Agent输出队列获取token
                token = await self.agent_output_queue.get()
                
                if token is None:
                    # 结束标记，处理剩余的句子，然后继续等待下一个响应
                    if current_sentence.strip():
                        self.asr_tts.put_tts_text(current_sentence.strip())
                        current_sentence = ""
                    continue
                
                # 累积token到当前句子
                current_sentence += token
                
                # 检测句子分隔符，将完整句子放入TTS队列
                if token in sentence_delimiters and current_sentence.strip():
                    self.asr_tts.put_tts_text(current_sentence.strip())
                    current_sentence = ""
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Agent到TTS桥接错误: {e}")
                await asyncio.sleep(0.1)
    
    async def _start_agent(self):
        """启动Agent处理任务"""
        # logger.info("正在启动Agent...")
        self.agent_task = asyncio.create_task(self._process_agent())
        self.agent_to_tts_task = asyncio.create_task(self._bridge_agent_to_tts())
        await asyncio.sleep(0.1)  # 等待初始化
        # logger.info("Agent已启动")
    
    async def _start_tts(self):
        """启动TTS处理器"""
        # logger.info("正在启动TTS...")
        self.tts_task = asyncio.create_task(self.asr_tts.start_tts_processor())
        await asyncio.sleep(0.1)  # 等待TTS初始化
        # logger.info("TTS已启动")
    
    async def _start_asr(self):
        """启动ASR识别和桥接任务"""
        # logger.info("正在启动ASR...")
        # 启动ASR到Agent的桥接任务
        self.asr_to_agent_task = asyncio.create_task(self._bridge_asr_to_agent())
        # 启动实时ASR识别
        self.asr_task = asyncio.create_task(
            self.asr_tts.start_realtime_asr(silence_timeout_ms=self.silence_timeout_ms)
        )
        # logger.info("ASR已启动")
    
    async def start(self):
        """启动语音Agent：按顺序启动Agent、TTS、ASR"""
        try:
            await self._start_agent()
            await self._start_tts()
            await self._start_asr()
            
            # 等待ASR任务完成（主循环）
            if self.asr_task:
                await self.asr_task
            
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止...")
        except Exception as e:
            logger.error(f"启动失败: {e}")
            raise
    
    async def stop(self):
        """停止语音Agent并清理资源"""
        logger.info("正在停止语音Agent...")
        
        # 停止TTS处理器
        if self.asr_tts:
            self.asr_tts.stop_tts_processor()
        
        # 取消所有任务
        tasks = [
            self.asr_task,
            self.asr_to_agent_task,
            self.agent_task,
            self.agent_to_tts_task,
            self.tts_task,
        ]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
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

