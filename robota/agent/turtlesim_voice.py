"""
Turtlesim Agent 模块.

本模块定义了 TurtlesimAgent 类,用于控制 ROS 2 的 Turtlesim 机器人.
Turtlesim 是 ROS 2 的一个简单的可视化仿真器,用于演示基本的机器人控制功能.
"""

import asyncio
from typing import List, Optional

from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage

from robota.model import DOUBAO_SEED_1_6_251015_NOTHINKING
from robota.agent.base import RobotAgent
from robota.audio.asr_tts import ASRTTS
from robota.utils.logging import default_logger as logger


def is_tool_message(token):
    """检查token是否是工具返回的消息"""
    return (
        isinstance(token, ToolMessage) or
        (hasattr(token, "tool_call_id") and token.tool_call_id) or
        (hasattr(token, "__class__") and "ToolMessage" in token.__class__.__name__)
    )


class TurtlesimAgentVoice(RobotAgent):
    """
    Turtlesim 机器人 Agent.
    
    该类继承自 RobotAgent,专门用于控制 ROS 2 Turtlesim 仿真机器人.
    Turtlesim 是一个简单的海龟机器人仿真器,常用于 ROS 2 学习和测试.
    
    该 Agent 使用豆包模型(无思考模式)作为底层语言模型,并配置了专门用于
    Turtlesim 机器人控制的系统提示词.
    
    支持语音交互功能:
    - ASR (语音识别): 将语音转换为文本
    - TTS (文本转语音): 将文本转换为语音并播放
    - 使用队列机制进行异步通信,支持实时流式处理
    
    示例:
        >>> tools = [move_tool, rotate_tool]  # 定义控制工具
        >>> agent = TurtlesimAgentVoice(tools=tools)
        >>> # 语音交互
        >>> await agent.start()
        >>> # 在后台运行,自动处理语音输入和输出
    """
    
    def __init__(
        self, 
        tools: List[BaseTool] = None,
        silence_timeout_ms: int = 600,
        enable_aec: bool = True,
    ):
        """
        初始化 Turtlesim Agent.
        
        Args:
            tools: 可选的工具列表,用于控制 Turtlesim 机器人.
                  如果为 None,则使用空列表(Agent 将无法调用任何工具).
                  工具通常包括移动、旋转、获取位置等 ROS 2 控制命令.
            silence_timeout_ms: ASR静音超时时间(毫秒),默认600ms
            enable_aec: 是否启用回声消除
        
        Note:
            - 使用 DOUBAO_SEED_1_6_251015_NOTHINKING 模型(无思考模式),适合实时控制场景
            - 系统提示词指导 Agent 作为 ROS 2 机器人助手
            - 继承自 RobotAgent 的所有功能,包括同步/异步调用和流式输出
            - 支持语音交互功能(ASR/TTS),使用队列机制进行异步通信
        """
        super().__init__(
            model=DOUBAO_SEED_1_6_251015_NOTHINKING,
            tools=tools,
            system_prompt="你是一个有用的 ROS 2 机器人助手，可以控制 Turtlesim 机器人, 可以聊天回答问题"
        )

        # ASR/TTS 集成类
        self.asr_tts = ASRTTS()
        self.silence_timeout_ms = silence_timeout_ms
        
        # 队列：ASR -> Agent -> TTS
        self.agent_input_queue = asyncio.Queue()  # Agent输入队列（从ASR接收）
        self.agent_output_queue = asyncio.Queue()  # Agent输出队列（发送给TTS）
        
        # 任务句柄
        self.asr_task = None
        self.asr_to_agent_task = None  # ASR到Agent的桥接任务
        self.agent_task = None  # Agent处理任务
        self.agent_to_tts_task = None  # Agent到TTS的桥接任务
        self.tts_task = None
        
        # 运行状态
        self._running = False
    
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
                    for token, metadata in self.stream(
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
        while self._running:
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
        while self._running:
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
        
        while self._running:
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
        self.agent_task = asyncio.create_task(self._process_agent())
        self.agent_to_tts_task = asyncio.create_task(self._bridge_agent_to_tts())
        await asyncio.sleep(0.1)  # 等待初始化
    
    async def _start_tts(self):
        """启动TTS处理器"""
        self.tts_task = asyncio.create_task(self.asr_tts.start_tts_processor())
        await asyncio.sleep(0.1)  # 等待TTS初始化
    
    async def _start_asr(self):
        """启动ASR识别和桥接任务"""
        # 启动ASR到Agent的桥接任务
        self.asr_to_agent_task = asyncio.create_task(self._bridge_asr_to_agent())
        # 启动实时ASR识别
        self.asr_task = asyncio.create_task(
            self.asr_tts.start_realtime_asr(silence_timeout_ms=self.silence_timeout_ms)
        )
    
    async def start(self):
        """启动语音Agent：按顺序启动Agent、TTS、ASR"""
        if self._running:
            logger.warning("语音Agent已经在运行中")
            return
        
        self._running = True
        try:
            await self._start_agent()
            await self._start_tts()
            await self._start_asr()
            
            logger.info("语音Agent已启动，开始监听语音...")
            
            # 等待ASR任务完成（主循环）
            if self.asr_task:
                await self.asr_task
            
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止...")
        except Exception as e:
            logger.error(f"启动失败: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """停止语音Agent并清理资源"""
        if not self._running:
            return
        
        logger.info("正在停止语音Agent...")
        self._running = False
        
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
        
        logger.info("语音Agent已停止，资源已清理")
    
    # 保留向后兼容的方法
    async def listen_and_transcribe(
        self, 
        duration_seconds: Optional[int] = None,
        timeout: float = 30.0,
        print_realtime: bool = True,
        silence_timeout: float = 0.8
    ) -> str:
        """
        监听语音并转换为文本（向后兼容方法）.
        
        注意：此方法已过时，建议使用 start() 方法启动完整的语音交互流程。
        
        Args:
            duration_seconds: 录音时长(秒),None 表示使用静音检测模式
            timeout: 超时时间(秒)
            print_realtime: 是否实时打印识别结果
            silence_timeout: 静音超时时间(秒)
            
        Returns:
            识别到的文本,如果超时或出错则返回空字符串
        """
        logger.warning("listen_and_transcribe 方法已过时，建议使用 start() 方法")
        # 简化实现，直接使用 ASRTTS
        silence_timeout_ms = int(silence_timeout * 1000)
        try:
            # 临时启动ASR
            asr_task = asyncio.create_task(
                self.asr_tts.start_realtime_asr(
                    duration_seconds=duration_seconds,
                    silence_timeout_ms=silence_timeout_ms
                )
            )
            # 等待识别结果
            while not self.asr_tts.asr_queue.empty() or not asr_task.done():
                await asyncio.sleep(0.1)
                if not self.asr_tts.asr_queue.empty():
                    result = self.asr_tts.asr_queue.get_nowait()
                    if result and isinstance(result, dict):
                        text = result.get("text", "").strip()
                        if text:
                            asr_task.cancel()
                            return text
            return ""
        except Exception as e:
            logger.error(f"ASR识别失败: {e}")
            return ""
    
    def speak_and_play(self, text: str) -> bool:
        """
        将文本放入TTS队列进行播放（向后兼容方法）.
        
        注意：此方法已过时，建议使用 start() 方法启动完整的语音交互流程。
        如果已经启动了 start()，可以直接使用 put_tts_text() 方法。
        
        Args:
            text: 要转换的文本
            
        Returns:
            是否成功
        """
        logger.warning("speak_and_play 方法已过时，建议使用 start() 方法")
        if self._running:
            self.asr_tts.put_tts_text(text)
            return True
        else:
            logger.error("语音Agent未启动，请先调用 start() 方法")
            return False
    
