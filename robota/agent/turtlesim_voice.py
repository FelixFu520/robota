"""
Turtlesim Agent 模块.

本模块定义了 TurtlesimAgent 类,用于控制 ROS 2 的 Turtlesim 机器人.
Turtlesim 是 ROS 2 的一个简单的可视化仿真器,用于演示基本的机器人控制功能.
"""

import asyncio
import io
import json
import uuid
import wave
from typing import List, Optional, AsyncIterator
from pydub import AudioSegment

from langchain_core.tools import BaseTool
import websockets

from robota.model import DOUBAO_SEED_1_6_251015_NOTHINKING
from robota.agent.base import RobotAgent
from robota.audio.audio_device import AudioDevice
from robota.audio.volcengine_doubao_asr import AsrWsClient
from robota.audio.volcengine_doubao_tts import (
    EventType,
    MsgType,
    finish_connection,
    finish_session,
    receive_message,
    start_connection,
    start_session,
    task_request,
    wait_for_event,
)
from robota.utils.logging import default_logger as logger


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
    
    示例:
        >>> tools = [move_tool, rotate_tool]  # 定义控制工具
        >>> agent = TurtlesimAgent(tools=tools)
        >>> response = agent.invoke({"input": "让机器人向前移动"})
        >>> # 语音交互
        >>> await agent.start_audio_streams()
        >>> text = await agent.listen_and_transcribe()  # 语音转文本
        >>> await agent.speak_and_play("你好")  # 文本转语音并播放
    """
    
    def __init__(
        self, 
        tools: List[BaseTool] = None,
        asr_url: str = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async",
        tts_url: str = "wss://openspeech.bytedance.com/api/v3/tts/bidirection",
        tts_appid: str = "5919896644",
        tts_access_token: str = "G-o4lEbyzOv9F6cLu9jYhkrOegOjorqU",
        tts_resource_id: str = "seed-tts-2.0",
        tts_voice_type: str = "zh_male_m191_uranus_bigtts",
        enable_aec: bool = True,
    ):
        """
        初始化 Turtlesim Agent.
        
        Args:
            tools: 可选的工具列表,用于控制 Turtlesim 机器人.
                  如果为 None,则使用空列表(Agent 将无法调用任何工具).
                  工具通常包括移动、旋转、获取位置等 ROS 2 控制命令.
            asr_url: ASR WebSocket 服务器 URL
            tts_url: TTS WebSocket 服务器 URL
            tts_appid: TTS App ID
            tts_access_token: TTS Access Token
            tts_resource_id: TTS 资源 ID
            tts_voice_type: TTS 语音类型
            enable_aec: 是否启用回声消除
        
        Note:
            - 使用 DOUBAO_SEED_1_6_251015_NOTHINKING 模型(无思考模式),适合实时控制场景
            - 系统提示词指导 Agent 作为 ROS 2 机器人助手
            - 继承自 RobotAgent 的所有功能,包括同步/异步调用和流式输出
            - 支持语音交互功能(ASR/TTS)
        """
        super().__init__(
            model=DOUBAO_SEED_1_6_251015_NOTHINKING,
            tools=tools,
            system_prompt="你是一个有用的 ROS 2 机器人助手，可以控制 Turtlesim 机器人, 可以聊天回答问题"
        )

        # 音频设备
        self.audio = AudioDevice(enable_aec=enable_aec)
        
        # ASR 配置
        self.asr_url = asr_url
        self.asr_client: Optional[AsrWsClient] = None
        
        # TTS 配置
        self.tts_url = tts_url
        self.tts_appid = tts_appid
        self.tts_access_token = tts_access_token
        self.tts_resource_id = tts_resource_id
        self.tts_voice_type = tts_voice_type
        self.tts_websocket: Optional[websockets.WebSocketClientProtocol] = None
        
        # 音频流状态
        self._audio_streams_started = False
    
    def start_audio_streams(self):
        """
        启动音频输入输出流.
        
        必须在调用语音相关功能之前调用此方法.
        """
        if not self._audio_streams_started:
            self.audio.start_streams()
            self._audio_streams_started = True
            logger.info("音频流已启动")
    
    def stop_audio_streams(self):
        """
        停止音频输入输出流.
        """
        if self._audio_streams_started:
            self.audio.stop_streams()
            self._audio_streams_started = False
            logger.info("音频流已停止")
    
    def _create_wav_chunk(self, pcm_data: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
        """
        将 PCM 数据转换为 WAV 格式.
        
        Args:
            pcm_data: PCM 原始音频数据
            sample_rate: 采样率
            channels: 声道数
            
        Returns:
            WAV 格式的音频数据
        """
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        return buffer.getvalue()
    
    async def _realtime_audio_generator(
        self, 
        duration_seconds: Optional[int] = None,
        chunk_duration_ms: int = 200,
        stop_event: Optional[asyncio.Event] = None
    ) -> AsyncIterator[bytes]:
        """
        实时音频生成器,从 AudioDevice 录音并产生音频片段.
        
        Args:
            duration_seconds: 录音时长(秒),None 表示无限录音
            chunk_duration_ms: 每个音频片段的时长(毫秒)
            stop_event: 可选的停止事件,当事件被设置时停止生成音频
            
        Yields:
            音频数据片段(bytes, WAV 格式)
        """
        sample_rate = self.audio.sample_rate
        channels = self.audio.channels
        chunk_size = self.audio.chunk_size
        
        samples_per_chunk = chunk_size
        bytes_per_sample = 2
        bytes_per_chunk = samples_per_chunk * channels * bytes_per_sample
        
        target_bytes = int(sample_rate * channels * bytes_per_sample * chunk_duration_ms / 1000)
        chunks_needed = max(1, target_bytes // bytes_per_chunk)
        
        start_time = asyncio.get_event_loop().time()
        accumulated_data = b''
        chunk_count = 0
        
        try:
            while True:
                # 检查停止事件
                if stop_event and stop_event.is_set():
                    break
                
                if duration_seconds is not None:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed >= duration_seconds:
                        break
                
                audio_chunk = await self.audio.async_get_recorded_data(timeout=1.0)
                if audio_chunk is None:
                    continue
                
                accumulated_data += audio_chunk
                chunk_count += 1
                
                if chunk_count >= chunks_needed:
                    wav_data = self._create_wav_chunk(accumulated_data, sample_rate, channels)
                    yield wav_data
                    accumulated_data = b''
                    chunk_count = 0
            
            if accumulated_data:
                wav_data = self._create_wav_chunk(accumulated_data, sample_rate, channels)
                yield wav_data
            
            yield None  # 结束信号
        except Exception as e:
            logger.error(f"录音生成器错误: {e}")
            raise
    
    async def listen_and_transcribe(
        self, 
        duration_seconds: Optional[int] = None,
        timeout: float = 30.0,
        print_realtime: bool = True,
        silence_timeout: float = 0.8
    ) -> str:
        """
        监听语音并转换为文本.
        
        Args:
            duration_seconds: 录音时长(秒),None 表示使用静音检测模式
            timeout: 超时时间(秒)
            print_realtime: 是否实时打印识别结果
            silence_timeout: 静音超时时间(秒),检测到最终结果后等待此时间,如果没有新语音则返回
            
        Returns:
            识别到的文本,如果超时或出错则返回空字符串
        """
        if not self._audio_streams_started:
            self.start_audio_streams()
        
        try:
            async with AsrWsClient(self.asr_url, segment_duration=200) as client:
                # 如果 duration_seconds 为 None，使用静音检测模式（设置一个大的超时值）
                max_duration = timeout if duration_seconds is None else duration_seconds
                
                # 创建停止事件来控制音频流
                audio_stream_stop_event = asyncio.Event()
                
                # 创建音频流生成器
                audio_stream = self._realtime_audio_generator(
                    duration_seconds=max_duration,
                    chunk_duration_ms=200,
                    stop_event=audio_stream_stop_event
                )
                
                full_text = ""
                last_text = ""  # 用于跟踪上一次的文本，避免重复打印
                last_definite_time = None  # 最后一次收到确定结果的时间
                has_received_definite = False  # 是否收到过确定结果
                
                # 创建一个任务来处理ASR响应
                async def process_responses():
                    nonlocal full_text, last_text, last_definite_time, has_received_definite
                    try:
                        async for response in client.execute_stream(audio_stream):
                            resp_dict = response.to_dict()
                            if resp_dict.get('payload_msg') and 'result' in resp_dict['payload_msg']:
                                result = resp_dict['payload_msg']['result']
                                if 'text' in result:
                                    text = result['text']
                                    is_definite = result.get('utterances', [{}])[0].get('definite', False) if result.get('utterances') else False
                                    
                                    # 实时打印识别结果（只在文本变化时打印）
                                    if print_realtime and text != last_text:
                                        status = "✅ 确定" if is_definite else "⏳ 临时"
                                        # 使用 \r 和空格清除之前的输出
                                        clear_line = " " * 100  # 足够长的空格来清除之前的文本
                                        print(f"\r{clear_line}\r🎤 ASR [{status}]: {text}", end="", flush=True)
                                        last_text = text
                                    
                                    if is_definite:
                                        full_text = text
                                        last_definite_time = asyncio.get_event_loop().time()
                                        has_received_definite = True
                                        if print_realtime:
                                            print()  # 换行，确认最终结果
                                        logger.info(f"ASR识别结果: {text}")
                    except Exception as e:
                        logger.error(f"处理ASR响应时出错: {e}")
                
                # 启动响应处理任务
                response_task = asyncio.create_task(process_responses())
                
                # 如果使用静音检测模式，监控静音超时
                if duration_seconds is None:
                    try:
                        while not response_task.done():
                            await asyncio.sleep(0.1)  # 每100ms检查一次
                            
                            # 如果收到确定结果，检查是否超过静音超时时间
                            if has_received_definite and last_definite_time is not None:
                                current_time = asyncio.get_event_loop().time()
                                elapsed = current_time - last_definite_time
                                if elapsed >= silence_timeout:
                                    # 静音超时，停止音频流
                                    if print_realtime:
                                        print(f"✅ 检测到 {silence_timeout} 秒静音，用户说完了")
                                    audio_stream_stop_event.set()
                                    # 等待响应任务完成
                                    await asyncio.sleep(0.5)  # 给一点时间让任务完成
                                    break
                    except Exception as e:
                        logger.error(f"监控静音超时时出错: {e}")
                else:
                    # 固定时长模式，等待任务完成
                    await response_task
                
                # 确保任务完成
                if not response_task.done():
                    await asyncio.sleep(0.5)  # 等待任务完成
                
                return full_text
        except asyncio.TimeoutError:
            logger.warning(f"ASR识别超时({timeout}秒)")
            if print_realtime:
                print()  # 换行
            return ""
        except Exception as e:
            logger.error(f"ASR识别失败: {e}")
            if print_realtime:
                print()  # 换行
            return ""
    
    def _convert_mp3_to_pcm(self, mp3_data: bytes, target_sample_rate: int = 16000) -> bytes:
        """
        将 MP3 音频数据转换为 PCM 格式.
        
        Args:
            mp3_data: MP3 格式的音频数据
            target_sample_rate: 目标采样率
            
        Returns:
            PCM 格式的音频数据
        """
        try:
            audio = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(target_sample_rate)
            audio = audio.set_sample_width(2)
            return audio.raw_data
        except Exception as e:
            logger.error(f"转换MP3到PCM失败: {e}")
            return b""
    
    async def speak_and_play(self, text: str, encoding: str = "mp3") -> bool:
        """
        将文本转换为语音并播放.
        
        Args:
            text: 要转换的文本
            encoding: 音频编码格式("mp3" 或 "pcm")
            
        Returns:
            是否成功
        """
        if not self._audio_streams_started:
            self.start_audio_streams()
        
        try:
            # 构建 WebSocket 连接请求头
            headers = {
                "X-Api-App-Key": self.tts_appid,
                "X-Api-Access-Key": self.tts_access_token,
                "X-Api-Resource-Id": self.tts_resource_id,
                "X-Api-Connect-Id": str(uuid.uuid4()),
            }
            
            # 连接到 WebSocket 服务器
            websocket = await websockets.connect(
                self.tts_url, 
                additional_headers=headers, 
                max_size=10 * 1024 * 1024
            )
            self.tts_websocket = websocket
            
            try:
                # 启动连接
                await start_connection(websocket)
                await wait_for_event(
                    websocket, MsgType.FullServerResponse, EventType.ConnectionStarted
                )
                
                # 按句号分割文本
                sentences = text.split("。")
                
                for sentence in sentences:
                    if not sentence:
                        continue
                    
                    # 构建请求参数
                    base_request = {
                        "user": {"uid": str(uuid.uuid4())},
                        "namespace": "BidirectionalTTS",
                        "req_params": {
                            "speaker": self.tts_voice_type,
                            "audio_params": {
                                "format": encoding,
                                "sample_rate": 24000,
                                "enable_timestamp": True,
                            },
                            "additions": json.dumps({"disable_markdown_filter": False}),
                        },
                    }
                    
                    # 启动会话
                    start_session_request = base_request.copy()
                    start_session_request["event"] = EventType.StartSession
                    session_id = str(uuid.uuid4())
                    await start_session(
                        websocket, json.dumps(start_session_request).encode(), session_id
                    )
                    await wait_for_event(
                        websocket, MsgType.FullServerResponse, EventType.SessionStarted
                    )
                    
                    # 逐字符发送文本
                    async def send_chars():
                        for char in sentence:
                            synthesis_request = base_request.copy()
                            synthesis_request["event"] = EventType.TaskRequest
                            synthesis_request["req_params"]["text"] = char
                            await task_request(
                                websocket, json.dumps(synthesis_request).encode(), session_id
                            )
                            await asyncio.sleep(0.005)
                        await finish_session(websocket, session_id)
                    
                    send_task = asyncio.create_task(send_chars())
                    
                    # 接收音频数据并播放
                    mp3_buffer = bytearray()
                    min_buffer_size = 8192
                    mp3_convert_threshold = 4096
                    playback_started = False
                    
                    while True:
                        msg = await receive_message(websocket)
                        
                        if msg.type == MsgType.FullServerResponse:
                            if msg.event == EventType.SessionFinished:
                                if len(mp3_buffer) > 0:
                                    if encoding == "mp3":
                                        pcm_data = self._convert_mp3_to_pcm(bytes(mp3_buffer), self.audio.sample_rate)
                                        if pcm_data:
                                            self.audio.put_playback_data(pcm_data)
                                    mp3_buffer.clear()
                                break
                        elif msg.type == MsgType.AudioOnlyServer:
                            if msg.payload:
                                if encoding == "mp3":
                                    mp3_buffer.extend(msg.payload)
                                    if not playback_started and len(mp3_buffer) >= min_buffer_size:
                                        playback_started = True
                                    if playback_started and len(mp3_buffer) >= mp3_convert_threshold:
                                        pcm_data = self._convert_mp3_to_pcm(bytes(mp3_buffer), self.audio.sample_rate)
                                        if pcm_data:
                                            self.audio.put_playback_data(pcm_data)
                                        mp3_buffer.clear()
                                elif encoding == "pcm":
                                    self.audio.put_playback_data(msg.payload)
                                    if not playback_started:
                                        playback_started = True
                        else:
                            # 未知消息类型，记录日志但继续处理
                            logger.warning(f"收到未知消息类型: {msg.type}")
                    
                    # 等待字符发送任务完成
                    await send_task
                    
                    # 等待播放完成
                    max_wait = 30
                    wait_count = 0
                    while self.audio.get_playback_queue_size() > 0 and wait_count < max_wait * 10:
                        await asyncio.sleep(0.1)
                        wait_count += 1
                    await asyncio.sleep(1.0)
                
                # 结束连接
                await finish_connection(websocket)
                await wait_for_event(
                    websocket, MsgType.FullServerResponse, EventType.ConnectionFinished
                )
                await websocket.close()
                
                return True
            finally:
                if websocket:
                    try:
                        await websocket.close()
                    except Exception as e:
                        logger.warning(f"关闭 WebSocket 时出错: {e}")
                self.tts_websocket = None
        except Exception as e:
            logger.error(f"TTS转换失败: {e}")
            return False
    
    async def voice_interact(
        self, 
        duration_seconds: Optional[int] = None,
        print_realtime: bool = True,
        silence_timeout: float = 0.8
    ) -> str:
        """
        完整的语音交互流程: 监听 -> 识别 -> 生成回复 -> 播放.
        
        Args:
            duration_seconds: 录音时长(秒),None 表示使用静音检测模式(检测到0.8秒静音后自动结束)
            print_realtime: 是否实时打印 ASR 和 Agent 的输出
            silence_timeout: 静音超时时间(秒),仅在 duration_seconds=None 时生效
            
        Returns:
            Agent 的文本回复
        """
        # 1. 监听并识别语音
        if print_realtime:
            print("\n" + "="*60)
            if duration_seconds is None:
                print(f"🎤 步骤 1/3: 正在监听语音... (检测到 {silence_timeout} 秒静音后自动结束)")
            else:
                print(f"🎤 步骤 1/3: 正在监听语音... (时长: {duration_seconds} 秒)")
            print("="*60)
        
        user_text = await self.listen_and_transcribe(
            duration_seconds=duration_seconds,
            print_realtime=print_realtime,
            silence_timeout=silence_timeout
        )
        
        if not user_text:
            if print_realtime:
                print("⚠️  未识别到语音")
            logger.warning("未识别到语音")
            return ""
        
        if print_realtime:
            print(f"\n✅ 识别完成: {user_text}\n")
        
        # 2. 使用 Agent 生成回复（流式生成并实时播放）
        if print_realtime:
            print("="*60)
            print("🤖 步骤 2/3: Agent 正在生成回复（流式播放）...")
            print("="*60)
            print("🤖 Agent: ", end="", flush=True)
        
        # 句子分隔符：中文和英文的句号、逗号、问号、感叹号、分号
        sentence_delimiters = set(['。', '.', '，', ',', '？', '?', '！', '!', '；', ';', '\n'])
        
        response_text = ""
        current_sentence = ""  # 当前正在收集的句子
        tts_queue = asyncio.Queue()  # TTS播放队列
        
        # 启动TTS播放队列处理器
        async def tts_queue_processor():
            """处理TTS播放队列，确保按顺序播放"""
            while True:
                text_segment = await tts_queue.get()
                if text_segment is None:  # 结束信号
                    break
                if text_segment.strip():  # 只播放非空文本
                    try:
                        await self.speak_and_play(text_segment.strip())
                    except Exception as e:
                        logger.error(f"TTS播放失败: {e}")
                tts_queue.task_done()
        
        tts_processor_task = asyncio.create_task(tts_queue_processor())
        
        try:
            async for event in self.astream_with_tools(user_text):
                event_type = event["type"]
                
                if event_type == "token":
                    token = event["content"]
                    # 实时打印每个 token
                    if print_realtime:
                        print(token, end="", flush=True)
                    
                    response_text += token
                    current_sentence += token
                    
                    # 检测句子分隔符
                    if token in sentence_delimiters:
                        # 将当前句子加入TTS播放队列
                        if current_sentence.strip():
                            await tts_queue.put(current_sentence)
                            if print_realtime:
                                print(f" [🔊 已加入播放队列]", end="", flush=True)
                        current_sentence = ""  # 重置当前句子
                
                elif event_type == "tool_call_start" and print_realtime:
                    # 显示工具调用信息
                    print(f"\n  🔧 [调用工具: {event['tool_name']}]")
                    print(f"     参数: {event['tool_args']}", flush=True)
                
                elif event_type == "tool_call_end" and print_realtime:
                    # 显示工具返回结果
                    print(f"  ✅ [工具返回: {event['tool_result']}]\n🤖 Agent: ", end="", flush=True)
                
                elif event_type == "error" and print_realtime:
                    print(f"\n  ❌ 错误: {event['error']}", flush=True)
                
                elif event_type == "done":
                    if print_realtime:
                        print("\n")
                    break
            
            # 将剩余的文本也加入播放队列
            if current_sentence.strip():
                await tts_queue.put(current_sentence.strip())
            
            # 发送结束信号
            await tts_queue.put(None)
            
            # 等待TTS播放队列处理完成
            await tts_processor_task
            
        except Exception as e:
            logger.error(f"生成回复时出错: {e}")
            # 确保发送结束信号
            try:
                await tts_queue.put(None)
            except:
                pass
        
        if print_realtime:
            print(f"✅ 回复生成和播放完成\n")
        
        return response_text
    
    def cleanup(self):
        """
        清理资源,包括停止音频流和关闭连接.
        
        Note:
            WebSocket 连接通常在 speak_and_play 方法中已经关闭,
            这里主要是确保资源清理和状态重置.
        """
        self.stop_audio_streams()
        # WebSocket 连接应该在 speak_and_play 方法中已经关闭
        # 这里只需要重置引用
        if self.tts_websocket:
            # 如果 websocket 仍然存在（异常情况），尝试关闭
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环正在运行，创建任务来关闭
                    asyncio.create_task(self._close_websocket())
                else:
                    # 如果事件循环未运行，直接运行
                    loop.run_until_complete(self._close_websocket())
            except RuntimeError:
                # 如果没有事件循环，直接重置引用
                self.tts_websocket = None
            except Exception as e:
                logger.warning(f"关闭 WebSocket 连接时出错: {e}")
                self.tts_websocket = None
        self.audio.cleanup()
        logger.info("资源已清理")
    
    async def _close_websocket(self):
        """
        异步关闭 WebSocket 连接.
        """
        if self.tts_websocket:
            try:
                await self.tts_websocket.close()
            except Exception as e:
                logger.warning(f"关闭 WebSocket 时出错: {e}")
            finally:
                self.tts_websocket = None
    
