"""
ASR（自动语音识别）和 TTS（文本转语音）集成模块

该模块提供了实时语音识别和文本转语音的功能，支持：
- 实时音频流式识别（ASR）
- 文本转语音播放（TTS）
- 音频格式转换（MP3转PCM）
- 队列管理和异步处理

使用火山引擎（Volcengine）的语音服务进行ASR和TTS处理。
"""

import asyncio
import io
import os
import wave
import json
import uuid
import copy
from queue import Queue
from pydub import AudioSegment
import websockets
from robota.utils.logging import default_logger as logger
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


class ASRTTS:
    """
    ASR和TTS集成类
    
    提供实时语音识别和文本转语音功能，支持：
    - 实时音频流式识别，自动将识别结果放入队列
    - 文本转语音播放，支持流式处理和低延迟播放
    - 音频设备管理（录音和播放）
    - 队列管理和事件通知机制
    """
    def __init__(self):
        """
        初始化ASR和TTS集成类
        
        初始化音频设备、ASR客户端配置和TTS客户端配置，并启动音频流。
        所有配置参数都针对火山引擎语音服务进行了优化。
        """
        # ========== 音频设备配置 ==========
        # 初始化音频设备，用于录音和播放
        self.audio_device = AudioDevice(
            sample_rate=16000,  # 采样率16kHz，符合ASR服务要求
            channels=1,         # 单声道
            chunk_size=1024,    # 每次读取1024个样本
            enable_aec=True     # 启用回声消除（实时ASR测试时可以关闭）
        )
        # 启动音频设备的录音和播放流
        self.audio_device.start_streams()

        # ========== ASR（语音识别）客户端配置 ==========
        self.asr_url = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"  # ASR WebSocket服务地址
        self.asr_seg_duration = 200  # ASR分段时长（毫秒），每200ms发送一次音频数据
        self.asr_queue = Queue()  # ASR识别结果队列，存储识别到的文本
        self.asr_queue_event = None  # 异步事件，用于通知有新识别结果放入队列（在异步上下文中创建）
        self.asr_chat_id = 1  # ASR chat_id 计数器，用于标识不同的识别会话

        # ========== TTS（文本转语音）客户端配置 ==========
        self.tts_queue = Queue()  # TTS文本队列，存储待转换为语音的文本
        self.tts_queue_event = None  # 异步事件，用于通知有新文本放入队列（在异步上下文中创建）
        self.tts_chat_id = 1  # TTS chat_id 计数器，用于标识不同的TTS会话
        self.tts_appid = os.getenv("TTS_APP_KEY")  # TTS应用ID（从环境变量获取）
        self.tts_access_token = os.getenv("TTS_ACCESS_KEY")  # TTS访问令牌（从环境变量获取）
        self.tts_resource_id = "seed-tts-2.0"  # TTS资源ID，指定使用的TTS模型
        self.tts_voice_type = "zh_male_m191_uranus_bigtts"  # TTS语音类型（中文男声）
        self.tts_encoding = "mp3"  # TTS音频编码格式（mp3或pcm）
        self.tts_endpoint = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"  # TTS WebSocket服务地址
        self.tts_sample_rate = 16000  # TTS播放采样率（Hz）
        self.tts_running = False  # TTS处理器运行状态标志

    async def _realtime_audio_generator(self, audio_device: AudioDevice, duration_seconds: int = None, chunk_duration_ms: int = 200):
        """
        实时音频生成器（异步生成器）
        
        从AudioDevice持续录音，将PCM音频数据转换为WAV格式，并按指定时长分块产生音频片段。
        用于实时ASR识别，支持流式处理。
        
        Args:
            audio_device: AudioDevice实例，用于获取录音数据
            duration_seconds: 录音时长(秒)，None表示无限录音直到手动停止
            chunk_duration_ms: 每个音频片段的时长(毫秒)，默认200ms，用于控制发送频率
            
        Yields:
            bytes: WAV格式的音频数据片段，最后yield None表示音频流结束
            
        Raises:
            Exception: 录音过程中发生错误时抛出异常
        """
        # ========== 获取音频设备参数 ==========
        sample_rate = audio_device.sample_rate  # 采样率（Hz）
        channels = audio_device.channels  # 声道数（1=单声道，2=立体声）
        chunk_size = audio_device.chunk_size  # 每次从设备读取的样本数
        
        # ========== 计算音频数据块大小 ==========
        samples_per_chunk = chunk_size  # 每个chunk的样本数
        bytes_per_sample = 2  # int16格式，每个样本2字节
        bytes_per_chunk = samples_per_chunk * channels * bytes_per_sample  # 每个chunk的字节数
        
        # ========== 计算需要累积多少个chunk才能达到目标时长 ==========
        # 目标字节数 = 采样率 × 声道数 × 每样本字节数 × 时长(秒)
        target_bytes = int(sample_rate * channels * bytes_per_sample * chunk_duration_ms / 1000)
        chunks_needed = max(1, target_bytes // bytes_per_chunk)  # 至少需要1个chunk
        
        # logger.info(f"实时录音配置: 采样率={sample_rate}, 块大小={chunk_size}, 每{chunk_duration_ms}ms发送一次")
        # logger.info(f"每次发送需要累积 {chunks_needed} 个chunk, 约 {target_bytes} 字节")
        
        # ========== 初始化录音状态 ==========
        start_time = asyncio.get_event_loop().time()  # 录音开始时间
        accumulated_data = b''  # 累积的音频数据（PCM格式）
        chunk_count = 0  # 已累积的chunk数量
        
        try:
            # ========== 主循环：持续录音并产生音频片段 ==========
            while True:
                # 检查是否达到录音时长限制
                if duration_seconds is not None:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed >= duration_seconds:
                        # logger.info(f"录音时长达到 {duration_seconds} 秒，停止录音")
                        break
                
                # 异步获取录音数据，超时时间1秒（避免阻塞）
                audio_chunk = await audio_device.async_get_recorded_data(timeout=1.0)
                if audio_chunk is None:
                    # 超时或没有数据，继续等待
                    continue
                
                # 累积音频数据
                accumulated_data += audio_chunk
                chunk_count += 1
                
                # 当累积足够的数据后，转换为WAV格式并发送
                if chunk_count >= chunks_needed:
                    # 将PCM数据转换为WAV格式（添加WAV文件头）
                    wav_data = self._create_wav_chunk(accumulated_data, sample_rate, channels)
                    yield wav_data  # 产生WAV格式的音频片段
                    
                    # 重置累积状态，准备下一批数据
                    accumulated_data = b''
                    chunk_count = 0
            
            # ========== 发送剩余的音频数据（如果有） ==========
            # 录音结束时，可能还有未达到chunks_needed的数据，需要发送
            if accumulated_data:
                wav_data = self._create_wav_chunk(accumulated_data, sample_rate, channels)
                yield wav_data
            
            # 发送结束信号（None表示音频流结束）
            yield None
            
        except Exception as e:
            logger.error(f"录音生成器错误: {e}")
            raise
    
    def _create_wav_chunk(self, pcm_data: bytes, sample_rate: int, channels: int) -> bytes:
        """
        创建WAV格式的音频数据
        
        将PCM原始音频数据转换为WAV格式，添加WAV文件头信息。
        WAV格式包含文件头（采样率、声道数、位深度等）和音频数据。
        
        Args:
            pcm_data: PCM原始音频数据（字节流），16位整数格式
            sample_rate: 采样率（Hz），如16000、44100等
            channels: 声道数，1=单声道，2=立体声
            
        Returns:
            bytes: 完整的WAV格式音频数据（包含文件头和数据）
        """
        # 创建内存缓冲区，用于存储WAV数据
        buffer = io.BytesIO()
        
        # 使用wave模块写入WAV文件头和数据
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)  # 设置声道数
            wav_file.setsampwidth(2)  # 设置样本宽度为2字节（16-bit）
            wav_file.setframerate(sample_rate)  # 设置采样率
            wav_file.writeframes(pcm_data)  # 写入PCM音频数据
        
        # 返回完整的WAV文件数据（包含文件头和数据）
        return buffer.getvalue()

    async def start_realtime_asr(self, duration_seconds: int = None, silence_timeout_ms: int = 600):
        """
        启动实时ASR（自动语音识别）识别
        
        持续录音并实时识别语音，将识别结果放入队列。
        使用静音超时机制：当超过指定时间没有识别到新文字时，将累积的文本放入队列。
        
        Args:
            duration_seconds: 录音时长(秒)，None表示无限录音直到手动停止
            silence_timeout_ms: 静音超时时间(毫秒)，默认800ms
                               当超过此时间没有识别到新文字时，将当前累积的识别结果放入队列
                               这样可以实现自动分段，将一句话识别完成后立即放入队列
                               
        Raises:
            Exception: ASR识别过程中发生错误时抛出异常
        """
        # ========== 初始化异步事件 ==========
        # 在异步上下文中创建事件，用于通知有新识别结果
        if self.asr_queue_event is None:
            self.asr_queue_event = asyncio.Event()
        
        try:
            # ========== 创建实时音频生成器 ==========
            # 从音频设备持续获取音频数据，转换为WAV格式并分块发送
            audio_stream = self._realtime_audio_generator(
                self.audio_device,
                duration_seconds=duration_seconds,
                chunk_duration_ms=200  # 每200ms发送一次音频数据
            )
            
            # ========== 创建ASR客户端并进行实时识别 ==========
            async with AsrWsClient(self.asr_url, self.asr_seg_duration) as asr_client:
                # ========== 跟踪识别状态 ==========
                last_text_time = None  # 最后一次收到识别文本的时间戳
                accumulated_text = ""  # 累积的识别文本（ASR服务会返回完整的累积文本）
                last_is_definite = False  # 最后一次识别结果的确定状态
                
                # ========== 创建超时检查任务 ==========
                # 后台任务：定期检查是否超过静音超时时间，如果是则将结果放入队列
                async def check_timeout():
                    """
                    超时检查任务
                    
                    每100ms检查一次，如果超过silence_timeout_ms没有收到新文本，
                    且有累积文本且is_definite为True，则将结果放入队列并重置状态。
                    """
                    nonlocal last_text_time, accumulated_text, last_is_definite
                    while True:
                        await asyncio.sleep(0.1)  # 每100ms检查一次（10Hz检查频率）
                        current_time = asyncio.get_event_loop().time()
                        
                        # 如果超过静音超时时间没有收到新文本，且有累积文本且is_definite为True，则放入队列
                        if last_text_time is not None and accumulated_text and last_is_definite:
                            elapsed_ms = (current_time - last_text_time) * 1000
                            if elapsed_ms >= silence_timeout_ms:
                                # 将识别结果放入队列
                                result = {
                                    "text": accumulated_text,
                                    "chat_id": self.asr_chat_id
                                }
                                self.asr_queue.put(result)
                                self.asr_chat_id += 1

                                # 通知有新结果（唤醒等待队列的代码）
                                self.asr_queue_event.set()
                                # logger.info(f"超时({elapsed_ms:.0f}ms)未识别到新文字，将结果放入队列: {accumulated_text}, chat_id: {self.chat_id}")
                                
                                # 重置状态，准备接收下一段识别文本
                                accumulated_text = ""
                                last_text_time = None
                                last_is_definite = False
                
                # 启动超时检查任务（后台运行）
                timeout_task = asyncio.create_task(check_timeout())
                
                try:
                    # ========== 处理ASR响应流 ==========
                    # 从ASR客户端接收识别结果（流式处理）
                    async for response in asr_client.execute_stream(audio_stream):
                        # 解析响应数据为字典格式
                        resp_dict = response.to_dict()
                        
                        # 检查响应中是否包含识别结果
                        if resp_dict.get('payload_msg') and 'result' in resp_dict['payload_msg']:
                            result = resp_dict['payload_msg']['result']
                            
                            # 如果包含识别文本（ASR服务返回的文本是累积的完整文本）
                            if 'text' in result:
                                text = result['text']
                                current_time = asyncio.get_event_loop().time()
                                
                                # 获取识别结果的确定状态（是否为最终结果）
                                is_definite = result.get('utterances', [{}])[0].get('definite', False) if result.get('utterances') else False
                                
                                # 更新累积文本（使用最新的完整文本，ASR会不断更新完整文本）
                                accumulated_text = text
                                last_text_time = current_time  # 更新最后收到文本的时间
                                last_is_definite = is_definite  # 更新确定状态
                                
                                # logger.debug(f"收到识别文本: {text}, is_definite: {is_definite}")
                
                except Exception as e:
                    logger.error(f"实时ASR处理失败: {e}")
                    raise
                finally:
                    # ========== 清理资源 ==========
                    # 取消超时检查任务
                    timeout_task.cancel()
                    try:
                        await timeout_task
                    except asyncio.CancelledError:
                        # 任务正常取消，忽略异常
                        pass
                    
                    # 如果还有未处理的文本（识别结束时可能还有文本未放入队列），且is_definite为True，放入队列
                    if accumulated_text and last_is_definite:
                        result = {
                            "text": accumulated_text,
                            "chat_id": self.asr_chat_id
                        }
                        self.asr_queue.put(result)
                        # 通知有新结果
                        self.asr_queue_event.set()
                        # logger.info(f"识别结束，将剩余结果放入队列: {accumulated_text}, chat_id: {self.chat_id}")
                        
        except Exception as e:
            logger.error(f"启动实时ASR失败: {e}")
            raise

    def _convert_mp3_to_pcm(self, mp3_data: bytes, target_sample_rate: int = 16000) -> bytes:
        """
        将 MP3 音频数据转换为 PCM 格式
        
        使用 pydub 库进行音频格式转换，将MP3压缩音频转换为PCM原始音频数据。
        转换过程包括：格式转换、声道转换（转为单声道）、采样率转换、位深度转换。
        增加了数据验证和错误处理，更好地处理流式MP3数据。
        
        Args:
            mp3_data: MP3 格式的音频数据（字节流）
            target_sample_rate: 目标采样率（Hz），默认 16000 Hz
                               需要与音频设备的采样率匹配
        
        Returns:
            bytes: PCM 格式的音频数据（字节流），16位整数格式，单声道
                  如果转换失败则返回空字节流 b""
        """
        # ========== 数据验证 ==========
        if not mp3_data or len(mp3_data) < 100:
            # MP3数据太短，可能不完整，返回空数据
            return b""
        
        try:
            # ========== 验证MP3数据是否包含有效的MP3头部 ==========
            # MP3文件通常以ID3标签或帧同步字节开始
            if mp3_data[:3] != b'ID3' and not (mp3_data[0] == 0xFF and (mp3_data[1] & 0xE0) == 0xE0):
                # 尝试查找MP3帧同步字节
                found_sync = False
                for i in range(min(100, len(mp3_data) - 1)):
                    if mp3_data[i] == 0xFF and (mp3_data[i+1] & 0xE0) == 0xE0:
                        mp3_data = mp3_data[i:]
                        found_sync = True
                        break
                if not found_sync:
                    logger.warning(f"MP3数据未找到有效头部，数据长度: {len(mp3_data)}")
                    return b""
            
            # ========== 使用 pydub 从字节流加载 MP3 数据 ==========
            audio = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
            
            # ========== 音频格式转换 ==========
            # 转换为单声道（TTS服务可能返回立体声，但播放设备需要单声道）
            audio = audio.set_channels(1)
            
            # 转换采样率到目标采样率（匹配音频设备的采样率）
            audio = audio.set_frame_rate(target_sample_rate)
            
            # 转换为 16 位 PCM（2 字节 = 16 位），这是标准的PCM格式
            audio = audio.set_sample_width(2)
            
            # ========== 获取原始 PCM 数据 ==========
            # raw_data 返回的是原始PCM字节流，可以直接用于播放
            pcm_data = audio.raw_data
            
            return pcm_data
        except Exception as e:
            # 转换失败时，记录警告但不抛出异常，返回空数据
            logger.debug(f"转换MP3到PCM失败: {e}, 数据长度: {len(mp3_data)}")
            return b""  # 转换失败时返回空字节流

    async def _create_websocket_connection(self):
        """
        创建 WebSocket 连接到 TTS 服务
        
        Returns:
            websockets.WebSocketClientProtocol: WebSocket 连接对象
            
        Raises:
            Exception: 连接失败时抛出异常
        """
        # ========== 构建 WebSocket 连接请求头 ==========
        headers = {
            "X-Api-App-Key": self.tts_appid,  # TTS应用ID
            "X-Api-Access-Key": self.tts_access_token,  # TTS访问令牌
            "X-Api-Resource-Id": (self.tts_resource_id),  # TTS资源ID（模型ID）
            "X-Api-Connect-Id": str(uuid.uuid4()),  # 生成唯一的连接 ID
        }

        # ========== 连接到 WebSocket 服务器 ==========
        # 使用更宽松的 ping 配置，避免长时间运行后超时
        websocket = await websockets.connect(
            self.tts_endpoint, 
            additional_headers=headers, 
            max_size=10 * 1024 * 1024,  # 最大消息大小10MB（用于接收音频数据）
            ping_interval=30,  # 每30秒发送一次ping保持连接活跃（从20秒增加到30秒）
            ping_timeout=20,   # ping超时时间20秒（从10秒增加到20秒，更宽松）
            close_timeout=10  # 关闭连接超时时间10秒
        )
        # logger.info(
        #     f"TTS: 已连接到服务器, Logid: {websocket.response.headers.get('x-tt-logid', 'N/A')}",
        # )
        
        # ========== 启动连接 ==========
        await start_connection(websocket)
        await wait_for_event(
            websocket, MsgType.FullServerResponse, EventType.ConnectionStarted
        )
        
        return websocket

    async def _close_websocket_connection(self, websocket):
        """
        安全关闭 WebSocket 连接
        
        Args:
            websocket: WebSocket 连接对象
        """
        if websocket is None:
            return
            
        try:
            # ========== 清理资源：结束连接 ==========
            await finish_connection(websocket)
            try:
                msg = await wait_for_event(
                    websocket, MsgType.FullServerResponse, EventType.ConnectionFinished
                )
            except Exception as e:
                logger.warning(f"等待连接结束事件失败: {e}")
            await websocket.close()
            logger.info("TTS: 连接已关闭")
        except Exception as e:
            logger.warning(f"关闭 WebSocket 连接时出错: {e}")
            try:
                await websocket.close()
            except:
                pass

    async def start_tts_processor(self):
        """
        启动 TTS（文本转语音）处理器
        
        该方法会持续运行，从 tts_queue 中读取文本，调用 TTS 服务转换为语音并实时播放。
        使用 WebSocket 连接进行双向通信，支持流式处理（一边接收音频数据一边播放）。
        支持自动重连机制：当连接断开时，自动重新建立连接并继续处理队列中的文本。
        
        工作流程：
        1. 建立 WebSocket 连接到 TTS 服务
        2. 持续监听队列中的文本
        3. 对每个文本调用 TTS 服务转换为语音
        4. 实时接收音频数据并播放（流式处理，降低延迟）
        5. 当连接断开时，自动重连并继续处理
        
        Raises:
            Exception: TTS处理过程中发生错误时抛出异常
        """
        # ========== 初始化异步事件 ==========
        # 在异步上下文中创建事件，用于通知有新文本放入队列
        if self.tts_queue_event is None:
            self.tts_queue_event = asyncio.Event()
        
        self.tts_running = True  # 设置运行标志
        
        websocket = None
        reconnect_delay = 1.0  # 初始重连延迟（秒）
        max_reconnect_delay = 30.0  # 最大重连延迟（秒）
        
        try:
            # ========== 主循环：支持自动重连 ==========
            while self.tts_running:
                try:
                    # ========== 建立或重新建立 WebSocket 连接 ==========
                    if websocket is None or websocket.closed:
                        logger.info("TTS: 正在建立 WebSocket 连接...")
                        websocket = await self._create_websocket_connection()
                        logger.info("TTS: WebSocket 连接已建立")
                        reconnect_delay = 1.0  # 连接成功后重置重连延迟

                    # ========== 持续处理队列中的文本 ==========
                    while self.tts_running:
                        try:
                            # 先检查队列，如果有数据立即处理（减少延迟）
                            # 批量处理队列中的所有文本，确保及时响应
                            processed_any = False
                            current_text_data = None  # 当前正在处理的文本数据
                            
                            while not self.tts_queue.empty():
                                text_data = self.tts_queue.get_nowait()
                                if text_data:
                                    # 提取文本内容（支持字典格式或字符串格式）
                                    text = text_data.get("text", "") if isinstance(text_data, dict) else str(text_data)
                                    if text:
                                        current_text_data = text_data  # 保存当前文本数据，用于重连时重新放入队列
                                        # logger.info(f"TTS: 处理文本: {text}")
                                        await self._process_tts_text(websocket, text)
                                        processed_any = True
                                        current_text_data = None  # 处理成功后清除
                            
                            # 如果已经处理了文本，立即继续循环检查下一个（保持高响应性）
                            if processed_any:
                                continue
                            
                            # ========== 队列为空时，等待事件通知 ==========
                            # 有新文本加入时会立即被唤醒，避免空转消耗CPU
                            try:
                                await asyncio.wait_for(self.tts_queue_event.wait(), timeout=1.0)
                                self.tts_queue_event.clear()
                                # 事件被触发后，立即检查队列并处理（可能有多条文本）
                                # 上面的循环会处理所有队列中的文本
                            except asyncio.TimeoutError:
                                # 超时后继续循环检查（保持响应性，同时避免空转）
                                continue
                                
                        except (websockets.exceptions.ConnectionClosed, 
                                websockets.exceptions.WebSocketException,
                                OSError) as e:
                            # ========== 连接断开异常：准备重连 ==========
                            logger.warning(f"TTS: WebSocket 连接断开: {e}")
                            
                            # 如果当前有正在处理的文本，重新放回队列
                            if current_text_data is not None:
                                self.tts_queue.put(current_text_data)
                                logger.info("TTS: 将未处理的文本重新放回队列")
                            
                            # 关闭旧连接
                            await self._close_websocket_connection(websocket)
                            websocket = None
                            
                            # 如果还在运行，等待后重连
                            if self.tts_running:
                                logger.info(f"TTS: {reconnect_delay:.1f} 秒后尝试重连...")
                                await asyncio.sleep(reconnect_delay)
                                # 指数退避：逐渐增加重连延迟，但不超过最大值
                                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                            break  # 跳出内层循环，重新建立连接
                            
                        except Exception as e:
                            # ========== 其他异常：记录错误并继续 ==========
                            error_msg = str(e)
                            # 检查是否是连接相关的错误
                            if "keepalive ping timeout" in error_msg or "connection" in error_msg.lower():
                                logger.warning(f"TTS: 检测到连接问题: {e}")
                                # 关闭连接并准备重连
                                await self._close_websocket_connection(websocket)
                                websocket = None
                                
                                # 如果当前有正在处理的文本，重新放回队列
                                if current_text_data is not None:
                                    self.tts_queue.put(current_text_data)
                                    logger.info("TTS: 将未处理的文本重新放回队列")
                                
                                # 如果还在运行，等待后重连
                                if self.tts_running:
                                    logger.info(f"TTS: {reconnect_delay:.1f} 秒后尝试重连...")
                                    await asyncio.sleep(reconnect_delay)
                                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                                break  # 跳出内层循环，重新建立连接
                            else:
                                # 其他类型的错误，记录但继续处理
                                logger.error(f"TTS处理队列文本失败: {e}")
                                await asyncio.sleep(0.1)  # 出错后稍等再继续（避免错误循环）

                except Exception as e:
                    # ========== 外层异常：记录错误并尝试重连 ==========
                    logger.error(f"TTS处理器外层循环失败: {e}")
                    await self._close_websocket_connection(websocket)
                    websocket = None
                    
                    if self.tts_running:
                        logger.info(f"TTS: {reconnect_delay:.1f} 秒后尝试重连...")
                        await asyncio.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

        except Exception as e:
            logger.error(f"TTS处理器失败: {e}")
            raise
        finally:
            # ========== 清理资源 ==========
            await self._close_websocket_connection(websocket)
            self.tts_running = False  # 重置运行标志

    async def _process_tts_text(self, websocket, text: str):
        """
        处理单个文本的 TTS 转换和播放
        
        实现流式处理：一边接收音频数据一边播放，降低延迟。
        使用生产者-消费者模式：发送文本任务、接收音频任务、播放音频任务并行运行。
        
        处理流程：
        1. 按句号分割文本，逐句处理
        2. 为每句话创建TTS会话
        3. 逐字符发送文本到TTS服务（流式输入）
        4. 并行接收音频数据并实时播放（流式输出）
        
        Args:
            websocket: WebSocket 连接对象，用于与TTS服务通信
            text: 要转换为语音的文本内容
        """
        # ========== 按句号分割文本，逐句处理 ==========
        # 将长文本分割成多个句子，每句话单独处理，提高响应性
        sentences = text.split("。")
        
        for sentence in sentences:
            # 跳过空句子
            if not sentence.strip():
                continue

            # ========== 构建基础请求参数 ==========
            # TTS服务的请求参数模板，包含用户信息、语音类型、音频格式等
            base_request = {
                "user": {
                    "uid": str(uuid.uuid4()),  # 生成唯一用户ID
                },
                "namespace": "BidirectionalTTS",  # TTS命名空间
                "req_params": {
                    "speaker": self.tts_voice_type,  # 语音类型（如：中文男声）
                    "audio_params": {
                        "format": self.tts_encoding,  # 音频编码格式（mp3或pcm）
                        "sample_rate": 24000,  # 服务器端采样率（TTS服务使用的采样率）
                        "enable_timestamp": True,  # 启用时间戳
                    },
                    "additions": json.dumps(
                        {
                            "disable_markdown_filter": False,  # 不禁用Markdown过滤
                        }
                    ),
                },
            }

            # ========== 启动TTS会话 ==========
            # 每个句子需要创建一个新的TTS会话
            start_session_request = copy.deepcopy(base_request)
            start_session_request["event"] = EventType.StartSession
            session_id = str(uuid.uuid4())  # 生成唯一的会话ID
            await start_session(
                websocket, json.dumps(start_session_request).encode(), session_id
            )
            # 等待会话启动确认
            await wait_for_event(
                websocket, MsgType.FullServerResponse, EventType.SessionStarted
            )

            # ========== 逐字符发送文本（异步函数） ==========
            # 流式输入：逐字符发送文本，TTS服务可以实时开始合成
            async def send_chars():
                """
                发送文本任务（生产者）
                
                逐字符发送文本到TTS服务，字符间有短暂延迟，模拟自然输入。
                """
                try:
                    for char in sentence:
                        synthesis_request = copy.deepcopy(base_request)
                        synthesis_request["event"] = EventType.TaskRequest
                        synthesis_request["req_params"]["text"] = char
                        await task_request(
                            websocket, json.dumps(synthesis_request).encode(), session_id
                        )
                        await asyncio.sleep(0.005)  # 字符间延迟 5ms（模拟自然输入速度）

                    # 发送会话结束请求（通知TTS服务文本发送完成）
                    await finish_session(websocket, session_id)
                except (websockets.exceptions.ConnectionClosed, 
                        websockets.exceptions.WebSocketException,
                        OSError) as e:
                    # 连接断开，向上抛出异常以便外层代码重连
                    logger.warning(f"TTS: 发送文本时连接断开: {e}")
                    raise
                except Exception as e:
                    # 其他错误也向上抛出
                    logger.error(f"TTS: 发送文本失败: {e}")
                    raise

            # ========== 启动发送文本任务（后台运行） ==========
            send_task = asyncio.create_task(send_chars())

            # ========== 创建异步队列用于存储接收到的音频数据 ==========
            # 生产者-消费者模式：接收任务将音频数据放入队列，播放任务从队列取出并播放
            audio_queue = asyncio.Queue()
            session_finished = False  # 会话结束标志
            
            # ========== 接收音频数据的任务（生产者） ==========
            async def receive_audio_task():
                """
                接收音频数据任务（生产者）
                
                持续从WebSocket接收音频数据消息，将音频数据放入队列。
                当收到会话结束事件时，发送结束标记（None）到队列。
                """
                nonlocal session_finished
                try:
                    while not session_finished:
                        try:
                            # 添加超时处理，避免长时间阻塞导致连接超时
                            msg = await asyncio.wait_for(receive_message(websocket), timeout=30.0)
                            
                            if msg.type == MsgType.FullServerResponse:
                                if msg.event == EventType.SessionFinished:
                                    # 会话结束，发送结束标记到队列
                                    await audio_queue.put(None)  # None 表示会话结束
                                    session_finished = True
                                    break
                            elif msg.type == MsgType.AudioOnlyServer:
                                # 处理音频数据消息（TTS服务返回的音频数据）
                                if msg.payload:
                                    # 将音频数据放入队列（供播放任务消费）
                                    await audio_queue.put(msg.payload)
                            else:
                                # 未知消息类型，记录警告但继续处理
                                logger.warning(f"TTS: 收到未知消息类型: {msg.type}")
                        except asyncio.TimeoutError:
                            # 超时后继续循环，保持连接活跃（ping会自动处理）
                            continue
                        except (websockets.exceptions.ConnectionClosed, 
                                websockets.exceptions.WebSocketException,
                                OSError) as e:
                            # WebSocket连接已关闭，向上抛出异常以便外层代码重连
                            logger.warning(f"TTS: 接收音频时连接断开: {e}")
                            session_finished = True
                            await audio_queue.put(None)  # 通知播放任务停止
                            raise  # 向上抛出异常，让外层代码处理重连
                except (websockets.exceptions.ConnectionClosed, 
                        websockets.exceptions.WebSocketException,
                        OSError):
                    # 重新抛出连接异常，让外层代码处理
                    raise
                except Exception as e:
                    logger.error(f"TTS: 接收音频数据任务失败: {e}")
                    await audio_queue.put(None)  # 出错时也发送结束标记（通知播放任务停止）
                    # 对于其他异常，也向上抛出以便外层代码处理
                    raise
            
            # ========== 播放音频数据的任务（消费者） ==========
            async def playback_audio_task():
                """
                播放音频数据任务（消费者）
                
                持续从队列取出音频数据，转换格式（如MP3转PCM）并播放。
                使用缓冲策略：累积一定量的音频数据再开始播放，降低延迟。
                """
                mp3_buffer = bytearray()  # MP3数据缓冲区（用于累积MP3数据）
                playback_started = False  # 是否已开始播放
                min_buffer_size = 2048  # 最小 MP3 缓冲大小（约 2KB，降低延迟）
                mp3_convert_threshold = 2048  # MP3 转换阈值（累积到2KB再转换，平衡延迟和效率）
                
                try:
                    while True:
                        # 从队列中获取音频数据，设置超时避免无限等待
                        try:
                            audio_data = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            # 超时但会话可能还在进行，继续等待
                            continue
                        
                        # ========== 检查是否收到结束标记 ==========
                        if audio_data is None:
                            # 会话结束，转换并播放剩余的 MP3 数据
                            if len(mp3_buffer) > 0:
                                # logger.info(f"TTS: 会话结束,转换剩余MP3数据 {len(mp3_buffer)} bytes")
                                if self.tts_encoding == "mp3":
                                    pcm_data = self._convert_mp3_to_pcm(bytes(mp3_buffer), self.tts_sample_rate)
                                    if pcm_data:
                                        self.audio_device.put_playback_data(pcm_data)
                                        # logger.info(f"TTS: 添加剩余PCM数据: {len(pcm_data)} bytes")
                                mp3_buffer.clear()
                            break
                        
                        # ========== 处理音频数据 ==========
                        if self.tts_encoding == "mp3":
                            # MP3 格式：需要转换为PCM才能播放
                            # 累积 MP3 数据到缓冲区
                            mp3_buffer.extend(audio_data)
                            
                            # 检查是否达到最小缓冲大小，开始播放（降低延迟）
                            if not playback_started and len(mp3_buffer) >= min_buffer_size:
                                # logger.info(f"TTS: MP3缓冲已满({len(mp3_buffer)} bytes),开始播放")
                                playback_started = True
                            
                            # 批量转换策略：累积足够的 MP3 数据再转换（提高转换效率）
                            if playback_started and len(mp3_buffer) >= mp3_convert_threshold:
                                pcm_data = self._convert_mp3_to_pcm(bytes(mp3_buffer), self.tts_sample_rate)
                                if pcm_data:
                                    self.audio_device.put_playback_data(pcm_data)
                                    # logger.info(f"TTS: 转换并添加PCM: {len(pcm_data)} bytes (来自{len(mp3_buffer)} MP3)")
                                    mp3_buffer.clear()
                                else:
                                    # logger.warning(f"TTS: MP3转PCM失败")
                                    mp3_buffer.clear()
                                    
                        elif self.tts_encoding == "pcm":
                            # PCM 格式：直接放入播放队列（无需转换）
                            self.audio_device.put_playback_data(audio_data)
                            if not playback_started:
                                playback_started = True
                            # logger.info(f"TTS: 添加PCM数据: {len(audio_data)} bytes")
                            
                except Exception as e:
                    logger.error(f"TTS: 播放音频数据任务失败: {e}")
            
            # ========== 启动接收和播放任务（并行运行） ==========
            # 三个任务并行运行：发送文本、接收音频、播放音频
            receive_task = asyncio.create_task(receive_audio_task())
            playback_task = asyncio.create_task(playback_audio_task())
            
            # ========== 等待所有任务完成 ==========
            try:
                await asyncio.gather(send_task, receive_task, playback_task)
            except (websockets.exceptions.ConnectionClosed, 
                    websockets.exceptions.WebSocketException,
                    OSError) as e:
                # 连接相关异常，向上抛出以便外层代码重连
                logger.warning(f"TTS: 处理任务时连接断开: {e}")
                # 取消未完成的任务（清理资源）
                if not send_task.done():
                    send_task.cancel()
                if not receive_task.done():
                    receive_task.cancel()
                if not playback_task.done():
                    playback_task.cancel()
                # 等待任务取消完成（确保资源清理完成）
                await asyncio.gather(send_task, receive_task, playback_task, return_exceptions=True)
                raise  # 重新抛出连接异常
            except Exception as e:
                # 其他异常，记录错误但继续处理
                error_msg = str(e)
                # 检查是否是连接相关的错误（如 ping timeout）
                if "keepalive ping timeout" in error_msg or "connection" in error_msg.lower():
                    logger.warning(f"TTS: 处理任务时检测到连接问题: {e}")
                    # 取消未完成的任务（清理资源）
                    if not send_task.done():
                        send_task.cancel()
                    if not receive_task.done():
                        receive_task.cancel()
                    if not playback_task.done():
                        playback_task.cancel()
                    # 等待任务取消完成（确保资源清理完成）
                    await asyncio.gather(send_task, receive_task, playback_task, return_exceptions=True)
                    # 将异常转换为连接异常，以便外层代码重连
                    raise websockets.exceptions.WebSocketException(f"连接问题: {e}") from e
                else:
                    logger.error(f"TTS: 处理任务失败: {e}")
                    # 取消未完成的任务（清理资源）
                    if not send_task.done():
                        send_task.cancel()
                    if not receive_task.done():
                        receive_task.cancel()
                    if not playback_task.done():
                        playback_task.cancel()
                    # 等待任务取消完成（确保资源清理完成）
                    await asyncio.gather(send_task, receive_task, playback_task, return_exceptions=True)
                    # 对于其他异常，也向上抛出以便外层代码处理
                    raise

    def put_tts_text(self, text: str):
        """
        将文本放入 TTS 队列
        
        将待转换为语音的文本放入队列，TTS处理器会从队列中取出并处理。
        这是一个线程安全的方法，可以在任何线程中调用。
        
        Args:
            text: 要转换为语音的文本内容
        """
        # 将文本和chat_id一起放入队列（字典格式）
        self.tts_queue.put({"text": text, "chat_id": self.tts_chat_id})
        self.tts_chat_id += 1  # 递增chat_id计数器
        
        # 如果有事件对象，触发事件通知TTS处理器（唤醒等待的处理器）
        if self.tts_queue_event:
            self.tts_queue_event.set()
        # logger.info(f"TTS: 文本已放入队列: {text}, chat_id: {self.tts_chat_id - 1}")

    def stop_tts_processor(self):
        """
        停止 TTS 处理器
        
        设置停止标志并唤醒TTS处理器，使其退出主循环并关闭连接。
        这是一个线程安全的方法，可以在任何线程中调用。
        """
        self.tts_running = False  # 设置停止标志
        if self.tts_queue_event:
            self.tts_queue_event.set()  # 唤醒等待的处理器以便退出（触发事件让处理器检查停止标志）

