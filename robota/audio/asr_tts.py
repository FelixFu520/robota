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
    def __init__(self):
        # 音频设备
        self.audio_device = AudioDevice(
            sample_rate=16000,  # 采样率16kHz，符合ASR服务要求
            channels=1,         # 单声道
            chunk_size=1024,    # 每次读取1024个样本
            enable_aec=True    # 实时ASR测试时可以关闭回声消除
        )
        # 启动音频设备
        self.audio_device.start_streams()

        # ASR客户端配置
        self.asr_url = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
        self.asr_seg_duration = 200
        self.asr_queue = Queue()
        self.asr_queue_event = None  # 用于通知有新结果放入队列，在异步上下文中创建
        self.asr_chat_id = 1  # ASR chat_id 计数器

        # TTS客户端配置
        self.tts_queue = Queue()
        self.tts_queue_event = None  # 用于通知有新文本放入队列，在异步上下文中创建
        self.tts_chat_id = 1  # TTS chat_id 计数器
        self.tts_appid = os.getenv("ASR_APP_KEY")
        self.tts_access_token = os.getenv("ASR_ACCESS_KEY")
        self.tts_resource_id = "seed-tts-2.0"
        self.tts_voice_type = "zh_male_m191_uranus_bigtts"
        self.tts_encoding = "mp3"
        self.tts_endpoint = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
        self.tts_sample_rate = 16000  # TTS播放采样率
        self.tts_running = False  # TTS处理是否正在运行

    async def _realtime_audio_generator(self, audio_device: AudioDevice, duration_seconds: int = None, chunk_duration_ms: int = 200):
        """
        实时音频生成器，从AudioDevice录音并产生音频片段
        
        Args:
            audio_device: AudioDevice实例
            duration_seconds: 录音时长(秒)，None表示无限录音
            chunk_duration_ms: 每个音频片段的时长(毫秒)
            
        Yields:
            音频数据片段(bytes)
        """
        # 获取音频设备参数
        sample_rate = audio_device.sample_rate
        channels = audio_device.channels
        chunk_size = audio_device.chunk_size
        
        # 计算每个chunk的字节数
        samples_per_chunk = chunk_size
        bytes_per_sample = 2  # int16格式，每个样本2字节
        bytes_per_chunk = samples_per_chunk * channels * bytes_per_sample
        
        # 计算需要累积多少个chunk才能达到目标时长
        # 目标字节数 = 采样率 × 声道数 × 每样本字节数 × 时长(秒)
        target_bytes = int(sample_rate * channels * bytes_per_sample * chunk_duration_ms / 1000)
        chunks_needed = max(1, target_bytes // bytes_per_chunk)
        
        # logger.info(f"实时录音配置: 采样率={sample_rate}, 块大小={chunk_size}, 每{chunk_duration_ms}ms发送一次")
        # logger.info(f"每次发送需要累积 {chunks_needed} 个chunk, 约 {target_bytes} 字节")
        
        # 初始化录音状态
        start_time = asyncio.get_event_loop().time()
        accumulated_data = b''  # 累积的音频数据
        chunk_count = 0         # 已累积的chunk数量
        
        try:
            while True:
                # 检查是否达到录音时长限制
                if duration_seconds is not None:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed >= duration_seconds:
                        # logger.info(f"录音时长达到 {duration_seconds} 秒，停止录音")
                        break
                
                # 异步获取录音数据，超时时间1秒
                audio_chunk = await audio_device.async_get_recorded_data(timeout=1.0)
                if audio_chunk is None:
                    continue
                
                # 累积音频数据
                accumulated_data += audio_chunk
                chunk_count += 1
                
                # 当累积足够的数据后，转换为WAV格式并发送
                if chunk_count >= chunks_needed:
                    # 将PCM数据转换为WAV格式
                    wav_data = self._create_wav_chunk(accumulated_data, sample_rate, channels)
                    yield wav_data
                    
                    # 重置累积状态，准备下一批数据
                    accumulated_data = b''
                    chunk_count = 0
            
            # 发送剩余的音频数据（如果有）
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
        
        Args:
            pcm_data: PCM原始音频数据(bytes)
            sample_rate: 采样率(Hz)
            channels: 声道数
            
        Returns:
            WAV格式的音频数据(bytes)
        """
        # 创建内存缓冲区
        buffer = io.BytesIO()
        
        # 写入WAV文件头和数据
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit，每个样本2字节
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        
        # 返回完整的WAV文件数据
        return buffer.getvalue()

    async def start_realtime_asr(self, duration_seconds: int = None, silence_timeout_ms: int = 800):
        """
        启动实时ASR识别
        
        Args:
            duration_seconds: 录音时长(秒)，None表示无限录音
            silence_timeout_ms: 静音超时时间(毫秒)，当超过此时间没有识别到文字时，将结果放入队列
        """
        # 在异步上下文中创建事件
        if self.asr_queue_event is None:
            self.asr_queue_event = asyncio.Event()
        
        try:
            # 创建实时音频生成器
            audio_stream = self._realtime_audio_generator(
                self.audio_device,
                duration_seconds=duration_seconds,
                chunk_duration_ms=200
            )
            
            # 创建ASR客户端并进行实时识别
            async with AsrWsClient(self.asr_url, self.asr_seg_duration) as asr_client:
                # 跟踪状态
                last_text_time = None  # 最后一次收到识别文本的时间
                accumulated_text = ""  # 累积的识别文本
                
                # 创建超时检查任务
                async def check_timeout():
                    """检查超时并处理队列"""
                    nonlocal last_text_time, accumulated_text
                    while True:
                        await asyncio.sleep(0.1)  # 每100ms检查一次
                        current_time = asyncio.get_event_loop().time()
                        
                        # 如果超过800ms没有收到新文本，且有累积文本，则放入队列
                        if last_text_time is not None and accumulated_text:
                            elapsed_ms = (current_time - last_text_time) * 1000
                            if elapsed_ms >= silence_timeout_ms:
                                # 将识别结果放入队列
                                result = {
                                    "text": accumulated_text,
                                    "chat_id": self.asr_chat_id
                                }
                                self.asr_queue.put(result)
                                self.asr_chat_id += 1

                                # 通知有新结果
                                self.asr_queue_event.set()
                                # logger.info(f"超时({elapsed_ms:.0f}ms)未识别到新文字，将结果放入队列: {accumulated_text}, chat_id: {self.chat_id}")
                                
                                # 重置状态
                                accumulated_text = ""
                                last_text_time = None
                
                # 启动超时检查任务
                timeout_task = asyncio.create_task(check_timeout())
                
                try:
                    # 处理ASR响应
                    async for response in asr_client.execute_stream(audio_stream):
                        # 解析响应数据
                        resp_dict = response.to_dict()
                        
                        # 检查响应中是否包含识别结果
                        if resp_dict.get('payload_msg') and 'result' in resp_dict['payload_msg']:
                            result = resp_dict['payload_msg']['result']
                            
                            # 如果包含识别文本
                            if 'text' in result:
                                text = result['text']
                                current_time = asyncio.get_event_loop().time()
                                
                                # 更新累积文本（使用最新的完整文本）
                                accumulated_text = text
                                last_text_time = current_time
                                
                                # logger.debug(f"收到识别文本: {text}")
                
                except Exception as e:
                    logger.error(f"实时ASR处理失败: {e}")
                    raise
                finally:
                    # 取消超时检查任务
                    timeout_task.cancel()
                    try:
                        await timeout_task
                    except asyncio.CancelledError:
                        pass
                    
                    # 如果还有未处理的文本，放入队列
                    if accumulated_text:
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
        
        Args:
            mp3_data: MP3 格式的音频数据（字节流）
            target_sample_rate: 目标采样率，默认 16000 Hz
        
        Returns:
            PCM 格式的音频数据（字节流），如果转换失败则返回空字节流
        """
        try:
            # 使用 pydub 从字节流加载 MP3 数据
            audio = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
            
            # 转换为单声道
            audio = audio.set_channels(1)
            
            # 转换采样率到目标采样率
            audio = audio.set_frame_rate(target_sample_rate)
            
            # 转换为 16 位 PCM（2 字节 = 16 位）
            audio = audio.set_sample_width(2)
            
            # 获取原始 PCM 数据
            pcm_data = audio.raw_data
            
            return pcm_data
        except Exception as e:
            logger.error(f"转换MP3到PCM失败: {e}")
            return b""

    async def start_tts_processor(self):
        """
        启动 TTS 处理器，从队列中读取文本并转换为语音播放
        
        该方法会持续运行，从 tts_queue 中读取文本，调用 TTS 服务转换为语音并播放
        """
        # 在异步上下文中创建事件
        if self.tts_queue_event is None:
            self.tts_queue_event = asyncio.Event()
        
        self.tts_running = True
        
        try:
            # 构建 WebSocket 连接请求头
            headers = {
                "X-Api-App-Key": self.tts_appid,
                "X-Api-Access-Key": self.tts_access_token,
                "X-Api-Resource-Id": (self.tts_resource_id),
                "X-Api-Connect-Id": str(uuid.uuid4()),  # 生成唯一的连接 ID
            }

            # 连接到 WebSocket 服务器
            # logger.info(f"TTS: 连接到 {self.tts_endpoint}")
            websocket = await websockets.connect(
                self.tts_endpoint, additional_headers=headers, max_size=10 * 1024 * 1024
            )
            # logger.info(
            #     f"TTS: 已连接到服务器, Logid: {websocket.response.headers.get('x-tt-logid', 'N/A')}",
            # )

            try:
                # 启动连接
                await start_connection(websocket)
                await wait_for_event(
                    websocket, MsgType.FullServerResponse, EventType.ConnectionStarted
                )

                # 持续处理队列中的文本
                while self.tts_running:
                    try:
                        # 先检查队列，如果有数据立即处理（减少延迟）
                        # 批量处理队列中的所有文本，确保及时响应
                        processed_any = False
                        while not self.tts_queue.empty():
                            text_data = self.tts_queue.get_nowait()
                            if text_data:
                                text = text_data.get("text", "") if isinstance(text_data, dict) else str(text_data)
                                if text:
                                    # logger.info(f"TTS: 处理文本: {text}")
                                    await self._process_tts_text(websocket, text)
                                    processed_any = True
                        
                        # 如果已经处理了文本，立即继续循环检查下一个（保持高响应性）
                        if processed_any:
                            continue
                        
                        # 队列为空时，等待事件通知（有新文本加入时会立即被唤醒）
                        try:
                            await asyncio.wait_for(self.tts_queue_event.wait(), timeout=1.0)
                            self.tts_queue_event.clear()
                            # 事件被触发后，立即检查队列并处理（可能有多条文本）
                            # 上面的循环会处理所有队列中的文本
                        except asyncio.TimeoutError:
                            # 超时后继续循环检查（保持响应性）
                            continue
                    except Exception as e:
                        logger.error(f"TTS处理队列文本失败: {e}")
                        await asyncio.sleep(0.1)  # 出错后稍等再继续

            finally:
                # 结束连接
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
            logger.error(f"TTS处理器失败: {e}")
            raise
        finally:
            self.tts_running = False

    async def _process_tts_text(self, websocket, text: str):
        """
        处理单个文本的 TTS 转换和播放
        实现一边接收数据一边播放的流式处理
        
        Args:
            websocket: WebSocket 连接
            text: 要转换的文本
        """
        # 按句号分割文本，逐句处理
        sentences = text.split("。")
        
        for sentence in sentences:
            if not sentence.strip():
                continue

            # 构建基础请求参数
            base_request = {
                "user": {
                    "uid": str(uuid.uuid4()),
                },
                "namespace": "BidirectionalTTS",
                "req_params": {
                    "speaker": self.tts_voice_type,
                    "audio_params": {
                        "format": self.tts_encoding,
                        "sample_rate": 24000,  # 服务器端采样率
                        "enable_timestamp": True,
                    },
                    "additions": json.dumps(
                        {
                            "disable_markdown_filter": False,
                        }
                    ),
                },
            }

            # 启动会话
            start_session_request = copy.deepcopy(base_request)
            start_session_request["event"] = EventType.StartSession
            session_id = str(uuid.uuid4())
            await start_session(
                websocket, json.dumps(start_session_request).encode(), session_id
            )
            await wait_for_event(
                websocket, MsgType.FullServerResponse, EventType.SessionStarted
            )

            # 逐字符发送文本（异步函数）
            async def send_chars():
                for char in sentence:
                    synthesis_request = copy.deepcopy(base_request)
                    synthesis_request["event"] = EventType.TaskRequest
                    synthesis_request["req_params"]["text"] = char
                    await task_request(
                        websocket, json.dumps(synthesis_request).encode(), session_id
                    )
                    await asyncio.sleep(0.005)  # 字符间延迟 5ms

                # 发送会话结束请求
                await finish_session(websocket, session_id)

            # 在后台任务中开始发送字符
            send_task = asyncio.create_task(send_chars())

            # 创建异步队列用于存储接收到的音频数据
            audio_queue = asyncio.Queue()
            session_finished = False
            
            # 接收音频数据的任务（生产者）
            async def receive_audio_task():
                """持续接收音频数据并放入队列"""
                nonlocal session_finished
                try:
                    while not session_finished:
                        msg = await receive_message(websocket)
                        
                        if msg.type == MsgType.FullServerResponse:
                            if msg.event == EventType.SessionFinished:
                                # 会话结束，发送结束标记
                                await audio_queue.put(None)  # None 表示会话结束
                                session_finished = True
                                break
                        elif msg.type == MsgType.AudioOnlyServer:
                            # 处理音频数据消息
                            if msg.payload:
                                # 将音频数据放入队列
                                await audio_queue.put(msg.payload)
                        else:
                            # 未知消息类型，记录警告但继续处理
                            logger.warning(f"TTS: 收到未知消息类型: {msg.type}")
                except Exception as e:
                    logger.error(f"TTS: 接收音频数据任务失败: {e}")
                    await audio_queue.put(None)  # 出错时也发送结束标记
            
            # 播放音频数据的任务（消费者）
            async def playback_audio_task():
                """持续从队列取出数据，转换并播放"""
                mp3_buffer = bytearray()
                playback_started = False
                min_buffer_size = 2048  # 最小 MP3 缓冲大小（约 2KB，降低延迟）
                mp3_convert_threshold = 2048  # MP3 转换阈值（降低延迟）
                
                try:
                    while True:
                        # 从队列中获取音频数据，设置超时避免无限等待
                        try:
                            audio_data = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            # 超时但会话可能还在进行，继续等待
                            continue
                        
                        # 检查是否收到结束标记
                        if audio_data is None:
                            # 会话结束，转换并播放剩余的 MP3 数据
                            if len(mp3_buffer) > 0:
                                logger.info(f"TTS: 会话结束,转换剩余MP3数据 {len(mp3_buffer)} bytes")
                                if self.tts_encoding == "mp3":
                                    pcm_data = self._convert_mp3_to_pcm(bytes(mp3_buffer), self.tts_sample_rate)
                                    if pcm_data:
                                        self.audio_device.put_playback_data(pcm_data)
                                        logger.info(f"TTS: 添加剩余PCM数据: {len(pcm_data)} bytes")
                                mp3_buffer.clear()
                            break
                        
                        # 处理音频数据
                        if self.tts_encoding == "mp3":
                            # MP3 格式：累积 MP3 数据
                            mp3_buffer.extend(audio_data)
                            
                            # 检查是否达到最小缓冲大小，开始播放
                            if not playback_started and len(mp3_buffer) >= min_buffer_size:
                                logger.info(f"TTS: MP3缓冲已满({len(mp3_buffer)} bytes),开始播放")
                                playback_started = True
                            
                            # 批量转换策略：累积足够的 MP3 数据再转换
                            if playback_started and len(mp3_buffer) >= mp3_convert_threshold:
                                pcm_data = self._convert_mp3_to_pcm(bytes(mp3_buffer), self.tts_sample_rate)
                                if pcm_data:
                                    self.audio_device.put_playback_data(pcm_data)
                                    logger.info(f"TTS: 转换并添加PCM: {len(pcm_data)} bytes (来自{len(mp3_buffer)} MP3)")
                                    mp3_buffer.clear()
                                else:
                                    logger.warning(f"TTS: MP3转PCM失败")
                                    mp3_buffer.clear()
                                    
                        elif self.tts_encoding == "pcm":
                            # PCM 格式：直接放入播放队列
                            self.audio_device.put_playback_data(audio_data)
                            if not playback_started:
                                playback_started = True
                            logger.info(f"TTS: 添加PCM数据: {len(audio_data)} bytes")
                            
                except Exception as e:
                    logger.error(f"TTS: 播放音频数据任务失败: {e}")
            
            # 启动接收和播放任务（并行运行）
            receive_task = asyncio.create_task(receive_audio_task())
            playback_task = asyncio.create_task(playback_audio_task())
            
            # 等待所有任务完成
            try:
                await asyncio.gather(send_task, receive_task, playback_task)
            except Exception as e:
                logger.error(f"TTS: 处理任务失败: {e}")
                # 取消未完成的任务
                if not send_task.done():
                    send_task.cancel()
                if not receive_task.done():
                    receive_task.cancel()
                if not playback_task.done():
                    playback_task.cancel()
                # 等待任务取消完成
                await asyncio.gather(send_task, receive_task, playback_task, return_exceptions=True)

    def put_tts_text(self, text: str):
        """
        将文本放入 TTS 队列
        
        Args:
            text: 要转换为语音的文本
        """
        self.tts_queue.put({"text": text, "chat_id": self.tts_chat_id})
        self.tts_chat_id += 1
        if self.tts_queue_event:
            self.tts_queue_event.set()
        # logger.info(f"TTS: 文本已放入队列: {text}, chat_id: {self.tts_chat_id - 1}")

    def stop_tts_processor(self):
        """
        停止 TTS 处理器
        """
        self.tts_running = False
        if self.tts_queue_event:
            self.tts_queue_event.set()  # 唤醒等待的处理器以便退出

