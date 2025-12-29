import asyncio
import io
import wave
from queue import Queue
from robota.utils.logging import default_logger as logger
from robota.audio.audio_device import AudioDevice
from robota.audio.volcengine_doubao_asr import AsrWsClient

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

        self.chat_id = 1


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
                                    "chat_id": self.chat_id
                                }
                                self.asr_queue.put(result)
                                self.chat_id += 1

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
                            "chat_id": self.chat_id
                        }
                        self.asr_queue.put(result)
                        # 通知有新结果
                        self.asr_queue_event.set()
                        # logger.info(f"识别结束，将剩余结果放入队列: {accumulated_text}, chat_id: {self.chat_id}")
                        
        except Exception as e:
            logger.error(f"启动实时ASR失败: {e}")
            raise

