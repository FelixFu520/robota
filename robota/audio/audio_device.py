import pyaudio
import numpy as np
import queue
import asyncio
from collections import deque

from robota.utils.logging import default_logger


class AudioEchoCancellation:
    """改进的回声消除实现 - 基于双讲检测的NLMS + 多重抑制"""
    
    def __init__(self, filter_length=512, step_size=0.01, noise_gate_threshold=500, enable=True):
        """
        初始化回声消除器
        
        Args:
            filter_length: 自适应滤波器长度(越长效果越好但计算量越大)
            step_size: NLMS算法的步长(学习率)
            noise_gate_threshold: 噪声门限,低于此值的信号被视为静音
            enable: 是否启用回声消除
        """
        self.filter_length = filter_length
        self.step_size = step_size
        self.noise_gate_threshold = noise_gate_threshold
        self.enable = enable
        
        # 自适应滤波器系数(NLMS)
        self.filter_coeffs = np.zeros(filter_length, dtype=np.float32)
        
        # 参考信号缓冲区(播放的音频) - 使用deque实现循环缓冲
        self.reference_buffer_size = filter_length * 4
        self.reference_buffer = deque(maxlen=self.reference_buffer_size)
        
        # 延迟估计相关
        self.estimated_delay = 0
        self.delay_search_range = 512
        self.delay_update_counter = 0
        self.delay_update_interval = 100  # 每N帧更新一次延迟估计
        
        # 双讲检测
        self.double_talk_threshold = 0.5
        self.is_double_talk = False
        
        # VAD (语音活动检测)
        self.vad_threshold = 0.02
        self.has_near_speech = False  # 近端语音(用户)
        self.has_far_speech = False   # 远端语音(播放)
        
        # 能量平滑
        self.near_energy_smooth = 0.0
        self.far_energy_smooth = 0.0
        self.energy_alpha = 0.8
        
        # 平滑因子用于归一化
        self.epsilon = 1e-8
        
        # 双重抑制：频域和时域
        self.enable_spectral_subtraction = True
        
        # 统计信息
        self.total_processed = 0
        self.echo_reduction_ratio = 0.0
        
        # 回声抑制因子
        self.suppression_factor = 0.5
        self.residual_suppression = 0.3  # 残留回声抑制
        
    def add_playback_reference(self, audio_data):
        """
        添加播放的参考信号到缓冲区
        
        Args:
            audio_data: 播放的音频数据
        """
        if not self.enable:
            return
            
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # 归一化到 [-1, 1]
        audio_array = audio_array / 32768.0
        
        # 添加到循环缓冲区
        self.reference_buffer.extend(audio_array)
    
    def _estimate_delay(self, recorded, reference):
        """
        使用互相关估计延迟
        
        Returns:
            估计的延迟(采样点数)
        """
        if len(reference) < self.delay_search_range * 2:
            return 0
        
        # 取最后的一段参考信号用于互相关
        ref_segment = reference[-self.delay_search_range*2:]
        
        # 计算互相关
        if len(recorded) < len(ref_segment):
            return 0
            
        correlation = np.correlate(recorded, ref_segment, mode='valid')
        
        # 找到最大相关的位置
        if len(correlation) > 0:
            delay = np.argmax(np.abs(correlation))
            return min(delay, self.delay_search_range)
        return 0
    
    def _detect_double_talk(self, recorded_energy, echo_estimate_energy):
        """
        双讲检测 - 判断是否同时存在真实语音和回声
        
        Returns:
            True: 双讲状态(有真实语音), False: 仅有回声
        """
        if echo_estimate_energy < self.epsilon:
            return True  # 没有回声，认为是真实语音
        
        # 计算能量比
        ratio = recorded_energy / (echo_estimate_energy + self.epsilon)
        
        # 如果录音能量显著大于回声估计,说明有真实语音
        return ratio > (1.0 + self.double_talk_threshold)
    
    def _voice_activity_detection(self, audio, reference):
        """
        语音活动检测
        
        Updates:
            self.has_near_speech: 是否有近端语音(用户说话)
            self.has_far_speech: 是否有远端语音(播放音频)
        """
        # 计算近端能量(录音)
        near_energy = np.mean(audio ** 2)
        self.near_energy_smooth = (self.energy_alpha * self.near_energy_smooth + 
                                   (1 - self.energy_alpha) * near_energy)
        
        # 计算远端能量(播放)
        far_energy = np.mean(reference ** 2)
        self.far_energy_smooth = (self.energy_alpha * self.far_energy_smooth + 
                                  (1 - self.energy_alpha) * far_energy)
        
        # VAD判断
        self.has_near_speech = self.near_energy_smooth > self.vad_threshold
        self.has_far_speech = self.far_energy_smooth > self.vad_threshold * 0.5
    
    def _nlms_filter(self, recorded, reference):
        """
        改进的NLMS (Normalized Least Mean Squares) 自适应滤波
        
        Args:
            recorded: 录制的音频样本
            reference: 参考信号(播放的音频)
            
        Returns:
            (cleaned_signal, echo_estimate)
        """
        output = np.zeros_like(recorded)
        echo_estimates = np.zeros_like(recorded)
        
        # 确保参考信号足够长
        if len(reference) < self.filter_length:
            return recorded, np.zeros_like(recorded)
        
        # 根据延迟估计对齐参考信号
        if self.estimated_delay > 0 and len(reference) > self.filter_length + self.estimated_delay:
            ref_aligned = reference[-(self.filter_length + self.estimated_delay):-self.estimated_delay]
        else:
            ref_aligned = reference[-self.filter_length:]
        
        for i in range(len(recorded)):
            # 获取当前的参考信号窗口
            if i < self.filter_length:
                if len(ref_aligned) >= self.filter_length - i:
                    ref_window = np.concatenate([
                        ref_aligned[-(self.filter_length-i):],
                        reference[:i] if i > 0 else np.array([])
                    ])
                else:
                    ref_window = ref_aligned
                    if len(ref_window) < self.filter_length:
                        ref_window = np.pad(ref_window, (self.filter_length - len(ref_window), 0), 'constant')
            else:
                if len(reference) >= i:
                    ref_window = reference[i-self.filter_length:i]
                else:
                    ref_window = ref_aligned
            
            if len(ref_window) < self.filter_length:
                output[i] = recorded[i]
                echo_estimates[i] = 0
                continue
            
            # 估计回声
            echo_estimate = np.dot(self.filter_coeffs, ref_window)
            echo_estimates[i] = echo_estimate
            
            # 计算误差(录音减去回声估计 = 干净信号)
            error = recorded[i] - echo_estimate
            output[i] = error
            
            # 双讲检测 - 只在非双讲时更新滤波器系数
            if not self.is_double_talk:
                # NLMS更新
                power = np.dot(ref_window, ref_window) + self.epsilon
                self.filter_coeffs += (self.step_size / power) * error * ref_window
        
        return output, echo_estimates
    
    def _residual_echo_suppression(self, audio, echo_estimate):
        """
        残留回声抑制 - 使用Wiener滤波思想
        
        Args:
            audio: NLMS输出的信号
            echo_estimate: 回声估计
        
        Returns:
            进一步处理的信号
        """
        # 计算残留回声的估计
        residual_estimate = echo_estimate * self.residual_suppression
        
        # 使用软抑制
        audio_energy = np.abs(audio)
        residual_energy = np.abs(residual_estimate)
        
        # 计算增益
        gain = np.maximum(
            (audio_energy - residual_energy) / (audio_energy + self.epsilon),
            0.1  # 最小增益,避免完全静音
        )
        
        return audio * gain
    
    def _spectral_subtraction(self, audio, chunk_size=512):
        """
        改进的频域谱减法进一步抑制残留回声
        
        Args:
            audio: 时域音频信号
            chunk_size: FFT窗口大小
            
        Returns:
            处理后的音频
        """
        if len(audio) < chunk_size:
            return audio
        
        # 分帧处理
        num_chunks = len(audio) // chunk_size
        output = np.zeros_like(audio)
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = audio[start:end]
            
            # FFT
            fft_data = np.fft.rfft(chunk)
            magnitude = np.abs(fft_data)
            phase = np.angle(fft_data)
            
            # 自适应噪声估计
            noise_floor = np.percentile(magnitude, 10)
            
            # Over-subtraction with spectral floor
            alpha = 2.0  # over-subtraction factor
            beta = 0.02  # spectral floor
            magnitude_cleaned = np.maximum(
                magnitude - alpha * noise_floor,
                beta * magnitude
            )
            
            # 重建
            fft_cleaned = magnitude_cleaned * np.exp(1j * phase)
            chunk_cleaned = np.fft.irfft(fft_cleaned, chunk_size)
            
            output[start:end] = chunk_cleaned
        
        # 处理剩余部分
        if num_chunks * chunk_size < len(audio):
            output[num_chunks * chunk_size:] = audio[num_chunks * chunk_size:]
        
        return output
    
    def _apply_noise_gate(self, audio, threshold):
        """
        改进的噪声门 - 使用平滑的gain曲线
        
        Args:
            audio: 音频信号
            threshold: 门限值
            
        Returns:
            处理后的音频
        """
        # 计算音频能量
        energy = np.abs(audio)
        
        # 平滑能量曲线
        window_size = 50
        if len(energy) >= window_size:
            # 使用汉明窗进行平滑
            kernel = np.hamming(window_size) / np.sum(np.hamming(window_size))
            energy_smoothed = np.convolve(energy, kernel, mode='same')
        else:
            energy_smoothed = energy
        
        # 软门限 - 使用sigmoid函数实现平滑过渡
        gate = 1.0 / (1.0 + np.exp(-10 * (energy_smoothed - threshold)))
        
        return audio * gate
    
    def process_recorded_audio(self, audio_data):
        """
        处理录制的音频,消除回声
        
        Args:
            audio_data: 录制的原始音频数据
            
        Returns:
            处理后的音频数据
        """
        # 如果禁用,直接返回
        if not self.enable:
            return audio_data
            
        # 转换为numpy数组并归一化
        recorded = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        recorded_normalized = recorded / 32768.0
        
        # 保存原始能量用于统计
        original_energy = np.sum(recorded_normalized ** 2)
        
        # 如果参考缓冲区还没有数据,直接返回
        if len(self.reference_buffer) < self.filter_length:
            return audio_data
        
        # 获取参考信号数组
        reference = np.array(self.reference_buffer, dtype=np.float32)
        
        # VAD检测
        self._voice_activity_detection(recorded_normalized, reference[-self.filter_length:])
        
        # 定期更新延迟估计
        self.delay_update_counter += 1
        if self.delay_update_counter >= self.delay_update_interval:
            self.delay_update_counter = 0
            if self.has_far_speech and len(reference) >= self.delay_search_range * 2:
                self.estimated_delay = self._estimate_delay(
                    recorded_normalized, 
                    reference[-self.delay_search_range*2:]
                )
        
        # 步骤1: NLMS自适应滤波
        cleaned, echo_estimate = self._nlms_filter(recorded_normalized, reference)
        
        # 计算回声能量
        echo_energy = np.sum(echo_estimate ** 2)
        cleaned_energy = np.sum(cleaned ** 2)
        
        # 双讲检测
        self.is_double_talk = self._detect_double_talk(original_energy, echo_energy)
        
        # 步骤2: 根据场景选择处理策略
        if not self.has_far_speech:
            # 没有播放,直接使用录音
            output = recorded_normalized
        elif not self.has_near_speech and self.has_far_speech:
            # 只有播放没有语音,强力抑制
            output = cleaned * 0.1
        elif self.is_double_talk:
            # 双讲状态,保守处理以保留语音
            output = cleaned
        else:
            # 只有回声,正常处理
            # 步骤3: 残留回声抑制
            output = self._residual_echo_suppression(cleaned, echo_estimate)
            
            # 步骤4: 频域谱减法(仅在有明显回声时)
            if self.enable_spectral_subtraction and len(output) >= 256:
                ref_energy = np.sum(reference[-self.filter_length:] ** 2)
                if ref_energy > self.epsilon * 100:
                    output = self._spectral_subtraction(output)
        
        # 步骤5: 噪声门
        output = self._apply_noise_gate(output, self.noise_gate_threshold / 32768.0)
        
        # 更新统计
        output_energy = np.sum(output ** 2)
        if original_energy > self.epsilon:
            reduction = 1.0 - (output_energy / original_energy)
            self.echo_reduction_ratio = 0.95 * self.echo_reduction_ratio + 0.05 * reduction
        
        self.total_processed += 1
        
        # 转换回int16
        output = np.clip(output * 32768.0, -32767, 32767)
        return output.astype(np.int16).tobytes()
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'total_processed': self.total_processed,
            'echo_reduction_ratio': self.echo_reduction_ratio * 100,
            'estimated_delay': self.estimated_delay,
            'is_double_talk': self.is_double_talk,
            'has_near_speech': self.has_near_speech,
            'has_far_speech': self.has_far_speech,
        }


class AudioDevice:
    """音频设备管理类"""
    
    def __init__(self, input_device_index=None, output_device_index=None, sample_rate=16000, channels=1, chunk_size=1024, enable_aec=True):
        """
        初始化音频设备
        
        Args:
            input_device_index: 输入设备索引,None表示使用默认输入设备
            output_device_index: 输出设备索引,None表示使用默认输出设备
            sample_rate: 采样率
            channels: 声道数
            chunk_size: 音频块大小
            enable_aec: 是否启用回声消除
        """
        self.p = pyaudio.PyAudio()
        
        # 分别获取输入和输出设备
        self.input_device_index = input_device_index if input_device_index is not None else self._get_default_input_device()
        self.output_device_index = output_device_index if output_device_index is not None else self._get_default_output_device()
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        
        # 获取设备信息
        self.input_device_info = self.p.get_device_info_by_index(self.input_device_index)
        self.output_device_info = self.p.get_device_info_by_index(self.output_device_index)
        self._print_device_info()
        
        # 回声消除器(使用优化的参数 - 平衡回声消除和语音保留)
        if enable_aec:
            self.aec = AudioEchoCancellation(
                filter_length=2048,       # 适中的滤波器长度
                step_size=0.005,          # 适中的学习率
                noise_gate_threshold=500, # 较低的噪声门限,避免过度抑制
                enable=True
            )
        else:
            self.aec = None
        
        # 音频流
        self.input_stream = None
        self.output_stream = None
        
        # 播放队列和录音队列
        self.playback_queue = queue.Queue()
        self.recording_queue = queue.Queue()
        
        # 控制标志
        self.is_running = False
        
    def _get_default_input_device(self):
        """获取系统默认输入设备"""
        try:
            default_device = self.p.get_default_input_device_info()
            return default_device['index']
        except Exception as e:
            default_logger.warning(f"无法获取默认输入设备,使用设备0: {e}")
            return 0
    
    def _get_default_output_device(self):
        """获取系统默认输出设备"""
        try:
            default_device = self.p.get_default_output_device_info()
            return default_device['index']
        except Exception as e:
            default_logger.error(f"无法获取默认输出设备,使用设备0: {e}")
            return 0
    
    def _print_device_info(self):
        """打印设备信息"""
        default_logger.info("="*60)
        default_logger.info("音频设备信息:")
        default_logger.info(f"输入设备: {self.input_device_info['name']}")
        default_logger.info(f"设备索引: {self.input_device_index}")
        default_logger.info(f"最大输入通道: {self.input_device_info['maxInputChannels']}")
        default_logger.info(f"默认采样率: {self.input_device_info['defaultSampleRate']}")
        default_logger.info(f"输出设备: {self.output_device_info['name']}")
        default_logger.info(f"设备索引: {self.output_device_index}")
        default_logger.info(f"最大输出通道: {self.output_device_info['maxOutputChannels']}")
        default_logger.info(f"默认采样率: {self.output_device_info['defaultSampleRate']}")
        default_logger.info(f"音频配置: {self.sample_rate}")
        default_logger.info(f"声道数: {self.channels}")
        default_logger.info(f"块大小: {self.chunk_size}")
        default_logger.info("="*60)
    
    def _input_callback(self, in_data, frame_count, time_info, status):
        """录音回调函数"""
        if status:
            default_logger.info(f"输入状态: {status}")
        
        # 应用回声消除(如果启用)
        if self.aec is not None:
            cleaned_data = self.aec.process_recorded_audio(in_data)
        else:
            cleaned_data = in_data
        
        # 将处理后的音频放入队列
        self.recording_queue.put(cleaned_data)
        
        return (None, pyaudio.paContinue)
    
    def _output_callback(self, in_data, frame_count, time_info, status):
        """播放回调函数"""
        if status:
            default_logger.info(f"输出状态: {status}")
        
        try:
            # 从队列获取要播放的数据
            data = self.playback_queue.get_nowait()
            
            # 添加到回声消除器的参考缓冲(如果启用)
            if self.aec is not None:
                self.aec.add_playback_reference(data)
            
            return (data, pyaudio.paContinue)
        except queue.Empty:
            # 如果队列为空,播放静音
            silence = b'\x00' * (frame_count * self.channels * 2)  # 2 bytes per sample (int16)
            return (silence, pyaudio.paContinue)
    
    def start_streams(self):
        """启动音频输入输出流"""
        try:
            # 启动输入流(录音)
            self.input_stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._input_callback
            )
            
            # 启动输出流(播放)
            self.output_stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._output_callback
            )
            
            self.is_running = True
            default_logger.info("音频流已启动")
            
        except Exception as e:
            default_logger.error(f"启动音频流失败, {e}")
            self.is_running = False
            raise
    
    def stop_streams(self):
        """停止音频流"""
        self.is_running = False
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        
        default_logger.info("音频流已停止")
    
    def cleanup(self):
        """清理资源"""
        self.stop_streams()
        self.p.terminate()
        default_logger.info("资源已清理")
    
    def put_playback_data(self, audio_data):
        """
        添加音频数据到播放队列
        
        Args:
            audio_data: 要播放的音频数据(bytes)
        """
        self.playback_queue.put(audio_data)
    
    def get_recorded_data(self, block=True, timeout=None):
        """
        从录音队列获取音频数据
        
        Args:
            block: 是否阻塞等待
            timeout: 超时时间(秒)
            
        Returns:
            录音的音频数据(bytes), 如果队列为空且非阻塞则返回None
        """
        try:
            return self.recording_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def clear_playback_queue(self):
        """清空播放队列"""
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
            except queue.Empty:
                break
    
    def clear_recording_queue(self):
        """清空录音队列"""
        while not self.recording_queue.empty():
            try:
                self.recording_queue.get_nowait()
            except queue.Empty:
                break
    
    def get_recording_queue_size(self):
        """获取录音队列大小"""
        return self.recording_queue.qsize()
    
    def get_playback_queue_size(self):
        """获取播放队列大小"""
        return self.playback_queue.qsize()

