"""
音频录音和播放测试程序,支持回声消除功能

功能特性:
1. 使用同一个音频设备进行录音和播放
2. 集成回声消除(AEC)功能,减少播放声音对录音的干扰
3. 自动选择系统默认音频设备
4. 支持播放随机测试音频
5. 实时音频流处理
"""

import pyaudio
import wave
import numpy as np
import argparse
import threading
import queue
import time
import sys
from collections import deque

from robota.utils.logging import default_logger


class AudioEchoCancellation:
    """高级回声消除实现 - 基于NLMS自适应滤波器"""
    
    def __init__(self, filter_length=512, step_size=0.01, noise_gate_threshold=500):
        """
        初始化回声消除器
        
        Args:
            filter_length: 自适应滤波器长度(越长效果越好但计算量越大)
            step_size: NLMS算法的步长(学习率)
            noise_gate_threshold: 噪声门限,低于此值的信号被视为静音
        """
        self.filter_length = filter_length
        self.step_size = step_size
        self.noise_gate_threshold = noise_gate_threshold
        
        # 自适应滤波器系数(NLMS)
        self.filter_coeffs = np.zeros(filter_length, dtype=np.float32)
        
        # 参考信号缓冲区(播放的音频)
        self.reference_buffer = np.zeros(filter_length, dtype=np.float32)
        
        # 平滑因子用于归一化
        self.epsilon = 1e-6
        
        # 双重抑制：频域和时域
        self.enable_spectral_subtraction = True
        
        # 统计信息
        self.total_processed = 0
        self.echo_reduction_ratio = 0.0
        
    def add_playback_reference(self, audio_data):
        """
        添加播放的参考信号到缓冲区
        
        Args:
            audio_data: 播放的音频数据
        """
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # 归一化到 [-1, 1]
        audio_array = audio_array / 32768.0
        
        # 滚动更新参考缓冲区
        if len(audio_array) >= self.filter_length:
            self.reference_buffer = audio_array[-self.filter_length:]
        else:
            self.reference_buffer = np.roll(self.reference_buffer, -len(audio_array))
            self.reference_buffer[-len(audio_array):] = audio_array
    
    def _nlms_filter(self, recorded, reference):
        """
        NLMS (Normalized Least Mean Squares) 自适应滤波
        
        Args:
            recorded: 录制的音频样本
            reference: 参考信号(播放的音频)
            
        Returns:
            处理后的音频
        """
        output = np.zeros_like(recorded)
        
        for i in range(len(recorded)):
            # 获取当前的参考信号窗口
            if i < self.filter_length:
                ref_window = np.concatenate([
                    self.reference_buffer[-(self.filter_length-i):],
                    reference[:i]
                ])
            else:
                ref_window = reference[i-self.filter_length:i]
            
            # 估计回声
            echo_estimate = np.dot(self.filter_coeffs, ref_window)
            
            # 计算误差(期望的输出)
            error = recorded[i] - echo_estimate
            output[i] = error
            
            # 更新滤波器系数(NLMS)
            power = np.dot(ref_window, ref_window) + self.epsilon
            self.filter_coeffs += (self.step_size / power) * error * ref_window
        
        return output
    
    def _spectral_subtraction(self, audio):
        """
        频域谱减法进一步抑制残留回声
        
        Args:
            audio: 时域音频信号
            
        Returns:
            处理后的音频
        """
        if len(audio) < 256:
            return audio
        
        # FFT
        fft_data = np.fft.rfft(audio)
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)
        
        # 估计噪声/回声幅度(使用最小值统计)
        noise_estimate = np.percentile(magnitude, 20)
        
        # 谱减法
        magnitude_cleaned = np.maximum(magnitude - noise_estimate * 1.5, magnitude * 0.1)
        
        # 重建信号
        fft_cleaned = magnitude_cleaned * np.exp(1j * phase)
        audio_cleaned = np.fft.irfft(fft_cleaned, len(audio))
        
        return audio_cleaned
    
    def _apply_noise_gate(self, audio, threshold):
        """
        应用噪声门限
        
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
            energy_smoothed = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
        else:
            energy_smoothed = energy
        
        # 应用门限
        gate = (energy_smoothed > threshold).astype(np.float32)
        
        # 平滑门限曲线避免突变
        gate = np.minimum(gate * 1.2, 1.0)
        
        return audio * gate
    
    def process_recorded_audio(self, audio_data):
        """
        处理录制的音频,消除回声
        
        Args:
            audio_data: 录制的原始音频数据
            
        Returns:
            处理后的音频数据
        """
        # 转换为numpy数组并归一化
        recorded = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        recorded_normalized = recorded / 32768.0
        
        # 保存原始能量用于统计
        original_energy = np.sum(recorded_normalized ** 2)
        
        # 如果参考缓冲区还没有数据,直接返回
        if np.sum(np.abs(self.reference_buffer)) < self.epsilon:
            return audio_data
        
        # 步骤1: NLMS自适应滤波
        cleaned = self._nlms_filter(recorded_normalized, self.reference_buffer)
        
        # 步骤2: 频域谱减法(可选)
        if self.enable_spectral_subtraction and len(cleaned) >= 256:
            cleaned = self._spectral_subtraction(cleaned)
        
        # 步骤3: 噪声门限
        cleaned = self._apply_noise_gate(cleaned, self.noise_gate_threshold / 32768.0)
        
        # 计算回声抑制比率
        cleaned_energy = np.sum(cleaned ** 2)
        if original_energy > self.epsilon:
            reduction = 1.0 - (cleaned_energy / original_energy)
            self.echo_reduction_ratio = 0.95 * self.echo_reduction_ratio + 0.05 * reduction
        
        # 限制幅度并转换回int16
        cleaned = np.clip(cleaned * 32768.0, -32767, 32767)
        
        self.total_processed += 1
        
        return cleaned.astype(np.int16).tobytes()
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'total_processed': self.total_processed,
            'echo_reduction_ratio': self.echo_reduction_ratio * 100
        }


class AudioDevice:
    """音频设备管理类"""
    
    def __init__(self, input_device_index=None, output_device_index=None, sample_rate=16000, channels=1, chunk_size=1024):
        """
        初始化音频设备
        
        Args:
            input_device_index: 输入设备索引,None表示使用默认输入设备
            output_device_index: 输出设备索引,None表示使用默认输出设备
            sample_rate: 采样率
            channels: 声道数
            chunk_size: 音频块大小
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
        
        # 回声消除器(使用更强的参数)
        self.aec = AudioEchoCancellation(
            filter_length=1024,       # 增加滤波器长度以捕获更长的回声
            step_size=0.005,          # 较小的步长保证稳定性
            noise_gate_threshold=300  # 噪声门限
        )
        
        # 音频流
        self.input_stream = None
        self.output_stream = None
        
        # 播放队列
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
            raise
    
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
    
    def generate_test_audio(self, duration=3, frequency=440):
        """
        生成测试音频(正弦波)
        
        Args:
            duration: 持续时间(秒)
            frequency: 频率(Hz)
            
        Returns:
            音频数据列表
        """
        default_logger.info(f"生成测试音频: {duration}秒, 频率={frequency}Hz")
        
        total_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, total_samples, False)
        
        # 生成正弦波(降低音量以减少回声)
        audio = np.sin(2 * np.pi * frequency * t) * 0.2  # 进一步降低音量
        
        # 添加淡入淡出效果,减少突变
        fade_samples = int(0.05 * self.sample_rate)  # 50ms淡入淡出
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
        
        # 转换为int16格式
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # 分割成块
        audio_chunks = []
        for i in range(0, len(audio_int16), self.chunk_size):
            chunk = audio_int16[i:i + self.chunk_size]
            if len(chunk) < self.chunk_size:
                # 填充最后一块
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
            audio_chunks.append(chunk.tobytes())
        
        return audio_chunks
    
    def play_audio(self, audio_chunks):
        """
        播放音频
        
        Args:
            audio_chunks: 音频数据块列表
        """
        default_logger.info(f"开始播放音频... (共{len(audio_chunks)}块)")
        for chunk in audio_chunks:
            self.playback_queue.put(chunk)
    
    def record_audio(self, duration=-1, save_path="output.wav"):
        """
        录制音频
        
        Args:
            duration: 录制时长(秒)
            save_path: 保存路径
        """
        duration = float('inf') if duration < 0 else duration
        default_logger.info(f"开始录音: {duration}秒")
        if self.aec is not None:
            default_logger.info("提示: 录音时会应用高级回声消除算法(NLMS + 谱减法)")
        else:
            default_logger.info("提示: 回声消除已禁用")
        
        frames = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # 获取录制的音频数据
                data = self.recording_queue.get(timeout=1)
                frames.append(data)
                
                # 显示进度
                progress = (time.time() - start_time) / duration * 100
                if self.aec is not None:
                    stats = self.aec.get_stats()
                    sys.stdout.write(
                        f"\r录音进度: {progress:.1f}% | "
                        f"回声抑制: {stats['echo_reduction_ratio']:.1f}%"
                    )
                else:
                    sys.stdout.write(f"\r录音进度: {progress:.1f}%")
                sys.stdout.flush()
            except queue.Empty:
                continue
        
        print("\n✓ 录音完成")
        
        # 显示最终统计
        if self.aec is not None:
            stats = self.aec.get_stats()
            print(f"  总处理帧数: {stats['total_processed']}")
            print(f"  平均回声抑制率: {stats['echo_reduction_ratio']:.1f}%")
        
        # 保存录音
        self._save_wav(frames, save_path)
        
        return frames
    
    def _save_wav(self, frames, filename):
        """保存WAV文件"""
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            print(f"✓ 录音已保存到: {filename}")
        except Exception as e:
            print(f"✗ 保存录音失败: {e}")
    
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='音频录音和播放测试程序(支持高级回声消除)')
    parser.add_argument('--input-device', type=int, default=None, 
                        help='输入设备索引(默认使用系统默认输入设备)')
    parser.add_argument('--output-device', type=int, default=None, 
                        help='输出设备索引(默认使用系统默认输出设备)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='采样率(默认16000Hz)')
    parser.add_argument('--channels', type=int, default=1,
                        help='声道数(默认1)')
    parser.add_argument('--chunk-size', type=int, default=1024,
                        help='音频块大小(默认1024)')
    parser.add_argument('--play-duration', type=int, default=3,
                        help='播放测试音频时长(秒,默认3秒)')
    parser.add_argument('--record-duration', type=int, default=5,
                        help='录音时长(秒,默认5秒)')
    parser.add_argument('--output', type=str, default='output_with_aec.wav',
                        help='输出文件名(默认output_with_aec.wav)')
    parser.add_argument('--frequency', type=int, default=440,
                        help='测试音频频率(Hz,默认440Hz)')
    parser.add_argument('--mode', type=str, default='simultaneous',
                        choices=['simultaneous', 'sequential'],
                        help='测试模式：simultaneous=同时播放和录音,sequential=先播放后录音(默认simultaneous)')
    parser.add_argument('--no-aec', action='store_true',
                        help='禁用回声消除(用于对比测试)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  音频录音和播放测试程序 (高级回声消除)")
    print("="*60)
    
    # 初始化音频设备
    device = AudioDevice(
        input_device_index=args.input_device,
        output_device_index=args.output_device,
        sample_rate=args.sample_rate,
        channels=args.channels,
        chunk_size=args.chunk_size
    )
    
    # 如果禁用AEC
    if args.no_aec:
        print("\n⚠️  警告: 回声消除已禁用(用于对比测试)")
        device.aec = None
    
    try:
        # 启动音频流
        device.start_streams()
        
        # 等待流稳定
        time.sleep(0.5)
        
        # 生成测试音频
        print("\n" + "-"*60)
        print("步骤 1: 生成测试音频")
        print("-"*60)
        
        # 生成多个频率的测试音
        frequencies = [440, 523, 659]  # A, C, E 和弦
        all_chunks = []
        
        for freq in frequencies:
            chunks = device.generate_test_audio(
                duration=args.play_duration // len(frequencies),
                frequency=freq
            )
            all_chunks.extend(chunks)
        
        if args.mode == 'simultaneous':
            # 模式1: 同时播放和录音(测试回声消除)
            print("\n" + "-"*60)
            print("步骤 2: 同时播放和录音(测试回声消除效果)")
            print("-"*60)
            print("⚠️  播放音量已降低,录音时应尽可能安静")
            
            # 开始播放
            device.play_audio(all_chunks)
            time.sleep(0.2)  # 短暂延迟让播放开始
            
            # 同时录音
            device.record_audio(
                duration=args.record_duration,
                save_path=args.output
            )
            
        else:
            # 模式2: 先播放后录音(对比测试)
            print("\n" + "-"*60)
            print("步骤 2: 播放测试音频")
            print("-"*60)
            
            device.play_audio(all_chunks)
            play_time = len(all_chunks) * args.chunk_size / args.sample_rate
            time.sleep(play_time + 1)
            
            print("\n" + "-"*60)
            print("步骤 3: 录音(播放已结束)")
            print("-"*60)
            
            device.record_audio(
                duration=args.record_duration,
                save_path=args.output
            )
        
        # 等待处理完成
        print("\n等待处理完成...")
        time.sleep(1)
        
        print("\n" + "="*60)
        print("  测试完成!")
        print("="*60)
        print(f"\n📁 录音文件: {args.output}")
        
        if not args.no_aec and args.mode == 'simultaneous':
            print("\n✅ 回声消除算法已应用:")
            print("   - NLMS自适应滤波器")
            print("   - 频域谱减法")
            print("   - 智能噪声门限")
            print("\n💡 提示: 可以使用 --no-aec 参数对比无回声消除的效果")
        
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        device.cleanup()


if __name__ == "__main__":
    main()

