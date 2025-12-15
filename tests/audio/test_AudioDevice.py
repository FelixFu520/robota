import argparse
import time
import wave
import os
import threading
import numpy as np
from datetime import datetime
from robota.audio.audio_device import AudioDevice


class AudioRecorderPlayer:
    """音频录音播放管理器"""
    
    def __init__(self, device, playback_file, output_dir='./recordings', record_interval=10):
        """
        初始化录音播放管理器
        
        Args:
            device: AudioDevice实例
            playback_file: 要循环播放的音频文件路径
            output_dir: 录音文件保存目录
            record_interval: 录音保存间隔(秒)
        """
        self.device = device
        self.playback_file = playback_file
        self.output_dir = output_dir
        self.record_interval = record_interval
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 控制标志
        self.is_running = False
        
        # 线程
        self.playback_thread = None
        self.recording_thread = None
        
        # 统计信息
        self.total_played_chunks = 0
        self.total_recorded_files = 0
    
    def _load_playback_audio(self):
        """加载要播放的音频文件"""
        try:
            with wave.open(self.playback_file, 'rb') as wf:
                # 验证音频参数
                if wf.getnchannels() != self.device.channels:
                    print(f"警告: 音频文件声道数({wf.getnchannels()})与设备声道数({self.device.channels})不匹配")
                if wf.getframerate() != self.device.sample_rate:
                    print(f"警告: 音频文件采样率({wf.getframerate()})与设备采样率({self.device.sample_rate})不匹配")
                if wf.getsampwidth() != 2:  # pyaudio.paInt16 = 2 bytes
                    print(f"警告: 音频文件采样宽度不是16位")
                
                # 读取所有音频数据
                audio_data = wf.readframes(wf.getnframes())
                print(f"✓ 成功加载音频文件: {self.playback_file}")
                print(f"  采样率: {wf.getframerate()}Hz, 声道数: {wf.getnchannels()}, 时长: {len(audio_data)/wf.getframerate()/wf.getnchannels()/2:.2f}秒")
                return audio_data
        except Exception as e:
            print(f"✗ 加载音频文件失败: {e}")
            raise
    
    def _playback_loop(self):
        """播放循环线程"""
        print("\n[播放线程] 启动")
        
        # 加载音频文件
        audio_data = self._load_playback_audio()
        
        # 计算每个chunk的字节数
        bytes_per_chunk = self.device.chunk_size * self.device.channels * 2  # 2 bytes per sample (int16)
        
        while self.is_running:
            try:
                # 将音频数据分块放入播放队列
                offset = 0
                while offset < len(audio_data) and self.is_running:
                    chunk = audio_data[offset:offset + bytes_per_chunk]
                    
                    # 如果是最后一块且不足chunk_size,补零
                    if len(chunk) < bytes_per_chunk:
                        chunk = chunk + b'\x00' * (bytes_per_chunk - len(chunk))
                    
                    # 放入播放队列
                    self.device.put_playback_data(chunk)
                    self.total_played_chunks += 1
                    
                    offset += bytes_per_chunk
                    
                    # 控制播放速度,避免队列堆积
                    while self.device.get_playback_queue_size() > 50 and self.is_running:
                        time.sleep(0.01)
                
                # 循环播放
                if self.is_running:
                    print(f"[播放线程] 完成一次播放循环,共播放 {self.total_played_chunks} 个chunk,继续循环...")
                    
            except Exception as e:
                print(f"[播放线程] 错误: {e}")
                break
        
        print("[播放线程] 退出")
    
    def _recording_loop(self):
        """录音循环线程"""
        print("\n[录音线程] 启动")
        
        while self.is_running:
            try:
                # 计算需要录制的帧数
                frames_to_record = int(self.device.sample_rate * self.record_interval)
                
                # 收集音频数据
                recorded_frames = []
                recorded_samples = 0
                
                print(f"\n[录音线程] 开始录制 {self.record_interval} 秒音频...")
                start_time = time.time()
                
                while recorded_samples < frames_to_record and self.is_running:
                    # 从录音队列获取数据
                    audio_data = self.device.get_recorded_data(block=True, timeout=1.0)
                    
                    if audio_data is not None:
                        recorded_frames.append(audio_data)
                        # 每个chunk有chunk_size个采样点
                        recorded_samples += self.device.chunk_size
                
                if not self.is_running:
                    break
                
                # 保存音频文件
                elapsed_time = time.time() - start_time
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = os.path.join(self.output_dir, f"recording_{timestamp}.wav")
                
                self._save_wav_file(output_filename, b''.join(recorded_frames))
                self.total_recorded_files += 1
                
                print(f"[录音线程] ✓ 保存录音: {output_filename}")
                print(f"[录音线程]   实际录制时长: {elapsed_time:.2f}秒, 文件: {len(recorded_frames)} 个chunk")
                
            except Exception as e:
                print(f"[录音线程] 错误: {e}")
                import traceback
                traceback.print_exc()
        
        print("[录音线程] 退出")
    
    def _save_wav_file(self, filename, audio_data):
        """
        保存音频数据为WAV文件
        
        Args:
            filename: 输出文件名
            audio_data: 音频数据(bytes)
        """
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.device.channels)
                wf.setsampwidth(2)  # 16-bit = 2 bytes
                wf.setframerate(self.device.sample_rate)
                wf.writeframes(audio_data)
        except Exception as e:
            print(f"保存WAV文件失败: {e}")
            raise
    
    def start(self):
        """启动录音和播放"""
        if self.is_running:
            print("已经在运行中")
            return
        
        print("\n" + "="*60)
        print("启动音频录音和播放系统")
        print("="*60)
        
        # 启动音频设备
        self.device.start_streams()
        
        self.is_running = True
        
        # 启动播放线程
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()
        
        # 启动录音线程
        self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.recording_thread.start()
        
        print("\n✓ 系统已启动")
        print(f"  - 循环播放: {self.playback_file}")
        print(f"  - 录音保存间隔: {self.record_interval}秒")
        print(f"  - 录音保存目录: {self.output_dir}")
        print("\n按 Ctrl+C 停止...")
    
    def stop(self):
        """停止录音和播放"""
        if not self.is_running:
            return
        
        print("\n\n正在停止...")
        self.is_running = False
        
        # 等待线程结束
        if self.playback_thread:
            self.playback_thread.join(timeout=2.0)
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
        
        # 停止音频设备
        self.device.stop_streams()
        
        print("\n" + "="*60)
        print("统计信息:")
        print(f"  - 总播放chunk数: {self.total_played_chunks}")
        print(f"  - 总录音文件数: {self.total_recorded_files}")
        print("="*60)
    
    def run(self):
        """运行(阻塞直到用户中断)"""
        self.start()
        
        try:
            # 定期显示状态
            while self.is_running:
                time.sleep(5)
                
                # 获取AEC统计信息
                aec_stats = ""
                if self.device.aec is not None:
                    stats = self.device.aec.get_stats()
                    aec_stats = (f"| AEC: 延迟={stats.get('estimated_delay', 0)}样本, "
                               f"双讲={'是' if stats.get('is_double_talk', False) else '否'}, "
                               f"近端语音={'是' if stats.get('has_near_speech', False) else '否'}, "
                               f"远端语音={'是' if stats.get('has_far_speech', False) else '否'}, "
                               f"抑制率={stats.get('echo_reduction_ratio', 0):.1f}%")
                
                print(f"\r[状态] 播放队列: {self.device.get_playback_queue_size()}, "
                      f"录音队列: {self.device.get_recording_queue_size()}, "
                      f"已播放: {self.total_played_chunks} chunks, "
                      f"已保存: {self.total_recorded_files} 个文件 {aec_stats}", end='')
        except KeyboardInterrupt:
            print("\n\n用户中断")
        finally:
            self.stop()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='音频录音和播放测试程序(支持回声消除)')
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
    parser.add_argument('--playback-file', type=str, 
                        default='/home/drobotics/projects/robota/assets/xiaozhan_ref_16000_concatenated.wav',
                        help='循环播放的音频文件路径')
    parser.add_argument('--output-dir', type=str, default='./recordings',
                        help='录音文件保存目录(默认./recordings)')
    parser.add_argument('--record-interval', type=int, default=10,
                        help='录音保存间隔(秒,默认10秒)')
    parser.add_argument('--no-aec', action='store_true',
                        help='禁用回声消除')
    parser.add_argument('--aec-suppression', type=float, default=0.5,
                        help='回声抑制因子(0.0-1.0, 越小抑制越强, 默认0.5)')
    parser.add_argument('--aec-filter-length', type=int, default=2048,
                        help='AEC滤波器长度(默认2048, 越大效果越好但计算量越大)')
    parser.add_argument('--aec-noise-gate', type=int, default=500,
                        help='AEC噪声门限(默认500, 越大越抑制低能量信号)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  音频录音和播放测试程序")
    print("="*60)
    
    # 初始化音频设备
    device = AudioDevice(
        input_device_index=args.input_device,
        output_device_index=args.output_device,
        sample_rate=args.sample_rate,
        channels=args.channels,
        chunk_size=args.chunk_size,
        enable_aec=not args.no_aec
    )
    
    if args.no_aec:
        print("\n⚠️  警告: 回声消除已禁用")
    else:
        # 调整AEC参数
        if device.aec:
            from collections import deque
            device.aec.suppression_factor = max(0.0, min(1.0, args.aec_suppression))
            device.aec.filter_length = args.aec_filter_length
            device.aec.noise_gate_threshold = args.aec_noise_gate
            device.aec.filter_coeffs = np.zeros(args.aec_filter_length, dtype=np.float32)
            device.aec.reference_buffer_size = args.aec_filter_length * 4
            device.aec.reference_buffer = deque(maxlen=device.aec.reference_buffer_size)
            
            print(f"\n✓ 智能回声消除已启用 (自适应模式)")
            print(f"  - 基础抑制因子: {device.aec.suppression_factor:.2f}")
            print(f"  - 滤波器长度: {device.aec.filter_length}")
            print(f"  - 噪声门限: {device.aec.noise_gate_threshold}")
            print(f"\n工作模式:")
            print(f"  • 自动检测回声与语音的能量比")
            print(f"  • 纯回声(>70%): 强抑制")
            print(f"  • 混合信号(30-70%): 自适应抑制")
            print(f"  • 主要语音(<30%): 仅轻微处理")
            print("\n调整建议:")
            print("  - 如果仍有回声: --aec-suppression 0.3 (更强)")
            print("  - 如果语音失真: --aec-suppression 0.7 (更弱)")
            print("  - 使用耳机可获得最佳效果")
    
    # 创建录音播放管理器
    manager = AudioRecorderPlayer(
        device=device,
        playback_file=args.playback_file,
        output_dir=args.output_dir,
        record_interval=args.record_interval
    )
    
    try:
        manager.run()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        device.cleanup()
        print("\n再见!")


if __name__ == "__main__":
    main()
