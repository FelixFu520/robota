#!/usr/bin/env python3
"""
WAV 文件重采样脚本
将音频文件重采样为 44100 Hz
"""

import argparse
import wave
import numpy as np
from scipy import signal
import os


def resample_wav(input_path, output_path, target_rate=44100):
    """
    将 WAV 文件重采样为目标采样率
    
    Args:
        input_path: 输入 WAV 文件路径
        output_path: 输出 WAV 文件路径
        target_rate: 目标采样率，默认 44100 Hz
    """
    # 读取原始 WAV 文件
    with wave.open(input_path, 'rb') as wav_in:
        # 获取音频参数
        n_channels = wav_in.getnchannels()
        sampwidth = wav_in.getsampwidth()
        framerate = wav_in.getframerate()
        n_frames = wav_in.getnframes()
        
        print(f"输入文件信息:")
        print(f"  声道数: {n_channels}")
        print(f"  采样宽度: {sampwidth} 字节")
        print(f"  采样率: {framerate} Hz")
        print(f"  总帧数: {n_frames}")
        print(f"  时长: {n_frames / framerate:.2f} 秒")
        
        # 读取音频数据
        audio_data = wav_in.readframes(n_frames)
        
        # 根据采样宽度转换为 numpy 数组
        if sampwidth == 1:
            dtype = np.uint8
        elif sampwidth == 2:
            dtype = np.int16
        elif sampwidth == 4:
            dtype = np.int32
        else:
            raise ValueError(f"不支持的采样宽度: {sampwidth}")
        
        # 转换为 numpy 数组
        audio_array = np.frombuffer(audio_data, dtype=dtype)
        
        # 如果是多声道，需要重新整形
        if n_channels > 1:
            audio_array = audio_array.reshape(-1, n_channels)
    
    # 如果采样率已经是目标采样率，直接复制
    if framerate == target_rate:
        print(f"\n采样率已经是 {target_rate} Hz，直接复制文件")
        with open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
        return
    
    # 重采样
    print(f"\n开始重采样: {framerate} Hz -> {target_rate} Hz")
    
    if n_channels == 1:
        # 单声道
        resampled_array = signal.resample_poly(audio_array, target_rate, framerate)
    else:
        # 多声道，分别处理每个声道
        resampled_channels = []
        for ch in range(n_channels):
            channel_data = audio_array[:, ch]
            resampled_channel = signal.resample_poly(channel_data, target_rate, framerate)
            resampled_channels.append(resampled_channel)
        resampled_array = np.column_stack(resampled_channels)
    
    # 转换回原始数据类型
    resampled_array = np.clip(resampled_array, np.iinfo(dtype).min, np.iinfo(dtype).max)
    resampled_array = resampled_array.astype(dtype)
    
    # 写入新的 WAV 文件
    with wave.open(output_path, 'wb') as wav_out:
        wav_out.setnchannels(n_channels)
        wav_out.setsampwidth(sampwidth)
        wav_out.setframerate(target_rate)
        wav_out.writeframes(resampled_array.tobytes())
    
    print(f"\n输出文件信息:")
    print(f"  声道数: {n_channels}")
    print(f"  采样宽度: {sampwidth} 字节")
    print(f"  采样率: {target_rate} Hz")
    print(f"  总帧数: {len(resampled_array) if n_channels == 1 else len(resampled_array)}")
    print(f"  时长: {len(resampled_array) / target_rate if n_channels == 1 else len(resampled_array) / target_rate:.2f} 秒")
    print(f"\n文件已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='WAV 文件重采样工具')
    parser.add_argument('input', help='输入 WAV 文件路径')
    parser.add_argument('-o', '--output', help='输出 WAV 文件路径（默认在同目录下添加 _44100 后缀）')
    parser.add_argument('-r', '--rate', type=int, default=44100, help='目标采样率（默认: 44100）')
    
    args = parser.parse_args()
    
    # 如果没有指定输出路径，自动生成
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_{args.rate}{ext}"
    
    # 执行重采样
    resample_wav(args.input, args.output, args.rate)


if __name__ == '__main__':
    main()

