#!/usr/bin/env python3
"""
豆包ASR测试脚本
测试豆包语音识别服务，支持文件ASR和实时录音ASR两种模式

用法：
    文件ASR: python3 test_doubao_asr.py --mode file --file /path/to/audio.wav
    实时录音ASR: python3 test_doubao_asr.py --mode realtime --duration 10
"""

import argparse
import asyncio
import wave
import struct

from robota.audio.volcengine_doubao_asr import AsrWsClient
from robota.audio.audio_device import AudioDevice
from robota.utils.logging import default_logger as logger


async def test_file_asr(args):
    """
    测试文件ASR识别
    
    Args:
        args: 命令行参数对象，包含以下属性：
            url: WebSocket服务器URL
            seg_duration: 音频片段时长(毫秒)
            file: 音频文件路径
    """
    logger.info("=== 测试文件ASR ===")
    
    # 创建ASR客户端并执行文件识别
    async with AsrWsClient(args.url, args.seg_duration) as client:
        try:
            # 遍历识别结果
            async for response in client.execute(args.file):
                # 检查响应中是否包含识别文本
                if "text" in response.to_dict()['payload_msg']['result']:
                    result = response.to_dict()['payload_msg']['result']
                    text = result['text']
                    # 获取识别结果的确定状态
                    is_definite = result['utterances'][0]['definite']
                    print(f"{text}   {is_definite}")
        except Exception as e:
            logger.error(f"ASR processing failed: {e}")


async def realtime_audio_generator(audio_device, duration_seconds=10, chunk_duration_ms=200):
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
    
    logger.info(f"实时录音配置: 采样率={sample_rate}, 块大小={chunk_size}, 每{chunk_duration_ms}ms发送一次")
    logger.info(f"每次发送需要累积 {chunks_needed} 个chunk, 约 {target_bytes} 字节")
    
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
                    logger.info(f"录音时长达到 {duration_seconds} 秒，停止录音")
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
                wav_data = create_wav_chunk(accumulated_data, sample_rate, channels)
                yield wav_data
                
                # 重置累积状态，准备下一批数据
                accumulated_data = b''
                chunk_count = 0
        
        # 发送剩余的音频数据（如果有）
        if accumulated_data:
            wav_data = create_wav_chunk(accumulated_data, sample_rate, channels)
            yield wav_data
        
        # 发送结束信号（None表示音频流结束）
        yield None
        
    except Exception as e:
        logger.error(f"录音生成器错误: {e}")
        raise


def create_wav_chunk(pcm_data, sample_rate, channels):
    """
    创建WAV格式的音频数据
    
    Args:
        pcm_data: PCM原始音频数据(bytes)
        sample_rate: 采样率(Hz)
        channels: 声道数
        
    Returns:
        WAV格式的音频数据(bytes)
    """
    import io
    
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


async def test_realtime_asr(args):
    """
    测试实时录音ASR识别
    
    Args:
        args: 命令行参数对象，包含以下属性：
            url: WebSocket服务器URL
            seg_duration: 音频片段时长(毫秒)
            duration: 录音时长(秒)
    """
    logger.info("=== 测试实时录音ASR ===")
    
    # 创建音频设备，配置录音参数
    audio_device = AudioDevice(
        sample_rate=16000,  # 采样率16kHz，符合ASR服务要求
        channels=1,         # 单声道
        chunk_size=1024,    # 每次读取1024个样本
        enable_aec=False    # 实时ASR测试时可以关闭回声消除
    )
    
    try:
        # 启动音频输入流
        audio_device.start_streams()
        logger.info(f"开始录音，时长: {args.duration} 秒")
        logger.info("请开始说话...")
        
        # 创建ASR客户端并开始识别
        async with AsrWsClient(args.url, args.seg_duration) as client:
            # 创建实时音频生成器，将录音数据转换为WAV格式片段
            audio_stream = realtime_audio_generator(
                audio_device,
                duration_seconds=args.duration,
                chunk_duration_ms=args.seg_duration
            )
            
            # 开始实时ASR识别，处理音频流
            try:
                async for response in client.execute_stream(audio_stream):
                    # 解析响应数据
                    resp_dict = response.to_dict()
                    
                    # 检查响应中是否包含识别结果
                    if resp_dict.get('payload_msg') and 'result' in resp_dict['payload_msg']:
                        result = resp_dict['payload_msg']['result']
                        
                        # 如果包含识别文本，则输出
                        if 'text' in result:
                            text = result['text']
                            # 获取识别结果的确定状态（是否为最终结果）
                            is_definite = result.get('utterances', [{}])[0].get('definite', False) if result.get('utterances') else False
                            print(f"[{'确定' if is_definite else '临时'}] {text}")
                            
            except Exception as e:
                logger.error(f"实时ASR处理失败: {e}")
                raise
                
    finally:
        # 清理音频设备资源
        audio_device.cleanup()
        logger.info("音频设备已清理")


async def main():
    """
    主函数，解析命令行参数并执行相应的测试模式
    """
    parser = argparse.ArgumentParser(description="豆包ASR WebSocket客户端测试工具")
    
    # 添加命令行参数
    parser.add_argument("--mode", type=str, choices=['file', 'realtime'], default='realtime',
                       help="测试模式: file=文件ASR, realtime=实时录音ASR（默认: realtime）")
    parser.add_argument("--file", type=str, 
                       default="/home/drobotics/projects/robota/assets/xiaozhan_ref_16000_concatenated.wav", 
                       help="音频文件路径（用于file模式）")
    parser.add_argument("--url", type=str, 
                       default="wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async", 
                       help="WebSocket服务器URL")
    parser.add_argument("--seg-duration", type=int, default=200, 
                       help="每个音频包的时长(毫秒)，默认: 200")
    parser.add_argument("--duration", type=int, default=10,
                       help="录音时长(秒)，仅用于realtime模式，默认: 10")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据模式执行相应的测试
    if args.mode == 'file':
        await test_file_asr(args)
    elif args.mode == 'realtime':
        await test_realtime_asr(args)


if __name__ == "__main__":
    asyncio.run(main())