import argparse
import asyncio
import wave
import struct

from robota.audio.volcengine_doubao_asr import AsrWsClient
from robota.audio.audio_device import AudioDevice
from robota.utils.logging import default_logger as logger


async def test_file_asr(args):
    """测试文件ASR"""
    logger.info("=== 测试文件ASR ===")
    async with AsrWsClient(args.url, args.seg_duration) as client:
        try:
            async for response in client.execute(args.file):
                if "text" in response.to_dict()['payload_msg']['result']:
                    print(response.to_dict()['payload_msg']['result']['text'] + "   " + str(response.to_dict()['payload_msg']['result']['utterances'][0]['definite']))
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
    sample_rate = audio_device.sample_rate
    channels = audio_device.channels
    chunk_size = audio_device.chunk_size
    
    # 计算需要多少个chunk才能达到chunk_duration_ms
    samples_per_chunk = chunk_size
    bytes_per_sample = 2  # int16
    bytes_per_chunk = samples_per_chunk * channels * bytes_per_sample
    
    # 计算需要累积多少个chunk才能达到目标时长
    target_bytes = int(sample_rate * channels * bytes_per_sample * chunk_duration_ms / 1000)
    chunks_needed = max(1, target_bytes // bytes_per_chunk)
    
    logger.info(f"实时录音配置: 采样率={sample_rate}, 块大小={chunk_size}, 每{chunk_duration_ms}ms发送一次")
    logger.info(f"每次发送需要累积 {chunks_needed} 个chunk, 约 {target_bytes} 字节")
    
    start_time = asyncio.get_event_loop().time()
    accumulated_data = b''
    chunk_count = 0
    
    try:
        while True:
            # 检查是否超时
            if duration_seconds is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= duration_seconds:
                    logger.info(f"录音时长达到 {duration_seconds} 秒，停止录音")
                    break
            
            # 异步获取录音数据
            audio_chunk = await audio_device.async_get_recorded_data(timeout=1.0)
            if audio_chunk is None:
                continue
            
            accumulated_data += audio_chunk
            chunk_count += 1
            
            # 当累积足够的数据后发送
            if chunk_count >= chunks_needed:
                # 创建WAV格式的数据
                wav_data = create_wav_chunk(accumulated_data, sample_rate, channels)
                yield wav_data
                
                # 重置累积
                accumulated_data = b''
                chunk_count = 0
        
        # 发送剩余数据
        if accumulated_data:
            wav_data = create_wav_chunk(accumulated_data, sample_rate, channels)
            yield wav_data
        
        # 发送结束信号
        yield None
        
    except Exception as e:
        logger.error(f"录音生成器错误: {e}")
        raise


def create_wav_chunk(pcm_data, sample_rate, channels):
    """
    创建WAV格式的音频数据
    
    Args:
        pcm_data: PCM原始音频数据
        sample_rate: 采样率
        channels: 声道数
        
    Returns:
        WAV格式的音频数据(bytes)
    """
    import io
    buffer = io.BytesIO()
    
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    
    return buffer.getvalue()


async def test_realtime_asr(args):
    """测试实时录音ASR"""
    logger.info("=== 测试实时录音ASR ===")
    
    # 创建音频设备
    audio_device = AudioDevice(
        sample_rate=16000,
        channels=1,
        chunk_size=1024,
        enable_aec=False  # 实时ASR测试时可以关闭AEC
    )
    
    try:
        # 启动音频流
        audio_device.start_streams()
        logger.info(f"开始录音，时长: {args.duration} 秒")
        logger.info("请开始说话...")
        
        # 创建ASR客户端并开始识别
        async with AsrWsClient(args.url, args.seg_duration) as client:
            # 创建实时音频生成器
            audio_stream = realtime_audio_generator(
                audio_device,
                duration_seconds=args.duration,
                chunk_duration_ms=args.seg_duration
            )
            
            # 开始实时ASR识别
            try:
                async for response in client.execute_stream(audio_stream):
                    resp_dict = response.to_dict()
                    if resp_dict.get('payload_msg') and 'result' in resp_dict['payload_msg']:
                        result = resp_dict['payload_msg']['result']
                        if 'text' in result:
                            text = result['text']
                            is_definite = result.get('utterances', [{}])[0].get('definite', False) if result.get('utterances') else False
                            print(f"[{'确定' if is_definite else '临时'}] {text}")
                            
            except Exception as e:
                logger.error(f"实时ASR处理失败: {e}")
                raise
                
    finally:
        # 清理资源
        audio_device.cleanup()
        logger.info("音频设备已清理")


async def main():
    parser = argparse.ArgumentParser(description="ASR WebSocket Client")
    parser.add_argument("--mode", type=str, choices=['file', 'realtime'], default='realtime',
                       help="测试模式: file=文件ASR, realtime=实时录音ASR")
    parser.add_argument("--file", type=str, 
                       default="/home/drobotics/projects/robota/assets/xiaozhan_ref_16000_concatenated.wav", 
                       help="Audio file path (for file mode)")
    parser.add_argument("--url", type=str, 
                       default="wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async", 
                       help="WebSocket URL")
    parser.add_argument("--seg-duration", type=int, default=200, 
                       help="Audio duration(ms) per packet, default:200")
    parser.add_argument("--duration", type=int, default=10,
                       help="录音时长(秒), 仅用于realtime模式")
    
    args = parser.parse_args()
    
    if args.mode == 'file':
        await test_file_asr(args)
    elif args.mode == 'realtime':
        await test_realtime_asr(args)


if __name__ == "__main__":
    asyncio.run(main())

    # 用法：
    # 文件ASR: python3 test_doubao_asr.py --mode file --file /path/to/audio.wav
    # 实时录音ASR: python3 test_doubao_asr.py --mode realtime --duration 10