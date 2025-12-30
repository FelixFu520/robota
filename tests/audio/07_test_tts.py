#!/usr/bin/env python3
"""
TTS 测试脚本
使用异步任务每隔 0.5s 发送文本到 TTS 队列，然后让 asr_tts.py 处理并播放
"""

import asyncio
from robota.audio.asr_tts import ASRTTS
from robota.utils.logging import default_logger as logger


async def text_sender(asr_tts: ASRTTS, interval: float = 0.5, count: int = 100):
    """
    文本发送异步任务，每隔指定时间发送一条文本到 TTS 队列
    
    Args:
        asr_tts: ASRTTS 实例
        interval: 发送间隔（秒），默认 0.5 秒
        count: 发送文本数量，默认 1000 条
    """
    logger.info(f"文本发送任务启动，将每隔 {interval} 秒发送一条文本，共 {count} 条")
    
    for i in range(count):
        asr_tts.put_tts_text(f"{i}, 这是条文本")
        await asyncio.sleep(interval)
    
    logger.info(f"文本发送完成，共发送 {count} 条文本")


async def main():
    """
    主函数：启动 TTS 处理器和文本发送异步任务，让它们并行工作
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="TTS 队列测试工具")
    parser.add_argument("--interval", type=float, default=0.1,
                       help="文本发送间隔（秒），默认: 0.1")
    parser.add_argument("--count", type=int, default=1000,
                       help="发送文本数量，默认: 300")
    
    args = parser.parse_args()
    
    # 创建 ASRTTS 实例
    logger.info("初始化 ASRTTS...")
    asr_tts = ASRTTS()
    
    # 启动 TTS 处理器任务
    tts_task = None
    sender_task = None
    
    try:
        # 启动 TTS 处理器（异步任务）
        # 注意：TTS 处理器会等待第一个文本放入队列时才建立连接
        logger.info("启动 TTS 处理器...")
        tts_task = asyncio.create_task(asr_tts.start_tts_processor())
        
        # 短暂等待确保 TTS 处理器任务已启动
        await asyncio.sleep(0.1)
        
        # 启动文本发送异步任务（与 TTS 处理器并行工作）
        # 第一个文本放入队列时，TTS 处理器会立即建立连接并开始处理
        logger.info("启动文本发送任务...")
        sender_task = asyncio.create_task(text_sender(asr_tts, args.interval, args.count))
        
        await asyncio.sleep(2.0)    # sleep一会，asr_tts.tts_queue多放些文本

        # 等待 TTS 队列处理完成（所有文本都被处理）
        while not asr_tts.tts_queue.empty():
            logger.info(f"等待 TTS 队列处理，队列大小: {asr_tts.tts_queue.qsize()}")
            await asyncio.sleep(1)  # 每 100ms 检查一次
        
        # 额外等待一段时间确保最后的音频播放完
        await asyncio.sleep(1.0)
        logger.info("所有语音播放完毕")
        
        # 停止 TTS 处理器
        logger.info("停止 TTS 处理器...")
        asr_tts.stop_tts_processor()
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
        asr_tts.stop_tts_processor()
        if sender_task:
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass
        if tts_task:
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise
    finally:
        # 清理资源
        if asr_tts.audio_device:
            asr_tts.audio_device.cleanup()
        logger.info("测试结束，资源已清理")


if __name__ == "__main__":
    asyncio.run(main())

