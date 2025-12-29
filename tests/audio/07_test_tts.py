#!/usr/bin/env python3
"""
TTS 测试脚本
使用线程每隔 0.5s 发送文本到 TTS 队列，然后让 asr_tts.py 处理并播放
"""

import asyncio
import threading
import time
from robota.audio.asr_tts import ASRTTS
from robota.utils.logging import default_logger as logger


def text_sender(asr_tts: ASRTTS, interval: float = 0.5, count: int = 1000, stop_event: threading.Event = None):
    """
    文本发送线程，每隔指定时间发送一条文本到 TTS 队列
    
    Args:
        asr_tts: ASRTTS 实例
        interval: 发送间隔（秒），默认 0.5 秒
        count: 发送文本数量，默认 1000 条
        stop_event: 停止事件，如果设置了，可以通过该事件提前停止
    """
    logger.info(f"文本发送线程启动，将每隔 {interval} 秒发送一条文本，共 {count} 条")
    
    for i in range(count):
        if stop_event and stop_event.is_set():
            logger.info(f"收到停止信号，已发送 {i} 条文本")
            break
        asr_tts.put_tts_text(f"这是第 {i} 条文本")
        time.sleep(interval)
    
    logger.info(f"文本发送完成，共发送 {i+1 if i < count else count} 条文本")


async def main():
    """
    主函数：启动 TTS 处理器和文本发送线程，让它们并行工作
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="TTS 队列测试工具")
    parser.add_argument("--interval", type=float, default=0.5,
                       help="文本发送间隔（秒），默认: 0.5")
    parser.add_argument("--count", type=int, default=10,
                       help="发送文本数量，默认: 1000")
    
    args = parser.parse_args()
    
    # 创建 ASRTTS 实例
    logger.info("初始化 ASRTTS...")
    asr_tts = ASRTTS()
    
    # 启动 TTS 处理器任务
    tts_task = None
    sender_thread = None
    stop_event = threading.Event()
    
    try:
        # 启动 TTS 处理器（异步任务）
        logger.info("启动 TTS 处理器...")
        tts_task = asyncio.create_task(asr_tts.start_tts_processor())
        
        # 等待 TTS 处理器初始化完成
        await asyncio.sleep(1.0)
        
        # 启动文本发送线程（与 TTS 处理器并行工作）
        logger.info("启动文本发送线程...")
        sender_thread = threading.Thread(
            target=text_sender,
            args=(asr_tts, args.interval, args.count, stop_event),
            daemon=True
        )
        sender_thread.start()
        
        # 监控状态：发送线程和 TTS 处理器并行工作
        logger.info("发送线程和 TTS 处理器已并行启动，正在工作...")
        
        # 等待文本发送完成（发送线程和 TTS 处理器并行工作，不需要等待 TTS 处理）
        sender_thread.join()
        logger.info("文本发送完成，等待 TTS 队列处理完成...")
        
        # 等待 TTS 队列处理完成（所有文本都被处理）
        max_wait_tts = 300  # 最多等待 30 秒（300 * 0.1）
        wait_count_tts = 0
        while not asr_tts.tts_queue.empty() and wait_count_tts < max_wait_tts:
            await asyncio.sleep(0.1)  # 每 100ms 检查一次
            wait_count_tts += 1
            if wait_count_tts % 10 == 0:  # 每秒打印一次状态
                logger.info(f"等待 TTS 队列处理，队列大小: {asr_tts.tts_queue.qsize()}")
        
        if not asr_tts.tts_queue.empty():
            logger.warning(f"TTS 队列仍有 {asr_tts.tts_queue.qsize()} 条未处理")
        else:
            logger.info("TTS 队列处理完成")
        
        # 等待播放队列播放完成
        logger.info("等待音频播放完成...")
        await asyncio.sleep(0.5)  # 先等待队列有足够的数据
        
        max_wait_playback = 600  # 最多等待 60 秒（600 * 0.1）
        wait_count_playback = 0
        while asr_tts.audio_device.get_playback_queue_size() > 0 and wait_count_playback < max_wait_playback:
            await asyncio.sleep(0.1)  # 每 100ms 检查一次
            wait_count_playback += 1
            if wait_count_playback % 10 == 0:  # 每秒打印一次状态
                queue_size = asr_tts.audio_device.get_playback_queue_size()
                logger.info(f"等待播放完成，播放队列大小: {queue_size}")
        
        if asr_tts.audio_device.get_playback_queue_size() > 0:
            logger.warning(f"播放队列仍有 {asr_tts.audio_device.get_playback_queue_size()} 条未播放")
        else:
            logger.info("播放队列已清空")
        
        # 额外等待一段时间确保最后的音频播放完
        await asyncio.sleep(1.0)
        logger.info("所有语音播放完毕")
        
        # 停止 TTS 处理器
        logger.info("停止 TTS 处理器...")
        asr_tts.stop_tts_processor()
        
        # 等待 TTS 任务完成
        try:
            await asyncio.wait_for(tts_task, timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("TTS 任务超时，强制取消")
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
        stop_event.set()  # 通知发送线程停止
        asr_tts.stop_tts_processor()
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

