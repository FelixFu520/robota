from robota.audio.asr_tts import ASRTTS
from robota.utils.logging import default_logger as logger
import asyncio


async def monitor_queue(asr_tts: ASRTTS):
    """
    监控ASR队列，完成一次识别后显示队列内容（不删除队列内容）
    
    Args:
        asr_tts: ASRTTS实例
    """
    print("开始监控ASR队列...")
    
    # 确保事件已创建
    if asr_tts.asr_queue_event is None:
        asr_tts.asr_queue_event = asyncio.Event()
    
    while True:
        try:
            # 等待有新结果放入队列的事件
            await asr_tts.asr_queue_event.wait()
            
            # 清除事件，准备下次等待
            asr_tts.asr_queue_event.clear()
            
            # 取出队列中的所有元素，打印后放回去
            if not asr_tts.asr_queue.empty():
                # 临时存储所有取出的元素
                temp_results = []
                
                # 取出队列中的所有元素
                while not asr_tts.asr_queue.empty():
                    try:
                        result = asr_tts.asr_queue.get_nowait()
                        if result:
                            temp_results.append(result)
                    except:
                        break
                
                # 打印所有结果
                if temp_results:
                    print(f"\n[队列内容] 共有 {len(temp_results)} 条结果:")
                    for i, result in enumerate(temp_results, 1):
                        print(f"  {i}. chat_id: {result['chat_id']}, text: {result['text']}")
                        # logger.info(f"队列结果 {i}: {result}")
                    
                    # 将所有元素放回队列（保持原有顺序）
                    for result in temp_results:
                        asr_tts.asr_queue.put(result)
        except asyncio.CancelledError:
            # 任务被取消，退出循环
            break
        except Exception as e:
            # 发生异常，等待后继续
            logger.debug(f"队列监控异常: {e}")
            await asyncio.sleep(0.1)


async def main():
    """
    测试程序主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="ASRTTS实时ASR测试工具")
    parser.add_argument("--silence-timeout", type=int, default=800,
                       help="静音超时时间(毫秒)，默认: 800")
    
    args = parser.parse_args()
    
    # 创建ASRTTS实例
    asr_tts = ASRTTS()

    queue_task = None
    try:
        # 创建队列监控任务
        queue_task = asyncio.create_task(monitor_queue(asr_tts))
        
        # 启动实时ASR识别
        await asr_tts.start_realtime_asr(silence_timeout_ms=args.silence_timeout)
        
        # ASR识别完成后，等待一小段时间让队列中的结果被处理
        # await asyncio.sleep(1.0)
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise
    finally:
        # 取消队列监控任务
        if queue_task:
            queue_task.cancel()
            try:
                await queue_task
            except asyncio.CancelledError:
                pass
        
        # 清理资源
        asr_tts.audio_device.cleanup()
        logger.info("测试结束，资源已清理")


if __name__ == "__main__":
    asyncio.run(main())