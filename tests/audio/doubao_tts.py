#!/usr/bin/env python3
"""
豆包 TTS (Text-to-Speech) 测试脚本
通过 WebSocket 连接到豆包 TTS 服务，将文本转换为语音
支持实时音频播放功能
"""

import argparse
import websockets
import asyncio
import copy
import json
import uuid
import wave
import io
from pydub import AudioSegment

from robota.audio.volcengine_doubao_tts import (
    EventType,
    MsgType,
    finish_connection,
    finish_session,
    receive_message,
    start_connection,
    start_session,
    task_request,
    wait_for_event,
)
from robota.audio.audio_device import AudioDevice
from robota.utils.logging import default_logger as logger


def get_resource_id(voice: str) -> str:
    """
    根据语音类型获取资源 ID
    
    Args:
        voice: 语音类型字符串，如果以 "S_" 开头则使用默认资源，否则使用服务类型资源
    
    Returns:
        资源 ID 字符串
    """
    if voice.startswith("S_"):
        return "volc.megatts.default"
    return "volc.service_type.10029"


def convert_mp3_to_pcm(mp3_data: bytes, target_sample_rate: int = 16000) -> bytes:
    """
    将 MP3 音频数据转换为 PCM 格式
    
    Args:
        mp3_data: MP3 格式的音频数据（字节流）
        target_sample_rate: 目标采样率，默认 16000 Hz
    
    Returns:
        PCM 格式的音频数据（字节流），如果转换失败则返回空字节流
    """
    try:
        # 使用 pydub 从字节流加载 MP3 数据
        audio = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
        
        # 转换为单声道
        audio = audio.set_channels(1)
        
        # 转换采样率到目标采样率
        audio = audio.set_frame_rate(target_sample_rate)
        
        # 转换为 16 位 PCM（2 字节 = 16 位）
        audio = audio.set_sample_width(2)
        
        # 获取原始 PCM 数据
        pcm_data = audio.raw_data
        
        return pcm_data
    except Exception as e:
        logger.error(f"转换MP3到PCM失败: {e}")
        return b""


async def main():
    """
    主函数：连接豆包 TTS 服务，将文本转换为语音并可选地实时播放
    
    流程：
    1. 解析命令行参数
    2. 初始化音频设备（如果启用播放）
    3. 建立 WebSocket 连接
    4. 启动连接会话
    5. 按句子处理文本，逐字符发送并接收音频
    6. 实时播放音频（如果启用）
    7. 清理资源并关闭连接
    """
    parser = argparse.ArgumentParser(description='豆包 TTS 测试工具')
    parser.add_argument("--appid", default="5919896644", help="APP ID")
    parser.add_argument("--access_token", default="G-o4lEbyzOv9F6cLu9jYhkrOegOjorqU", help="Access Token")
    parser.add_argument("--resource_id", default="seed-tts-2.0", help="Resource ID")
    parser.add_argument("--text", default="你好，我是地瓜君", help="Text to convert")
    parser.add_argument("--voice_type", default="zh_male_m191_uranus_bigtts", help="Voice type")
    parser.add_argument("--encoding", default="mp3", help="Output file encoding")
    parser.add_argument(
        "--endpoint",
        default="wss://openspeech.bytedance.com/api/v3/tts/bidirection",
        help="WebSocket endpoint URL",
    )
    parser.add_argument("--enable_playback", action="store_true", help="Enable real-time audio playback")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate for playback")

    args = parser.parse_args()
    
    # 初始化音频设备（如果启用播放）
    audio_device = None
    if args.enable_playback:
        logger.info(f"初始化音频设备,采样率: {args.sample_rate}")
        audio_device = AudioDevice(
            sample_rate=args.sample_rate,
            channels=1,
            chunk_size=1024,
            enable_aec=False  # TTS 播放不需要回声消除
        )
        audio_device.start_streams()
        logger.info("音频设备已启动")

    # 构建 WebSocket 连接请求头
    headers = {
        "X-Api-App-Key": args.appid,
        "X-Api-Access-Key": args.access_token,
        "X-Api-Resource-Id": (
            args.resource_id if args.resource_id else get_resource_id(args.voice_type)
        ),
        "X-Api-Connect-Id": str(uuid.uuid4()),  # 生成唯一的连接 ID
    }

    # 连接到 WebSocket 服务器
    logger.info(f"Connecting to {args.endpoint} with headers: {headers}")
    websocket = await websockets.connect(
        args.endpoint, additional_headers=headers, max_size=10 * 1024 * 1024
    )
    logger.info(
        f"Connected to WebSocket server, Logid: {websocket.response.headers['x-tt-logid']}",
    )

    try:
        # 启动连接
        await start_connection(websocket)
        await wait_for_event(
            websocket, MsgType.FullServerResponse, EventType.ConnectionStarted
        )

        # 按句号分割文本，逐句处理
        sentences = args.text.split("。")
        audio_received = False  # 标记是否接收到音频数据

        for i, sentence in enumerate(sentences):
            if not sentence:
                continue

            # 构建基础请求参数（每个会话可以有不同的参数）
            base_request = {
                "user": {
                    "uid": str(uuid.uuid4()),  # 生成唯一的用户 ID
                },
                "namespace": "BidirectionalTTS",
                "req_params": {
                    "speaker": args.voice_type,  # 语音类型
                    "audio_params": {
                        "format": args.encoding,  # 音频编码格式
                        "sample_rate": 24000,  # 服务器端采样率
                        "enable_timestamp": True,  # 启用时间戳
                    },
                    "additions": json.dumps(
                        {
                            "disable_markdown_filter": False,  # 不禁用 Markdown 过滤
                        }
                    ),
                },
            }

            # 启动会话
            start_session_request = copy.deepcopy(base_request)
            start_session_request["event"] = EventType.StartSession
            session_id = str(uuid.uuid4())  # 生成唯一的会话 ID
            await start_session(
                websocket, json.dumps(start_session_request).encode(), session_id
            )
            await wait_for_event(
                websocket, MsgType.FullServerResponse, EventType.SessionStarted
            )

            # 逐字符发送文本（异步函数）
            async def send_chars():
                for char in sentence:
                    synthesis_request = copy.deepcopy(base_request)
                    synthesis_request["event"] = EventType.TaskRequest
                    synthesis_request["req_params"]["text"] = char
                    await task_request(
                        websocket, json.dumps(synthesis_request).encode(), session_id
                    )
                    await asyncio.sleep(0.005)  # 字符间延迟 5ms，避免发送过快

                # 发送会话结束请求
                await finish_session(websocket, session_id)

            # 在后台任务中开始发送字符
            send_task = asyncio.create_task(send_chars())

            # 接收音频数据
            audio_data = bytearray()  # 累积所有接收到的音频数据
            mp3_buffer = bytearray()  # 用于累积 MP3 数据
            playback_started = False  # 是否已开始播放
            min_buffer_size = 8192 if audio_device else 0  # 最小 MP3 缓冲大小（约 8KB）
            mp3_convert_threshold = 4096  # MP3 转换阈值，累积到这个大小再转换
            
            # 循环接收消息直到会话结束
            while True:
                msg = await receive_message(websocket)

                if msg.type == MsgType.FullServerResponse:
                    # 处理服务器完整响应消息
                    if msg.event == EventType.SessionFinished:
                        # 会话结束，转换并播放剩余的 MP3 数据
                        if audio_device and len(mp3_buffer) > 0:
                            logger.info(f"会话结束,转换剩余MP3数据 {len(mp3_buffer)} bytes")
                            if args.encoding == "mp3":
                                pcm_data = convert_mp3_to_pcm(bytes(mp3_buffer), args.sample_rate)
                                if pcm_data:
                                    audio_device.put_playback_data(pcm_data)
                                    logger.info(f"添加剩余PCM数据: {len(pcm_data)} bytes")
                            mp3_buffer.clear()
                            playback_started = True
                        break
                elif msg.type == MsgType.AudioOnlyServer:
                    # 处理音频数据消息
                    if not audio_received and len(msg.payload) > 0:
                        audio_received = True  # 标记已接收到音频
                    
                    # 收集所有音频数据
                    audio_data.extend(msg.payload)
                    
                    # 实时播放音频片段（优化的批量转换策略）
                    if audio_device and msg.payload:
                        try:
                            if args.encoding == "mp3":
                                # MP3 格式：累积 MP3 数据
                                mp3_buffer.extend(msg.payload)
                                
                                # 检查是否达到最小缓冲大小，开始播放
                                if not playback_started and len(mp3_buffer) >= min_buffer_size:
                                    logger.info(f"MP3缓冲已满({len(mp3_buffer)} bytes),开始播放")
                                    playback_started = True
                                
                                # 批量转换策略：累积足够的 MP3 数据再转换，减少转换次数
                                if playback_started and len(mp3_buffer) >= mp3_convert_threshold:
                                    # 一次性转换累积的 MP3 数据
                                    pcm_data = convert_mp3_to_pcm(bytes(mp3_buffer), args.sample_rate)
                                    if pcm_data:
                                        # 直接放入播放队列，让 AudioDevice 的缓冲区处理
                                        audio_device.put_playback_data(pcm_data)
                                        logger.info(f"转换并添加PCM: {len(pcm_data)} bytes (来自{len(mp3_buffer)} MP3), 队列: {audio_device.get_playback_queue_size()}")
                                        mp3_buffer.clear()
                                    else:
                                        logger.warning(f"MP3转PCM失败")
                                        mp3_buffer.clear()
                                        
                            elif args.encoding == "pcm":
                                # PCM 格式：直接放入播放队列
                                audio_device.put_playback_data(msg.payload)
                                if not playback_started:
                                    playback_started = True
                                logger.info(f"添加PCM数据: {len(msg.payload)} bytes, 队列: {audio_device.get_playback_queue_size()}")
                        except Exception as e:
                            logger.error(f"播放音频片段失败: {e}")
                else:
                    # 未知消息类型，抛出异常
                    raise RuntimeError(f"TTS conversion failed: {msg}")

            # 等待字符发送任务完成
            await send_task
            
            # 等待播放队列播放完成
            if audio_device:
                logger.info(f"等待播放完成,当前队列大小: {audio_device.get_playback_queue_size()}")
                # 先等待队列有足够的数据
                await asyncio.sleep(0.5)
                
                # 然后等待队列播放完
                max_wait = 30  # 最多等待 30 秒
                wait_count = 0
                while audio_device.get_playback_queue_size() > 0 and wait_count < max_wait * 10:
                    await asyncio.sleep(0.1)  # 每 100ms 检查一次
                    wait_count += 1
                
                # 额外等待一段时间确保最后的音频播放完
                await asyncio.sleep(1.0)
                logger.info("音频播放完成")

            logger.info(f"会话 {i} 完成,总接收: {len(audio_data)} bytes")

        # 检查是否接收到音频数据
        if not audio_received:
            raise RuntimeError("No audio data received")

    finally:
        # 确保所有音频播放完成
        if audio_device:
            logger.info(f"等待所有音频播放完成,当前队列大小: {audio_device.get_playback_queue_size()}")
            max_wait = 30  # 最多等待 30 秒
            wait_count = 0
            while audio_device.get_playback_queue_size() > 0 and wait_count < max_wait * 10:
                await asyncio.sleep(0.1)  # 每 100ms 检查一次
                wait_count += 1
            # 额外等待确保播放完成
            await asyncio.sleep(1.0)
        
        # 结束连接
        await finish_connection(websocket)
        msg = await wait_for_event(
            websocket, MsgType.FullServerResponse, EventType.ConnectionFinished
        )
        await websocket.close()
        logger.info("Connection closed")
        
        # 清理音频设备资源
        if audio_device:
            audio_device.cleanup()
            logger.info("音频设备已清理")


if __name__ == "__main__":
    asyncio.run(main())
