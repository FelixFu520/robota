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
    if voice.startswith("S_"):
        return "volc.megatts.default"
    return "volc.service_type.10029"


def convert_mp3_to_pcm(mp3_data: bytes, target_sample_rate: int = 16000) -> bytes:
    """将MP3数据转换为PCM格式"""
    try:
        # 使用pydub加载MP3数据
        audio = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
        
        # 转换为单声道
        audio = audio.set_channels(1)
        
        # 转换采样率
        audio = audio.set_frame_rate(target_sample_rate)
        
        # 转换为16位PCM
        audio = audio.set_sample_width(2)  # 2 bytes = 16 bits
        
        # 获取原始PCM数据
        pcm_data = audio.raw_data
        
        return pcm_data
    except Exception as e:
        logger.error(f"转换MP3到PCM失败: {e}")
        return b""


async def main():
    parser = argparse.ArgumentParser()
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
    
    # 初始化音频设备(如果启用播放)
    audio_device = None
    if args.enable_playback:
        logger.info(f"初始化音频设备,采样率: {args.sample_rate}")
        audio_device = AudioDevice(
            sample_rate=args.sample_rate,
            channels=1,
            chunk_size=1024,
            enable_aec=False  # TTS播放不需要回声消除
        )
        audio_device.start_streams()
        logger.info("音频设备已启动")

    # Connect to server
    headers = {
        "X-Api-App-Key": args.appid,
        "X-Api-Access-Key": args.access_token,
        "X-Api-Resource-Id": (
            args.resource_id if args.resource_id else get_resource_id(args.voice_type)
        ),
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }

    logger.info(f"Connecting to {args.endpoint} with headers: {headers}")
    websocket = await websockets.connect(
        args.endpoint, additional_headers=headers, max_size=10 * 1024 * 1024
    )
    logger.info(
        f"Connected to WebSocket server, Logid: {websocket.response.headers['x-tt-logid']}",
    )

    try:
        # Start connection
        await start_connection(websocket)
        await wait_for_event(
            websocket, MsgType.FullServerResponse, EventType.ConnectionStarted
        )

        # Process each sentence
        sentences = args.text.split("。")
        audio_received = False

        for i, sentence in enumerate(sentences):
            if not sentence:
                continue

            # every session can have different parameters
            base_request = {
                "user": {
                    "uid": str(uuid.uuid4()),
                },
                "namespace": "BidirectionalTTS",
                "req_params": {
                    "speaker": args.voice_type,
                    "audio_params": {
                        "format": args.encoding,
                        "sample_rate": 24000,
                        "enable_timestamp": True,
                    },
                    "additions": json.dumps(
                        {
                            "disable_markdown_filter": False,
                        }
                    ),
                },
            }

            # Start session
            start_session_request = copy.deepcopy(base_request)
            start_session_request["event"] = EventType.StartSession
            session_id = str(uuid.uuid4())
            await start_session(
                websocket, json.dumps(start_session_request).encode(), session_id
            )
            await wait_for_event(
                websocket, MsgType.FullServerResponse, EventType.SessionStarted
            )

            # Send characters one by one
            async def send_chars():
                for char in sentence:
                    synthesis_request = copy.deepcopy(base_request)
                    synthesis_request["event"] = EventType.TaskRequest
                    synthesis_request["req_params"]["text"] = char
                    await task_request(
                        websocket, json.dumps(synthesis_request).encode(), session_id
                    )
                    await asyncio.sleep(0.005)  # 5ms delay between characters

                await finish_session(websocket, session_id)

            # Start sending characters in background
            send_task = asyncio.create_task(send_chars())

            # Receive audio data
            audio_data = bytearray()
            while True:
                msg = await receive_message(websocket)

                if msg.type == MsgType.FullServerResponse:
                    if msg.event == EventType.SessionFinished:
                        break
                elif msg.type == MsgType.AudioOnlyServer:
                    if not audio_received and len(msg.payload) > 0:
                        audio_received = True
                    
                    # 先收集音频数据
                    audio_data.extend(msg.payload)
                    
                    # 实时播放音频片段
                    if audio_device and msg.payload:
                        try:
                            if args.encoding == "mp3":
                                # 将MP3转换为PCM格式
                                pcm_data = convert_mp3_to_pcm(msg.payload, args.sample_rate)
                                if pcm_data:
                                    # 将PCM数据分块放入播放队列
                                    chunk_size = audio_device.chunk_size * 2  # 2 bytes per sample
                                    chunks_added = 0
                                    for j in range(0, len(pcm_data), chunk_size):
                                        chunk = pcm_data[j:j+chunk_size]
                                        audio_device.put_playback_data(chunk)
                                        chunks_added += 1
                                    logger.info(f"播放音频片段: {len(pcm_data)} bytes, 分为 {chunks_added} 个块, 队列大小: {audio_device.get_playback_queue_size()}")
                                else:
                                    logger.warning(f"MP3转PCM失败,跳过该片段")
                            elif args.encoding == "pcm":
                                # PCM格式直接播放
                                chunk_size = audio_device.chunk_size * 2
                                chunks_added = 0
                                for j in range(0, len(msg.payload), chunk_size):
                                    chunk = msg.payload[j:j+chunk_size]
                                    audio_device.put_playback_data(chunk)
                                    chunks_added += 1
                                logger.info(f"播放音频片段: {len(msg.payload)} bytes, 分为 {chunks_added} 个块, 队列大小: {audio_device.get_playback_queue_size()}")
                        except Exception as e:
                            logger.error(f"播放音频片段失败: {e}")
                else:
                    raise RuntimeError(f"TTS conversion failed: {msg}")

            # Wait for send_chars to complete
            await send_task
            
            # 等待播放队列播放完成
            if audio_device:
                logger.info(f"等待播放完成,当前队列大小: {audio_device.get_playback_queue_size()}")
                # 先等待队列有足够的数据
                await asyncio.sleep(0.5)
                
                # 然后等待队列播放完
                max_wait = 30  # 最多等待30秒
                wait_count = 0
                while audio_device.get_playback_queue_size() > 0 and wait_count < max_wait * 10:
                    await asyncio.sleep(0.1)
                    wait_count += 1
                
                # 额外等待一段时间确保最后的音频播放完
                await asyncio.sleep(1.0)
                logger.info("音频播放完成")

            logger.info(f"会话 {i} 完成,总接收: {len(audio_data)} bytes")

        if not audio_received:
            raise RuntimeError("No audio data received")

    finally:
        # 确保所有音频播放完成
        if audio_device:
            logger.info(f"等待所有音频播放完成,当前队列大小: {audio_device.get_playback_queue_size()}")
            max_wait = 30
            wait_count = 0
            while audio_device.get_playback_queue_size() > 0 and wait_count < max_wait * 10:
                await asyncio.sleep(0.1)
                wait_count += 1
            # 额外等待确保播放完成
            await asyncio.sleep(1.0)
        
        # Finish connection
        await finish_connection(websocket)
        msg = await wait_for_event(
            websocket, MsgType.FullServerResponse, EventType.ConnectionFinished
        )
        await websocket.close()
        logger.info("Connection closed")
        
        # 清理音频设备
        if audio_device:
            audio_device.cleanup()
            logger.info("音频设备已清理")


if __name__ == "__main__":
    asyncio.run(main())
