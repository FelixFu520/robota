import argparse
import asyncio

from robota.audio.volcengine_doubao_asr import AsrWsClient
from robota.utils.logging import default_logger as logger

async def main():
    
    parser = argparse.ArgumentParser(description="ASR WebSocket Client")
    parser.add_argument("--file", type=str, default="/home/drobotics/projects/robota/assets/xiaozhan_ref_16000_concatenated.wav", help="Audio file path")

    parser.add_argument("--url", type=str, default="wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async", 
                       help="WebSocket URL")
    parser.add_argument("--seg-duration", type=int, default=200, 
                       help="Audio duration(ms) per packet, default:200")
    
    args = parser.parse_args()
    
    async with AsrWsClient(args.url, args.seg_duration) as client:  # 使用async with
        try:
            async for response in client.execute(args.file):
                if "text" in response.to_dict()['payload_msg']['result']:
                    print(response.to_dict()['payload_msg']['result']['text'] + "   " + str(response.to_dict()['payload_msg']['result']['utterances'][0]['definite']))
                # logger.info(f"Received response: {json.dumps(response.to_dict(), indent=2, ensure_ascii=False)}")
        except Exception as e:
            logger.error(f"ASR processing failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())

    # 用法：
    # python3 sauc_websocket_demo.py --file /Users/bytedance/code/python/eng_ddc_itn.wav