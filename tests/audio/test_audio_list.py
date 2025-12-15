"""
ALSA: https://zhuanlan.zhihu.com/p/813448431
"""
import pyaudio
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="all")
    args = parser.parse_args()

    device_list = []

    print(f"正在扫描,音频设备: {args.device} ...")
    # 重定向错误输出
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if args.device == "all":
            device_list.append(dev)
        elif args.device in dev['name']:
            device_list.append(dev)
    p.terminate()

    for dev in device_list:
        print(dev)