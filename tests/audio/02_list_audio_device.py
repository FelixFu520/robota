#!/usr/bin/env python3
"""
音频设备列表脚本
列出系统中所有可用的音频设备，支持按设备名称过滤

参考: ALSA: https://zhuanlan.zhihu.com/p/813448431
"""
import pyaudio
import argparse


def list_audio_devices(device_filter="all"):
    """
    列出音频设备
    
    Args:
        device_filter: 设备过滤条件，"all" 表示列出所有设备，否则按设备名称过滤
    """
    device_list = []
    
    print(f"正在扫描音频设备: {device_filter} ...")
    
    # 初始化 PyAudio
    p = pyaudio.PyAudio()
    
    # 遍历所有音频设备
    for i in range(p.get_device_count()):
        # 获取设备信息
        dev = p.get_device_info_by_index(i)
        
        # 根据过滤条件添加设备
        if device_filter == "all":
            device_list.append(dev)
        elif device_filter in dev['name']:
            device_list.append(dev)
    
    # 关闭 PyAudio
    p.terminate()
    
    # 输出设备信息
    for dev in device_list:
        print(dev)
    
    return device_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='音频设备列表工具')
    parser.add_argument("--device", type=str, default="all", 
                       help='设备过滤条件，"all" 表示列出所有设备，否则按设备名称过滤（默认: all）')
    args = parser.parse_args()
    
    # 执行设备列表查询
    list_audio_devices(args.device)