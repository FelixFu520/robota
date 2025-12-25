#!/usr/bin/env python3
"""
测试不同的模型流式配置
"""
import os
from langchain_openai import ChatOpenAI

# 测试1: 基础流式配置
print("=" * 60)
print("测试1: 基础模型流式输出")
print("=" * 60)

model1 = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    streaming=True,  # 启用流式
)

print("开始流式输出...")
for chunk in model1.stream("请详细介绍一下你自己，包括你的功能"):
    print(chunk.content, end="", flush=True)

print("\n" + "=" * 60)
print("测试2: 带extra_body的流式配置")
print("=" * 60)

model2 = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    streaming=True,
    model_kwargs={
        "extra_body": {
            "thinking": {"type": "disabled"},
            "stream": True,  # 确保流式输出
        }
    }
)

print("开始流式输出...")
for chunk in model2.stream("请详细介绍一下你自己，包括你的功能"):
    print(chunk.content, end="", flush=True)

print("\n" + "=" * 60)
print("测试3: 尝试工具流式参数")
print("=" * 60)

model3 = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    streaming=True,
    model_kwargs={
        "extra_body": {
            "thinking": {"type": "disabled"},
            "stream": True,
            "tool_stream": True,  # 尝试工具流式输出
        }
    }
)

print("开始流式输出...")
for chunk in model3.stream("请详细介绍一下你自己，包括你的功能"):
    print(chunk.content, end="", flush=True)

print("\n🎉 测试完成!")
