"""
测试agent的流式输出和工具调用
测试stream的messages和updates两种模式
messages模式：输出LLM生成的token
updates模式：输出Agent状态更新，包括工具调用等
"""

import os
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from typing import Union

 
@tool
def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    计算两数之和
    
    Args:
        a: 第一个数
        b: 第二个数

    Returns:
        两数之和

    Examples:
        >>> add(1, 2)
        3
        >>> add(-1, 1)
        0
        >>> add(0, 0)
        0
    """
    return 1234567890

model = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    streaming=True,
    extra_body={"thinking": {"type": "disabled"}},
)

agent = create_agent(
            model=model,
            tools=[add],  # 启用工具以测试工具调用
            system_prompt="""
                你是个智能助手，可以和用户聊天。
                重要：当你调用工具后，必须完全使用工具返回的结果，不要自己计算或推断。
                如果工具返回的结果与你预期的不同，也要如实报告工具返回的结果。""",
)


def test_stream():
    """测试agent的流式输出和工具调用"""
    chunk_count = 0
    try:
        for token, metadata in agent.stream(
            {"messages": [{"role": "user", "content": "介绍你自己"}]},
            stream_mode="messages",
        ):
            # 从token中提取文本内容
            content = None
            if hasattr(token, "text"):
                content = token.text
            
            if content and content.strip():
                chunk_count += 1
                print(f"{content}", end="", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
    print(f"\n总共收到 {chunk_count} 个文本chunk")


    print('\n' + '-' * 60 + '\n')

    chunk_count = 0
    tool_call_count = 0
    update_count = 0
    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": "计算34253264357643655234 + 22134213495732940587391465 等于多少"}]},
            stream_mode="updates",
        ):
            update_count += 1
            # chunk是一个字典，key是节点名，value是状态更新
            for step_name, state_update in chunk.items():
                # 检查是否有消息更新
                if "messages" in state_update:
                    messages = state_update["messages"]
                    if messages:
                        last_message = messages[-1]
                        
                        # 检查是否是工具调用消息
                        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                            tool_call_count += 1
                            print(f"  🔧 工具调用 #{tool_call_count}:")
                            for tool_call in last_message.tool_calls:
                                tool_name = tool_call.get('name', 'unknown') if isinstance(tool_call, dict) else getattr(tool_call, 'name', 'unknown')
                                tool_args = tool_call.get('args', {}) if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})
                                print(f"     工具名: {tool_name}")
                                print(f"     参数: {tool_args}")
                        
                        # 检查是否是工具返回消息
                        elif hasattr(last_message, "content") and hasattr(last_message, "tool_call_id"):
                            print(f"  ✅ 工具返回: {last_message.content}")
                        
                        # 检查是否是普通文本消息
                        elif hasattr(last_message, "content") and last_message.content:
                            content_preview = str(last_message.content)[:100]
                            if len(str(last_message.content)) > 100:
                                content_preview += "..."
                            print(f"  💬 消息: {content_preview}")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    print('\n' + '=' * 60)
    print("总结:")
    print(f"  - 文本chunk数量: {chunk_count}")
    print(f"  - 状态更新数量: {update_count}")
    print(f"  - 工具调用次数: {tool_call_count}")
    if tool_call_count > 0:
        print("  ✓ Agent成功调用了工具")
    else:
        print("  ⚠ Agent没有调用工具（可能工具调用被合并到文本中，或模型选择不使用工具）")
    print()
    

if __name__ == "__main__":
    test_stream()