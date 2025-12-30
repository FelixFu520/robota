from typing import Union
import json
from datetime import datetime
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from robota.agent.base import RobotAgent
from robota.model import DOUBAO_SEED_1_6_251015_NOTHINKING


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
    return 567868

def is_tool_message(token):
    """
    检查 token 是否是工具返回的消息（ToolMessage）
    
    Args:
        token: 从 agent.stream 返回的 token 对象
        
    Returns:
        bool: 如果是 ToolMessage 返回 True，否则返回 False
    """
    return (
        isinstance(token, ToolMessage) or
        (hasattr(token, "tool_call_id") and token.tool_call_id) or
        (hasattr(token, "__class__") and "ToolMessage" in token.__class__.__name__)
    )

def extract_object_info(obj, max_depth=3, current_depth=0):
    """
    递归提取对象的所有属性信息，转换为可序列化的字典
    
    Args:
        obj: 要提取信息的对象
        max_depth: 最大递归深度，防止无限递归
        current_depth: 当前递归深度
        
    Returns:
        dict: 包含对象信息的字典
    """
    if current_depth >= max_depth:
        return {"_truncated": True, "_reason": "max_depth_reached"}
    
    if obj is None:
        return None
    
    # 处理基本类型
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # 处理列表和元组
    if isinstance(obj, (list, tuple)):
        return [extract_object_info(item, max_depth, current_depth + 1) for item in obj]
    
    # 处理字典
    if isinstance(obj, dict):
        return {k: extract_object_info(v, max_depth, current_depth + 1) for k, v in obj.items()}
    
    # 处理对象
    info = {
        "_type": obj.__class__.__name__,
        "_module": getattr(obj.__class__, "__module__", "unknown"),
    }
    
    # 提取所有属性
    if hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            # 跳过私有属性（以双下划线开头且不以双下划线结尾的）
            if not (key.startswith("__") and not key.endswith("__")):
                try:
                    info[key] = extract_object_info(value, max_depth, current_depth + 1)
                except Exception as e:
                    info[key] = {"_error": str(e)}
    
    # 提取常用属性
    common_attrs = [
        "content", "text", "role", "name", "tool_calls", "tool_call_id",
        "id", "type", "additional_kwargs", "response_metadata"
    ]
    for attr in common_attrs:
        if hasattr(obj, attr):
            try:
                value = getattr(obj, attr)
                if value is not None:
                    info[attr] = extract_object_info(value, max_depth, current_depth + 1)
            except Exception as e:
                info[f"{attr}_error"] = str(e)
    
    # 如果有__str__方法，也保存字符串表示
    try:
        str_repr = str(obj)
        if len(str_repr) < 500:  # 只保存较短的字符串表示
            info["_str"] = str_repr
    except:
        pass
    
    return info

def test_stream(agent: RobotAgent):
    # 收集所有token和metadata的详细信息
    all_tokens_data = []
    token_index = 0

    for token, metadata in agent.stream({"messages": [{"role": "user", "content": "你好，介绍你自己，然后计算234235+ 3245324643573457634 等于多少"}]}):
        token_index += 1
        
        # 提取token和metadata的详细信息
        token_info = extract_object_info(token)
        metadata_info = extract_object_info(metadata)
        
        # 判断是否是工具消息
        is_tool = is_tool_message(token)
        
        # 提取文本内容（如果有）
        text_content = None
        if hasattr(token, "text") and token.text:
            text_content = token.text
        
        # 保存当前token和metadata的信息
        token_data = {
            "index": token_index,
            "is_tool_message": is_tool,
            "text_content": text_content,
            "token": token_info,
            "metadata": metadata_info,
            "timestamp": datetime.now().isoformat()
        }
        all_tokens_data.append(token_data)
        
        # 原有的输出逻辑：过滤掉工具返回的消息（ToolMessage），只输出 LLM 生成的文本消息
        if is_tool:
            continue
        if text_content:
            print(text_content, end="", flush=True)
    
    print('\n')
    
    # 保存为JSON文件
    output_file = "token_metadata_analysis.json"
    output_data = {
        "summary": {
            "total_tokens": len(all_tokens_data),
            "tool_messages_count": sum(1 for item in all_tokens_data if item["is_tool_message"]),
            "text_messages_count": sum(1 for item in all_tokens_data if item["text_content"]),
            "generated_at": datetime.now().isoformat(),
            "input_message": "你好，介绍你自己，然后计算234235+ 3245324643573457634 等于多少"
        },
        "tokens": all_tokens_data
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 已将所有 {len(all_tokens_data)} 个token和metadata的详细信息保存到: {output_file}")
    print(f"   - 工具消息数量: {output_data['summary']['tool_messages_count']}")
    print(f"   - 文本消息数量: {output_data['summary']['text_messages_count']}")


if __name__ == "__main__":
    agent = RobotAgent(
        model=DOUBAO_SEED_1_6_251015_NOTHINKING,
        tools=[add],
        system_prompt="""你是个智能助手，可以和用户聊天。"""
    )
    assert agent is not None

    test_stream(agent)
