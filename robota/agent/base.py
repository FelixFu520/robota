"""
机器人Agent基类模块.

本模块定义了 RobotAgent 抽象基类,提供了基于 LangChain 的智能Agent功能,
支持工具调用、流式输出和异步操作.
"""

from abc import ABC
from typing import List, Optional, Dict, Any, AsyncIterator

from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.agents import create_agent


class RobotAgent(ABC):
    """
    机器人Agent抽象基类.
    
    该类封装了 LangChain Agent的核心功能,提供了统一的接口用于：
    - 同步和异步调用
    - 流式输出响应
    - 工具调用和管理
    - 支持工具调用的流式输出
    
    子类需要实现具体的业务逻辑,此类提供通用的Agent框架.
    """
    def __init__(self, model: BaseChatModel, tools: List[BaseTool], 
    *,
    system_prompt: Optional[str] = None,
    max_iterations: int = 10,
    ):
        """
        初始化机器人Agent.
        
        Args:
            model: LangChain 聊天模型实例,用于生成响应
            tools: 工具列表,Agent可以调用的工具集合
            system_prompt: 可选的系统提示词,用于指导Agent的行为
            max_iterations: 最大迭代次数,限制Agent的工具调用循环次数,默认 10 次
            
        Note:
            max_iterations 用于防止无限循环,当Agent需要多次调用工具时,
            如果超过此限制,将停止执行并返回错误.
        """
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

        # 使用 LangChain 创建Agent实例
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )
        
        # 创建工具名称到工具对象的映射,便于快速查找工具
        self.tools_by_name = {tool.name: tool for tool in self.tools}
    
    def stream(self, messages: Dict[str, Any], stream_mode: str = "messages"):
        """
        同步流式输出Agent响应.
        
        Args:
            messages: 输入消息字典,包含对话历史和当前用户输入
            stream_mode: 流式模式,可选值：
                - "messages": 流式输出 LLM 生成的 token
                - "updates": 流式输出Agent状态更新(包括工具调用等)
        
        Returns:
            迭代器,根据 stream_mode 返回不同类型的数据：
            - stream_mode="messages": 返回 (token, metadata) 元组的迭代器
            - stream_mode="updates": 返回状态更新字典的迭代器
        """
        return self.agent.stream(
            messages, 
            stream_mode=stream_mode,
        )
    