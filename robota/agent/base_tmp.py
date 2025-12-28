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
    
    def invoke(self, messages: Dict[str, Any]) -> str:
        """
        同步调用Agent.
        
        Args:
            messages: 输入消息字典,包含对话历史和当前用户输入
            
        Returns:
            Agent生成的响应字符串
        """
        return self.agent.invoke(messages)
    
    async def ainvoke(self, messages: Dict[str, Any]) -> str:
        """
        异步调用Agent.
        
        Args:
            messages: 输入消息字典,包含对话历史和当前用户输入
            
        Returns:
            Agent生成的响应字符串
        """
        return await self.agent.ainvoke(messages)
    
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
    
    async def astream(self, messages: Dict[str, Any], stream_mode: str = "messages"):
        """
        异步流式输出Agent响应.
        
        Args:
            messages: 输入消息字典,包含对话历史和当前用户输入
            stream_mode: 流式模式,可选值：
                - "messages": 流式输出 LLM 生成的 token
                - "updates": 流式输出Agent状态更新(包括工具调用等)
        
        Yields:
            根据 stream_mode 返回不同类型的数据：
            - stream_mode="messages": 异步迭代器,产生 (token, metadata) 元组
            - stream_mode="updates": 异步迭代器,产生状态更新字典
        """
        async for chunk in self.agent.astream(
            messages,
            stream_mode=stream_mode,
        ):
            yield chunk
    
    async def astream_with_tools(self, user_input: str) -> AsyncIterator[Dict[str, Any]]:
        """
        真正的流式输出,支持工具调用且实时打印每个token.
        
        Args:
            user_input: 用户输入
            
        Yields:
            字典,包含以下可能的键：
            - type: "token" | "tool_call_start" | "tool_call_end" | "error" | "done"
            - content: 文本内容(type="token"时)
            - tool_name: 工具名称(type="tool_call_*"时)
            - tool_args: 工具参数(type="tool_call_start"时)
            - tool_result: 工具结果(type="tool_call_end"时)
            - error: 错误信息(type="error"时)
        """
        # 构建消息历史,包含系统提示和用户输入
        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=user_input))
        
        # 将工具绑定到模型,使模型能够识别和调用这些工具
        model_with_tools = self.model.bind_tools(self.tools)
        
        # 工具调用循环：Agent可能需要多次调用工具才能完成任务
        for iteration in range(self.max_iterations):
            # 流式调用模型,获取响应
            full_response = None
            async for chunk in model_with_tools.astream(messages):
                # 累积完整响应：将流式返回的多个 chunk 合并成完整响应
                if full_response is None:
                    full_response = chunk
                else:
                    full_response += chunk
                
                # 实时输出文本 token：在生成过程中立即输出每个 token
                if hasattr(chunk, 'content') and chunk.content:
                    yield {
                        "type": "token",
                        "content": chunk.content,
                    }
            
            # 将完整响应添加到消息历史,供后续迭代使用
            messages.append(full_response)
            
            # 检查是否有工具调用：如果没有工具调用,说明任务已完成
            if not full_response.tool_calls:
                # 没有工具调用,任务完成,结束循环
                yield {"type": "done"}
                break
            
            # 执行工具调用：处理模型请求的所有工具调用
            for tool_call in full_response.tool_calls:
                tool_name = tool_call["name"]  # 工具名称
                tool_args = tool_call["args"]  # 工具参数
                tool_call_id = tool_call["id"]  # 工具调用 ID,用于关联结果
                
                # 通知工具调用开始
                yield {
                    "type": "tool_call_start",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                }
                
                # 查找并执行工具
                if tool_name in self.tools_by_name:
                    tool = self.tools_by_name[tool_name]
                    try:
                        # 优先使用异步方法调用工具
                        if hasattr(tool, 'ainvoke'):
                            tool_result = await tool.ainvoke(tool_args)
                        else:
                            # 如果工具只有同步方法,在线程池中执行以避免阻塞
                            import asyncio
                            tool_result = await asyncio.to_thread(tool.invoke, tool_args)
                        
                        # 将工具结果添加到消息历史,供模型在下一轮迭代中使用
                        messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call_id,
                        ))
                        
                        # 通知工具调用成功完成
                        yield {
                            "type": "tool_call_end",
                            "tool_name": tool_name,
                            "tool_result": tool_result,
                        }
                    except Exception as e:
                        # 工具调用出错,记录错误信息
                        error_msg = f"调用工具 {tool_name} 时出错: {str(e)}"
                        messages.append(ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call_id,
                        ))
                        yield {
                            "type": "error",
                            "tool_name": tool_name,
                            "error": error_msg,
                        }
                else:
                    # 工具不存在,返回错误
                    error_msg = f"工具 {tool_name} 不存在"
                    yield {
                        "type": "error",
                        "tool_name": tool_name,
                        "error": error_msg,
                    }
        else:
            # 循环正常结束(未 break),说明达到最大迭代次数
            # 这通常意味着Agent陷入了循环或需要更多迭代才能完成任务
            yield {
                "type": "error",
                "error": f"达到最大迭代次数 {self.max_iterations}",
            }
    