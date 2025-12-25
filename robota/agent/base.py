from abc import ABC
from typing import List, Optional, Dict, Any, AsyncIterator, Union

from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.agents import create_agent

class RobotAgent(ABC):
    def __init__(self, model: BaseChatModel, tools: List[BaseTool], 
    *,
    system_prompt: Optional[str] = None,
    max_iterations: int = 10,
    ):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )
        
        # 创建工具名称到工具对象的映射
        self.tools_by_name = {tool.name: tool for tool in self.tools}
    
    def invoke(self, messages: Dict[str, Any]) -> str:
        return self.agent.invoke(messages)
    
    async def ainvoke(self, messages: Dict[str, Any]) -> str:
        return await self.agent.ainvoke(messages)
    
    def stream(self, messages: Dict[str, Any], stream_mode: str = "messages"):
        """
        Stream agent responses.
        
        Args:
            messages: Input messages
            stream_mode: Stream mode - "messages" for LLM tokens, "updates" for agent progress
        
        Returns:
            If stream_mode="messages": Iterator of (token, metadata) tuples
            If stream_mode="updates": Iterator of state update dictionaries
        """
        return self.agent.stream(
            messages, 
            stream_mode=stream_mode,
        )
    
    async def astream(self, messages: Dict[str, Any], stream_mode: str = "messages"):
        """
        Async stream agent responses.
        
        Args:
            messages: Input messages
            stream_mode: Stream mode - "messages" for LLM tokens, "updates" for agent progress
        
        Returns:
            If stream_mode="messages": AsyncIterator of (token, metadata) tuples
            If stream_mode="updates": AsyncIterator of state update dictionaries
        """
        async for chunk in self.agent.astream(
            messages,
            stream_mode=stream_mode,
        ):
            yield chunk
    
    async def astream_with_tools(self, user_input: str) -> AsyncIterator[Dict[str, Any]]:
        """
        真正的流式输出，支持工具调用且实时打印每个token。
        
        Args:
            user_input: 用户输入
            
        Yields:
            字典，包含以下可能的键：
            - type: "token" | "tool_call_start" | "tool_call_end" | "error" | "done"
            - content: 文本内容（type="token"时）
            - tool_name: 工具名称（type="tool_call_*"时）
            - tool_args: 工具参数（type="tool_call_start"时）
            - tool_result: 工具结果（type="tool_call_end"时）
            - error: 错误信息（type="error"时）
        """
        # 构建消息历史
        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=user_input))
        
        # 绑定工具的模型
        model_with_tools = self.model.bind_tools(self.tools)
        
        # 工具调用循环
        for iteration in range(self.max_iterations):
            # 流式调用模型
            full_response = None
            async for chunk in model_with_tools.astream(messages):
                # 累积完整响应
                if full_response is None:
                    full_response = chunk
                else:
                    full_response += chunk
                
                # 实时输出文本token
                if hasattr(chunk, 'content') and chunk.content:
                    yield {
                        "type": "token",
                        "content": chunk.content,
                    }
            
            # 将完整响应添加到消息历史
            messages.append(full_response)
            
            # 检查是否有工具调用
            if not full_response.tool_calls:
                # 没有工具调用，结束
                yield {"type": "done"}
                break
            
            # 执行工具调用
            for tool_call in full_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]
                
                yield {
                    "type": "tool_call_start",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                }
                
                # 查找并执行工具
                if tool_name in self.tools_by_name:
                    tool = self.tools_by_name[tool_name]
                    try:
                        # 异步调用工具
                        if hasattr(tool, 'ainvoke'):
                            tool_result = await tool.ainvoke(tool_args)
                        else:
                            # 同步工具，在异步上下文中调用
                            import asyncio
                            tool_result = await asyncio.to_thread(tool.invoke, tool_args)
                        
                        # 将工具结果添加到消息历史
                        messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call_id,
                        ))
                        
                        yield {
                            "type": "tool_call_end",
                            "tool_name": tool_name,
                            "tool_result": tool_result,
                        }
                    except Exception as e:
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
                    error_msg = f"工具 {tool_name} 不存在"
                    yield {
                        "type": "error",
                        "tool_name": tool_name,
                        "error": error_msg,
                    }
        else:
            # 达到最大迭代次数
            yield {
                "type": "error",
                "error": f"达到最大迭代次数 {self.max_iterations}",
            }
    