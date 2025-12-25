from abc import ABC
from typing import List, Optional, Dict, Any, AsyncIterator

from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent

class RobotAgent(ABC):
    def __init__(self, model: BaseChatModel, tools: List[BaseTool], 
    *,
    system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt

        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )
    
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
        return await self.agent.astream(
            messages,
            stream_mode=stream_mode,
        )
    