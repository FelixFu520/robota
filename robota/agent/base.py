from abc import ABC
from typing import List, Optional, Dict, Any

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
    
    async def ainvoke(self, messages: Dict[str, Any]) -> str:
        return await self.agent.ainvoke(messages)