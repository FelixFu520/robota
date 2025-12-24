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
        return self.agent.invoke(messages, extra_body={
            "thinking": {
                "type": "disabled"  # 不使用深度思考能力
                # "type": "enabled" # 使用深度思考能力
                # "type": "auto" # 模型自行判断是否使用深度思考能力
            }
        })
    
    async def ainvoke(self, messages: Dict[str, Any]) -> str:
        return await self.agent.ainvoke(messages, extra_body={
            "thinking": {
                "type": "disabled"  # 不使用深度思考能力
                # "type": "enabled" # 使用深度思考能力
                # "type": "auto" # 模型自行判断是否使用深度思考能力
            }
        })
    
    def stream(self, messages: Dict[str, Any]) -> AsyncIterator[str]:
        return self.agent.stream(messages, extra_body={
            "thinking": {
                "type": "disabled"  # 不使用深度思考能力
                # "type": "enabled" # 使用深度思考能力
                # "type": "auto" # 模型自行判断是否使用深度思考能力
            }
        })
    
    async def astream(self, messages: Dict[str, Any]) -> AsyncIterator[str]:
        return await self.agent.astream(messages, extra_body={
            "thinking": {
                "type": "disabled"  # 不使用深度思考能力
                # "type": "enabled" # 使用深度思考能力
                # "type": "auto" # 模型自行判断是否使用深度思考能力
            }
        })
    