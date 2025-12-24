"""
Turtlesim Agent for ROS 2 robot.

测试使用
"""
import os
import asyncio
from typing import List
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from robota.model import DOUBAO_SEED_1_6_251015
from robota.agent.base import RobotAgent

class TurtlesimAgent(RobotAgent):
    def __init__(self, tools: List[BaseTool] = None):
        """
        Initialize TurtlesimAgent.
        
        Args:
            tools: Optional list of tools. If None, will try to initialize tools.
                  Note: In async contexts, use TurtlesimAgent.create() instead.
        """
        if tools is None:
            tools = self._get_tools_sync()
        
        super().__init__(
            model=DOUBAO_SEED_1_6_251015,
            tools=tools,
            system_prompt="You are a helpful ROS 2 robot assistant that can control a Turtlesim robot."
        )
    
    @classmethod
    async def create(cls):
        """
        Async factory method to create TurtlesimAgent.
        Use this when you're already in an async context.
        
        Example:
            agent = await TurtlesimAgent.create()
        """
        tools = await cls._get_tools_async()
        return cls(tools=tools)
    
    @staticmethod
    def _get_tools_sync() -> List[BaseTool]:
        """Synchronously get tools from MCP server."""
        async def _get_tools():
            math_server_path = os.path.join(os.path.dirname(__file__), "..", "mcp", "math.py")
            math_server_path = os.path.abspath(math_server_path)
            command = f"python {math_server_path}"

            mcp_client = MultiServerMCPClient({
                "math": {
                    "transport": "stdio",
                    "command": command,
                },
            })
            return await mcp_client.get_tools()
        
        # Check if there's a running event loop
        try:
            asyncio.get_running_loop()
            # If we get here, there's a running loop, can't use asyncio.run()
            raise RuntimeError(
                "Cannot initialize tools synchronously when an event loop is already running. "
                "Use 'await TurtlesimAgent.create()' instead of 'TurtlesimAgent()'."
            )
        except RuntimeError as e:
            # Check if it's our custom error or the "no running loop" error
            if "Cannot initialize tools" in str(e):
                raise
            # No running loop, safe to use asyncio.run()
            return asyncio.run(_get_tools())
    
    @staticmethod
    async def _get_tools_async() -> List[BaseTool]:
        """Asynchronously get tools from MCP server."""
        math_server_path = os.path.join(os.path.dirname(__file__), "..", "mcp", "math.py")
        math_server_path = os.path.abspath(math_server_path)
        command = f"python {math_server_path}"

        mcp_client = MultiServerMCPClient({
            "math": {
                "transport": "stdio",
                "command": command,
            },
        })
        return await mcp_client.get_tools()
