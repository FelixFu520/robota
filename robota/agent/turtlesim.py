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
from robota.model import DOUBAO_SEED_1_6_251015_NOTHINKING, DOUBAO_SEED_1_6_251015
from robota.agent.base import RobotAgent


class TurtlesimAgent(RobotAgent):
    def __init__(self, tools: List[BaseTool] = None):
        super().__init__(
            model=DOUBAO_SEED_1_6_251015_NOTHINKING,
            tools=tools,
            system_prompt="You are a helpful ROS 2 robot assistant that can control a Turtlesim robot."
        )
    
