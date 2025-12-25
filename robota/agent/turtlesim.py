"""
Turtlesim Agent 模块.

本模块定义了 TurtlesimAgent 类,用于控制 ROS 2 的 Turtlesim 机器人.
Turtlesim 是 ROS 2 的一个简单的可视化仿真器,用于演示基本的机器人控制功能.
"""

from typing import List
from langchain_core.tools import BaseTool
from robota.model import DOUBAO_SEED_1_6_251015_NOTHINKING
from robota.agent.base import RobotAgent


class TurtlesimAgent(RobotAgent):
    """
    Turtlesim 机器人 Agent.
    
    该类继承自 RobotAgent,专门用于控制 ROS 2 Turtlesim 仿真机器人.
    Turtlesim 是一个简单的海龟机器人仿真器,常用于 ROS 2 学习和测试.
    
    该 Agent 使用豆包模型(无思考模式)作为底层语言模型,并配置了专门用于
    Turtlesim 机器人控制的系统提示词.
    
    示例:
        >>> tools = [move_tool, rotate_tool]  # 定义控制工具
        >>> agent = TurtlesimAgent(tools=tools)
        >>> response = agent.invoke({"input": "让机器人向前移动"})
    """
    
    def __init__(self, tools: List[BaseTool] = None):
        """
        初始化 Turtlesim Agent.
        
        Args:
            tools: 可选的工具列表,用于控制 Turtlesim 机器人.
                  如果为 None,则使用空列表(Agent 将无法调用任何工具).
                  工具通常包括移动、旋转、获取位置等 ROS 2 控制命令.
        
        Note:
            - 使用 DOUBAO_SEED_1_6_251015_NOTHINKING 模型(无思考模式),适合实时控制场景
            - 系统提示词指导 Agent 作为 ROS 2 机器人助手
            - 继承自 RobotAgent 的所有功能,包括同步/异步调用和流式输出
        """
        super().__init__(
            model=DOUBAO_SEED_1_6_251015_NOTHINKING,
            tools=tools,
            system_prompt="你是一个有用的 ROS 2 机器人助手，可以控制 Turtlesim 机器人, 可以聊天回答问题"
        )
    
