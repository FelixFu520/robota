# turtlesim+mcp示例
参考[Langgraph-mcp-ROS2](https://github.com/sahars93/Langgraph-mcp-ROS2)和[rosa](https://github.com/nasa-jpl/rosa)快速让mcp和agent连通。

## 前提
需要安装ros2(humble), 然后安装本项目的环境
## 运行
```
cd tutorials/01_turtlesim-mcp
python turtlesim_mcp_server.py  # 终端1启动
python turtlesim_agent.py # 终端2启动
```
## 缺点
1. 环境耦合，因为`turtlesim_mcp_server.py`中用到了`import rclpy`，那么python环境和ros2环境要对应上，所以Python必须是3.10，否则运行报错
2. mcp和Node耦合，要为每个node配置一个mcp，扩展不方便
3. 我跑的时候速度还有点慢
4. 想办法解耦合, 参考https://github.com/punkpeye/awesome-mcp-servers中的ros2-server