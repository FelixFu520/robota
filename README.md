# RobotA
Robot Agent Built with ROS 2 + MCP + LangChain 1.0

## 演示视频
<div align="center">

[![B站视频演示](http://i1.hdslb.com/bfs/archive/ffa54d3aff63f6b91ab7b4450bfd51a108c4034a.jpg)](https://www.bilibili.com/video/BV16rv8BzEgk/)

**点击图片观看演示视频** | [直接访问](https://www.bilibili.com/video/BV16rv8BzEgk/)

</div>

## 前提
- System: Ubuntu 22.04-desktop
- 环境管理: [uv](https://docs.astral.sh/uv/getting-started/installation/), 版本大于0.9.13
- API KEY: 整个项目需要火山的API KEY，包括ASR，TTS，豆包，如果不清楚如何获取KEY，问火山客服
    ```
    export ARK_API_KEY=替换成自己的
    export ASR_APP_KEY=替换成自己的
    export ASR_ACCESS_KEY=替换成自己的
    export TTS_APP_KEY=替换成自己的
    export TTS_ACCESS_KEY=替换成自己的
    ```
## 安装
### 安装ROS 2
参考[官方教程](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
```
# Set locale
locale  # check for UTF-8

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

locale  # verify settings


# Setup Sources
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb


# Install ROS 2 packages
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install ros-dev-tools


# Environment setup
source /opt/ros/humble/setup.bash

```
### 安装Turtlesim, 并演示
```
# Install turtlesim
sudo apt update
sudo apt install ros-humble-turtlesim


# Start turtlesim
ros2 run turtlesim turtlesim_node


# Use turtlesim, open a new terminal and source ROS 2 again
ros2 run turtlesim turtle_teleop_key
```
### 安装ros-mcp-server
```
mkdir -p ~/projects/
cd ~/projects/
git clone https://github.com/robotmcp/ros-mcp-server
cd ~/projects/ros-mcp-server
uv sync
```
### 安装本项目
```
sudo apt-get install portaudio19-dev    # 安装portaudio19-dev

cd ~/projects/
git clone https://github.com/FelixFu520/robota.git
cd ~/projects/robota
uv sync
```

## 运行 Turtlesim Demo
### 终端1： 启动 Turtlesim
```
source /opt/ros/humble/setup.bash
ros2 run turtlesim turtlesim_node
```
### 终端2： 启动 rosbridge
```
source /opt/ros/humble/setup.bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```
### 终端3： 启动ros-mcp-server
```
cd ~/projects/ros-mcp-server
uv run server.py --transport streamable-http --host 127.0.0.1 --port 9000
```
### 终端4： 启动测试程序
```
cd ~/projects/robota
uv run tests/agent/04_test_turtlesim_mcp.py
```