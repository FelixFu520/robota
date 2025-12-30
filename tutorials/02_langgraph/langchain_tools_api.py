# 1.导入相关库
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_community.utilities import OpenWeatherMapAPIWrapper
import os
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import requests
import json
from datetime import datetime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig


# 2.导入模型和工具
model = ChatOpenAI(
    model="doubao-seed-1-6-251015",  # Specify a model available on OpenRouter
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

web_search = TavilySearch(max_results=2)

@tool
def get_weather(loc):
    """
    查询即时天气函数
    :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
    注意，中国的城市需要用对应城市的英文名称代替，例如如果需要查询北京市天气，则loc参数需要输入'Beijing'；
    :return：OpenWeather API查询即时天气的结果，具体URL请求地址为：https://api.openweathermap.org/data/2.5/weather\
    返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    # Step 1.构建请求
    url = "https://api.openweathermap.org/data/2.5/weather"

    # Step 2.设置查询参数
    params = {
        "q": loc,               
        "appid": os.getenv("OPENWEATHER_API_KEY"),    # 输入API key
        "units": "metric",            # 使用摄氏度而不是华氏度
        "lang":"zh_cn"                # 输出语言为简体中文
    }

    # Step 3.发送GET请求
    response = requests.get(url, params=params)
    
    # Step 4.解析响应
    data = response.json()
    return json.dumps(data)

@tool
def write_file(content: str) -> str:
    """
    将指定内容写入本地文件。
    :param content: 必要参数，字符串类型，用于表示需要写入文档的具体内容。
    :return: 写入结果提示信息。
    """
    try:
        # ✅ 始终先定义文件名（防止未绑定变量）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_{timestamp}.txt"

        # 写入文件
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

        abs_path = os.path.abspath(filename)
        return f"✅ 已成功写入本地文件：{abs_path}"

    except Exception as e:
        return f"❌ 文件写入失败：{str(e)}"

checkpointer = InMemorySaver()
config: RunnableConfig = {"configurable": {"thread_id": "1"}}

# 3.创建Agent
agent = create_agent(
    model=model,
    tools=[web_search, get_weather, write_file],
    system_prompt="你是一名多才多艺的智能助手，可以调用工具帮助用户解决问题。",
    checkpointer=checkpointer
)

# 4.运行Agent获得结果
result = agent.invoke(
    {"messages": [{"role": "user", "content": "我是FelixFu, 请帮我查询天津、石家庄、上海等地天气，并写入本地文件"}]},
    config
)
for m in result["messages"]:
    m.pretty_print()
    print('*'*100)


print('='*100)


result = agent.invoke(
    {"messages": [{"role": "user", "content": "你好，请问你还记得我叫什么名字么？"}]},
    config
)
for m in result["messages"]:
    m.pretty_print()
    print('*'*100)
