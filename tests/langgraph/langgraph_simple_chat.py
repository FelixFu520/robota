import os
import operator
from typing_extensions import TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, AIMessage


# 定义State
class InputState(TypedDict):
    question: str
    llm_answer: Optional[str]
class OutputState(TypedDict):
    answer: str
class OverallState(InputState, OutputState):
    pass

# 定义节点
def llm_node(state: InputState):
    messages = [
        ('system', '你是一位乐于助人的智能小助理.'),
        ('human', state['question'])
    ]

    llm = init_chat_model(
        model="doubao-seed-1-6-251015",
        model_provider="openai",
        api_key=os.environ.get("ARK_API_KEY"),
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        temperature=0
    )

    response = llm.invoke(messages)
    # print(response)   
    # print(type(response))

    return {
        "llm_answer": response.content
    }
def action_node(state: InputState):
    messages = [
        ("system","无论你接收到什么语言的文本，请翻译成法语",),
        ("human", state["llm_answer"])
    ]
    
    llm = init_chat_model(
        model="doubao-seed-1-6-251015",
        model_provider="openai",
        api_key=os.environ.get("ARK_API_KEY"),
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        temperature=0
    )

    response = llm.invoke(messages) 

    return {"answer": response.content}
# 定义图
graph = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
# 添加节点
graph.add_node("llm_node", llm_node)
graph.add_node("action_node", action_node)
# 添加边
graph.add_edge(START, "llm_node")
graph.add_edge("llm_node", "action_node")
graph.add_edge("action_node", END)

# 编译图
compiled_graph = graph.compile()

# 保存图
png_data = compiled_graph.get_graph(xray=True).draw_mermaid_png()
with open("simple_chat.png", "wb") as f:
    f.write(png_data)

# 运行图
answer = compiled_graph.invoke({"question": "你好，请你详细的介绍一下你自己"})
print(answer)