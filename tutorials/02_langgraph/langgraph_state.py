
def state_dict():
    """使用dict作为状态"""
    from langgraph.graph import StateGraph, START, END
    from langchain.chat_models import init_chat_model
    from langchain.messages import AnyMessage, AIMessage

    builder = StateGraph(dict)
    print(builder.state_schema)

    def addition(state):
        print(state)
        return {"x": state["x"] + 1}

    def subtraction(state):
        print(state)
        return {"y": state["x"] - 2}

    builder.add_node("addition", addition)
    builder.add_node("subtraction", subtraction)
    builder.add_edge(START, "addition")
    builder.add_edge("addition", "subtraction")
    builder.add_edge("subtraction", END)
    print(builder.edges)
    print(builder.nodes)
    print(builder.schemas)

    compiled_graph = builder.compile()
    png_data = compiled_graph.get_graph(xray=True).draw_mermaid_png()
    with open("state_dict.png", "wb") as f:
        f.write(png_data)

    print(compiled_graph.invoke({"x":10}))

def state_typeddict():
    from typing_extensions import TypedDict
    from langgraph.graph import START, StateGraph, END

    def addition(state):
        print(state)
        return {"x": state["x"] + 1}

    def subtraction(state):
        print(state)
        return {"y": state["x"] - 2}

    class State(TypedDict):
        x: int
        y: int

    # 构建图
    builder = StateGraph(State) 

    # 向图中添加两个节点
    builder.add_node("addition", addition)
    builder.add_node("subtraction", subtraction)

    # 构建节点之间的边
    builder.add_edge(START, "addition")
    builder.add_edge("addition", "subtraction")
    builder.add_edge("subtraction", END)

    graph = builder.compile()

    png_data = graph.get_graph(xray=True).draw_mermaid_png()
    with open("state_typeddict.png", "wb") as f:
        f.write(png_data)
    
    # 定义一个初始化的状态
    initial_state = {"x":10}

    print(graph.invoke(initial_state))

def state_typeddict2():
    import operator
    from typing import Annotated, TypedDict, List
    from langgraph.graph import START, StateGraph, END

    class State(TypedDict):
        messages: Annotated[List[str], operator.add]

    def addition(state):
        print(state)
        msg = state['messages'][-1]
        response = {"x": msg["x"] + 1}
        return {"messages": [response]}

    def subtraction(state):
        print(state)
        msg = state['messages'][-1]
        response = {"x": msg["x"] - 2}
        return {"messages": [response]}

    # 构建图
    builder = StateGraph(State) 

    # 向图中添加两个节点
    builder.add_node("node1", addition)
    builder.add_node("node2", subtraction)

    # 构建节点之间的边
    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")
    builder.add_edge("node2", END)

    graph = builder.compile()


    png_data = graph.get_graph(xray=True).draw_mermaid_png()
    with open("state_typeddict2.png", "wb") as f:
        f.write(png_data)

    # 定义一个初始化的状态
    initial_state = {"messages": [{'x':10}]}

    print(graph.invoke(initial_state))

if __name__ == "__main__":
    # state_dict()
    # state_typeddict()
    state_typeddict2()