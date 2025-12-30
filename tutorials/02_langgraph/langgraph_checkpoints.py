from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
import json
class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
print(graph.invoke({"foo": ""}, config))

print(f"len:{len(list(checkpointer.list(config)))}")
for checkpoint in checkpointer.list(config):
    print('*'*100)
    # 格式化输出
    print(json.dumps(checkpoint, indent=4))


config = {"configurable": {"thread_id": "1"}}
print(graph.get_state(config))

for state_graph in list(graph.get_state_history(config)):
    print('-'*100)
    print(json.dumps(state_graph, indent=4))