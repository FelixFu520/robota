import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# Initialize model with default extra_body (applies to all calls)
model = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    model_kwargs={  # Add default parameters here
        "extra_body": {
            "thinking": {"type": "disabled"}
        }
    }
)

# Create agent with the model (inherits default extra_body)
agent = create_agent(model)

print("Agent")
print('*'*100)
# Remove extra_body from agent.stream (now uses model's default)
import time
time_start = time.time()
Time_flag = True
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "介绍一下你自己"}]},  
    stream_mode="messages"
):
    time_end = time.time()
    if Time_flag:
        print(f"Time: {time_end - time_start:.2f} seconds", end="", flush=True)
        Time_flag = False
    print(token.text, end="|", flush=True)
print('*'*100)

print("Model")
print('*'*100)
# Keep extra_body here (optional, since model already has default)
time_start = time.time()
Time_flag = True
for chunk in model.stream("介绍一下你自己", extra_body={
        "thinking": {"type": "disabled"}
    }):
    time_end = time.time()
    if Time_flag:
        print(f"Time: {time_end - time_start:.2f} seconds", end="", flush=True)
        Time_flag = False
    print(chunk.text, end="|", flush=True)
print('*'*100)