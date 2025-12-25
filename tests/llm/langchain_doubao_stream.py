import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# Initialize model without extra_body in model_kwargs (to avoid warning)
base_model = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# Bind extra_body to the model (this ensures it's properly passed to all calls)
default_extra_body = {
    "thinking": {"type": "disabled"}
}
model = base_model.bind(extra_body=default_extra_body)

# Create agent with the bound model (inherits extra_body)
agent = create_agent(model)

print("Agent")
print('*'*100)
import time
time_start = time.time()
Time_flag = True
# Explicitly pass extra_body to ensure it's used (even though model is bound)
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "介绍一下你自己"}]},  
    stream_mode="messages",
    # extra_body=default_extra_body
):
    print("|", end="|", flush=True)
    time_end = time.time()
    if Time_flag:
        print(f"Time: {time_end - time_start:.2f} seconds", end="", flush=True)
        Time_flag = False
    print(token.text, end="|", flush=True)
print('*'*100)

# print("Model")
# print('*'*100)
# # Model already has extra_body bound, no need to pass it again
# time_start = time.time()
# Time_flag = True
# for chunk in model.stream("介绍一下你自己"):
#     time_end = time.time()
#     if Time_flag:
#         print(f"Time: {time_end - time_start:.2f} seconds", end="", flush=True)
#         Time_flag = False
#     print(chunk.text, end="|", flush=True)
# print('*'*100)