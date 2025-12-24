import os
from robota.model.doubao import DOUBAO_SEED_1_6_251015 as model

print('*'*100)
for chunk in model.stream("介绍一下你自己", extra_body={
        "thinking": {
            "type": "disabled"  # 不使用深度思考能力
            # "type": "enabled" # 使用深度思考能力
            # "type": "auto" # 模型自行判断是否使用深度思考能力
        }
    }):
    print(chunk.text, end="|", flush=True)
print('*'*100)

# print('='*100)
# for chunk in model.stream("介绍一下你自己"):
#     for block in chunk.content_blocks:
#         if block["type"] == "reasoning" and (reasoning := block.get("reasoning")):
#             print(f"Reasoning: {reasoning}")
#         elif block["type"] == "tool_call_chunk":
#             print(f"Tool call chunk: {block}")
#         elif block["type"] == "text":
#             print(block["text"])
#         else:
#             pass
# print('='*100)