import os
from robota.model.doubao import DOUBAO_SEED_1_6_251015 as model

response = model.invoke("介绍一下你自己")
print(response)