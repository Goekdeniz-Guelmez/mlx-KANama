from mlx_kanama.models.kan_llama import Model, ModelArgs

import mlx.core as mx

model = Model(ModelArgs())

output = model(mx.array([[1, 345, 756]]))

print(output)