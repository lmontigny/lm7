import torch

import lm7

model = torch.nn.Sequential(
    torch.nn.Linear(16, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 4),
).eval()

compiled_model = lm7.compile(model, target="auto")
output = compiled_model(torch.randn(2, 16))
print(output.shape)
print(lm7.explain(model, target="auto"))
