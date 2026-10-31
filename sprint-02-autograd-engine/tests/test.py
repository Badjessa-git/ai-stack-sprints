import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
from src.engine import Value
from src.nn import MLP

random.seed(0)
mlp = MLP(2, [4, 1])
data = [
    ([2.0, 3.0], 1.0),
    ([3.0, 1.0], 1.0),
    ([1.0, 2.0], 1.0),
    ([-2.0, -3.0], -1.0),
    ([-3.0, -1.0], -1.0),
    ([-1.0, -2.0], -1.0),
]
loss = sum([(mlp(x) - y) ** 2 for x, y in data], Value(0.0)) * (1.0 / len(data))
print("loss", loss.data)
mlp.zero_grad()
loss.backward()
params = mlp.parameters()
print("params", len(params), "unique", len({id(p) for p in params}))
print("max grad", max(abs(p.grad) for p in params))
print("max |param|", max(abs(p.data) for p in params))
