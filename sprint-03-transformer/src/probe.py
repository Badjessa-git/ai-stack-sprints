import sys, time

sys.path.insert(0, ".")
from engine import Value


def ladder(depth):
    # Two nodes per level; each depends on BOTH nodes below.
    # Unique nodes: ~2*depth. Paths from top to bottom: 2**depth.
    l1, l2 = Value(1.0), Value(2.0)
    for _ in range(depth):
        n1 = l1 + l2
        n2 = l1 + l2
        l1, l2 = n1, n2
    return l1 + l2


for depth in [8, 12, 16, 20]:
    loss = ladder(depth)
    t0 = time.time()
    loss.backward()
    print(f"depth {depth}: backward {time.time()-t0:.3f}s")
