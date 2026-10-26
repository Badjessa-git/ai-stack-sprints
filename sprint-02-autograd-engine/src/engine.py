"""Scalar autograd engine — SCAFFOLD.

You implement everything marked NotImplementedError. One decision is already
made for you: the *shape* of a graph node (see __init__). A Value wraps one
scalar float, remembers which Values produced it (_prev), which op made it
(_op), and carries a local _backward closure. That is the whole
representation. The differentiation — the local derivative inside each
closure, and the topological walk in backward() — is yours to write.

Spec (tests rely on every line of this):
- Binary ops accept `other` as a Value, int, or float. Coerce numbers.
- `**` accepts int or float exponents.
- backward(): zero every grad in the graph first, set this node's grad to
  1.0, then visit nodes in reverse topological order, calling _backward.
- Accumulation: a Value feeding N ops collects grad from all N uses (+=).
"""

import math


class Value:
    """One scalar in a computational graph.

    `data` is the number. `grad` is dL/d(self) for whatever scalar loss L
    backward() was last called on — it starts at 0.0 and is filled in by
    backward(). `_prev` / `_op` / `_backward` are the graph structure; see
    the README discussion ("The standard feasible representation").
    """

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        # graph structure: which Values produced me, via which op,
        # and how to push my grad back to them (you fill this per-op)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.label = label

    # ---- binary ops: forward value + children + local backward rule ----

    def __add__(self, other):
        if isinstance(other, Value):
            return Value(self.data + other.data, (self, other), "+")
        elif isinstance(other, int) or isinstance(other, float):
            return Value(self.data + other, (self, Value(other)), "+")
        else:
            raise RuntimeError("Incompatible types as input")

    def __radd__(self, other):
        if isinstance(other, Value):
            return Value(self.data + other.data, (self, other), "+")
        elif isinstance(other, int) or isinstance(other, float):
            return Value(self.data + other, (self, Value(other)), "+")
        else:
            raise RuntimeError("Incompatible types as input")

    def __mul__(self, other):
        if isinstance(other, Value):
            return Value(self.data * other.data, (self, other), "*")
        elif isinstance(other, int) or isinstance(other, float):
            return Value(self.data * other, (self, Value(other)), "*")
        else:
            raise RuntimeError("Incompatible types as input")

    def __rmul__(self, other):
        if isinstance(other, Value):
            return Value(self.data * other.data, (self, other), "*")
        elif isinstance(other, int) or isinstance(other, float):
            return Value(self.data * other, (self, Value(other)), "*")
        else:
            raise RuntimeError("Incompatible types as input")

    def __pow__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Value(self.data**other, (self, Value(other)), "**")
        else:
            raise RuntimeError("Incompatible types as input")

    def __neg__(self):
        return Value(self.data * -1)

    def __sub__(self, other):
        if isinstance(other, Value):
            return Value(self.data - other.data, (self, other), "-")
        elif isinstance(other, int) or isinstance(other, float):
            return Value(self.data - other, (self, Value(other)), "-")
        else:
            raise RuntimeError("Incompatible types as input")

    def __rsub__(self, other):
        if isinstance(other, Value):
            return Value(other.data - self.data, (self, other), "-")
        elif isinstance(other, int) or isinstance(other, float):
            return Value(other - self.data, (self, Value(other)), "-")
        else:
            raise RuntimeError("Incompatible types as input")

    def __truediv__(self, other):
        if isinstance(other, Value):
            return Value(self.data / other.data, (self, other), "/")
        elif isinstance(other, int) or isinstance(other, float):
            return Value(self.data / other, (self, Value(other)), "/")
        else:
            raise RuntimeError("Incompatible types as input")

    # ---- unary ops ----

    def relu(self):
        if self.data <= 0:
            return Value(0.0)
        return self

    def tanh(self):
        return Value(math.tanh(self.data))

    # ---- the sweep ----

    def backward(self):
        """Reverse-mode differentiation over the graph ending at this node.

        Zero every grad in the graph, set this node's grad to 1.0, then walk
        nodes in reverse topological order calling each _backward closure.
        Implement the topological sort iteratively (explicit stack) — the
        benchmark builds graphs deeper than Python's recursion limit.
        """
        raise NotImplementedError("your turn: topo sort + reverse walk")

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
