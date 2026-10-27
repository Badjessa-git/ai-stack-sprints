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

from collections import defaultdict
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
            out = Value(self.data + other.data, (self, other), "add")
            out._backward = lambda: [
                setattr(self, "grad", out.grad * (self.grad + 1)),
                setattr(other, "grad", out.grad * (other.grad + 1)),
            ]
            return out
        elif isinstance(other, int) or isinstance(other, float):
            val = Value(other)
            out = Value(self.data + other, (self, val), "add")
            out._backward = lambda: [
                setattr(self, "grad", out.grad * (self.grad + 1)),
                setattr(val, "grad", out.grad * (val.grad + 1)),
            ]
            return out
        else:
            raise RuntimeError("Incompatible types as input")

    def __radd__(self, other):
        if isinstance(other, Value):
            out = Value(self.data + other.data, (self, other), "radd")
            out._backward = lambda: [
                setattr(self, "grad", out.grad * (self.grad + 1)),
                setattr(other, "grad", out.grad * (other.grad + 1)),
            ]
            return out
        elif isinstance(other, int) or isinstance(other, float):
            val = Value(other)
            out = Value(self.data + other, (self, val), "radd")
            out._backward = lambda: [
                setattr(self, "grad", out.grad * (self.grad + 1)),
                setattr(val, "grad", out.grad * (val.grad + 1)),
            ]
            return out
        else:
            raise RuntimeError("Incompatible types as input")

    def __mul__(self, other):
        if isinstance(other, Value):
            out = Value(self.data * other.data, (self, other), "mul")
            out._backward = lambda: [
                setattr(self, "grad", out.grad * other.data + self.grad),
                setattr(other, "grad", out.grad * self.data + other.grad),
            ]
            return out
        elif isinstance(other, int) or isinstance(other, float):
            val = Value(other)
            out = Value(self.data * other, (self, val), "mul")
            out._backward = lambda: [
                setattr(self, "grad", out.grad * val.data + self.grad),
                setattr(val, "grad", out.grad * self.data + val.grad),
            ]
            return out
        else:
            raise RuntimeError("Incompatible types as input")

    def __rmul__(self, other):
        if isinstance(other, Value):
            out = Value(self.data * other.data, (self, other), "rmul")
            out._backward = lambda: [
                setattr(self, "grad", out.grad * other.data + self.grad),
                setattr(other, "grad", out.grad * self.data + other.grad),
            ]
            return out
        elif isinstance(other, int) or isinstance(other, float):
            val = Value(other)
            out = Value(self.data * other, (self, val), "rmul")
            out._backward = lambda: [
                setattr(self, "grad", out.grad * val.data + self.grad),
                setattr(val, "grad", out.grad * self.data + val.grad),
            ]
            return out
        else:
            raise RuntimeError("Incompatible types as input")

    def __pow__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            out = Value(self.data**other, (self, Value(other)), "pow")
            out._backward = lambda: [
                setattr(
                    self,
                    "grad",
                    self.grad + (other * self.data ** (other - 1) * out.grad),
                )
            ]
            return out
        else:
            raise RuntimeError("Incompatible types as input")

    def __neg__(self):
        out = Value(self.data * -1, [self], "neg")
        out._backward = lambda: [setattr(self, "grad", self.grad + out.grad * -1)]
        return out

    def __sub__(self, other):
        if isinstance(other, Value):
            out = Value(self.data - other.data, (self, other), "sub")
            out._backward = lambda: [
                setattr(self, "grad", self.grad + out.grad),
                setattr(other, "grad", other.grad + (-out.grad)),
            ]
            return out
        elif isinstance(other, int) or isinstance(other, float):
            val = Value(other)
            out = Value(self.data - other, (self, val), "sub")
            out._backward = lambda: [
                setattr(self, "grad", self.grad + out.grad),
                setattr(val, "grad", val.grad + (-out.grad)),
            ]
            return out
        else:
            raise RuntimeError("Incompatible types as input")

    def __rsub__(self, other):
        if isinstance(other, Value):
            out = Value(other.data - self.data, (self, other), "rsub")
            out._backward = lambda: [
                setattr(other, "grad", other.grad + out.grad),
                setattr(self, "grad", self.grad + (-out.grad)),
            ]
            return out
        elif isinstance(other, int) or isinstance(other, float):
            val = Value(other)
            out = Value(other - self.data, (self, val), "rsub")
            out._backward = lambda: [
                setattr(val, "grad", val.grad + out.grad),
                setattr(self, "grad", self.grad + (-out.grad)),
            ]
            return out
        else:
            raise RuntimeError("Incompatible types as input")

    def __truediv__(self, other):
        if isinstance(other, Value):
            out = Value(self.data / other.data, (self, other), "div")
            out._backward = lambda: [
                setattr(self, "grad", self.grad + (out.grad * (1 / other.data))),
                setattr(
                    other,
                    "grad",
                    other.grad + (out.grad * (-self.data / other.data**2)),
                ),
            ]
            return out
        elif isinstance(other, int) or isinstance(other, float):
            val = Value(other)
            out = Value(self.data / other, (self, val), "div")
            out._backward = lambda: [
                setattr(self, "grad", self.grad + (out.grad * (1 / val.data))),
                setattr(
                    val, "grad", val.grad + (out.grad * (-self.data / val.data**2))
                ),
            ]
            return out
        else:
            raise RuntimeError("Incompatible types as input")

    # ---- unary ops ----

    def relu(self):
        if self.data <= 0:
            out = Value(0.0, [self], "rlu")
            out._backward = lambda: [setattr(self, "grad", self.grad + 0)]
            return out

        out = Value(self.data, [self], "rlu")
        out._backward = lambda: [setattr(self, "grad", self.grad + out.grad)]
        return out

    def tanh(self):
        y = math.tanh(self.data)
        out = Value(y, [self], "tanh")
        out._backward = lambda: [
            setattr(self, "grad", self.grad + (1 - y**2) * out.grad)
        ]
        return out

    # ---- the sweep ----

    def backward(self):
        """Reverse-mode differentiation over the graph ending at this node.

        Zero every grad in the graph, set this node's grad to 1.0, then walk
        nodes in reverse topological order calling each _backward closure.
        Implement the topological sort iteratively (explicit stack) — the
        benchmark builds graphs deeper than Python's recursion limit.
        """

        # self.grad = 1.0
        # self.backward()
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
            topo.append(v)

        build_topo(self)
        
        for x in reversed(topo):
            

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
