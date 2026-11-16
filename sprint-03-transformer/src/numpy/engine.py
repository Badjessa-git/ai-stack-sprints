"""NumPy engine: the same reverse-mode autograd as src/scalar/engine.py, but each
node holds a whole array of numbers instead of one scalar.

Why this exists
---------------
The scalar engine (Value) does one Python operation per number. A single
training step builds hundreds of thousands of Value objects, and Python
pays the cost of creating each one. This engine keeps the exact same
autograd idea — every op records its parents and a _backward closure, and
backward() walks the graph in reverse topological order — but the numbers
live in NumPy arrays, so one call does a whole layer's math in compiled
C code. That is the entire speedup: same math, same gradients, far fewer
Python operations. NumPy also spreads big matrix work across cores, so
the other cores get used without any parallel code.

How to read it next to src/scalar/engine.py
--------------------------------------
- Tensor is Value, with `data` an np.ndarray (float64) instead of a float.
- Every op has the same shape: compute the forward array, build the
  output Tensor with the parents recorded, and install a _backward
  closure that pushes the output grad back to the parents.
- backward() is the same iterative topo sort with the visited set (the
  Step 0 fix from the scalar engine), seeded with ones.
- mean() is built from sum() and division, so its backward comes free.
- The one new idea is broadcasting: a (d,) bias added to a (T, d) array
  is one add, and its backward has to add up the grad over the rows it
  was copied across. _unbroadcast does that summing.

What it does NOT do: no GPU, no batch dimension, no fancy indexing
beyond what the transformer needs. Matmul is 2D only. That keeps the
backward rules short enough to verify by gradcheck, which the tests do
for every op.
"""

import numpy as np


def _as_tensor(x):
    """Wrap a plain number or array as a leaf Tensor (no parents)."""
    if isinstance(x, Tensor):
        return x
    return Tensor(np.asarray(x, dtype=np.float64))


def _unbroadcast(grad, shape):
    """Sum `grad` down to `shape`, undoing NumPy broadcasting.

    grad is the gradient of a broadcast result; shape is the shape of
    one operand. Axes that were broadcast (the operand had size 1, or
    the operand had fewer dimensions) get summed away.
    """
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, s in enumerate(shape) if s == 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


def concat(tensors, axis=0):
    """Join Tensors along `axis` (the array version of list extend)."""
    data = np.concatenate([t.data for t in tensors], axis=axis)
    out = Tensor(data, tuple(tensors), "concat")

    def _backward():
        sizes = [t.data.shape[axis] for t in tensors]
        idx = np.cumsum([0] + sizes)
        for t, (a, b) in zip(tensors, zip(idx[:-1], idx[1:])):
            sl = [slice(None)] * out.data.ndim
            sl[axis] = slice(a, b)
            t.grad += out.grad[tuple(sl)]

    out._backward = _backward
    return out


class Tensor:
    """One array in a computational graph.

    `data` is the array. `grad` is dL/d(self) for whatever scalar loss L
    backward() was last called on — it starts at zeros and is filled in
    by backward(). `_prev` / `_op` / `_backward` are the graph structure,
    exactly as in src/scalar/engine.py.
    """

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.label = label

    @property
    def shape(self):
        return self.data.shape

    # ---- binary ops: forward array + parents + local backward rule ----

    def __add__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data + other.data, (self, other), "add")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data * other.data, (self, other), "mul")

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        out = Tensor(-self.data, (self,), "neg")

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data - other.data, (self, other), "sub")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(-out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rsub__(self, other):
        return _as_tensor(other).__sub__(self)

    def __truediv__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data / other.data, (self, other), "div")

        def _backward():
            self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            other.grad += _unbroadcast(
                -out.grad * self.data / other.data**2, other.data.shape
            )

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return _as_tensor(other).__truediv__(self)

    def __matmul__(self, other):
        """2D matrix multiply: (m, k) @ (k, n) -> (m, n)."""
        other = _as_tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "matmul")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        # other is a plain number (the transformer uses 2, 3, and 0.5)
        out = Tensor(self.data**other, (self,), "pow")

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    # ---- unary ops ----

    def tanh(self):
        y = np.tanh(self.data)
        out = Tensor(y, (self,), "tanh")

        def _backward():
            self.grad += (1 - y**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        y = np.exp(self.data)
        out = Tensor(y, (self,), "exp")

        def _backward():
            self.grad += y * out.grad

        out._backward = _backward
        return out

    def log(self):
        y = np.log(self.data)
        out = Tensor(y, (self,), "log")

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    # ---- reductions / shape ops ----

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is None:
                self.grad += np.broadcast_to(g, self.data.shape)
            else:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                if keepdims:
                    self.grad += np.broadcast_to(g, self.data.shape)
                else:
                    shape = list(self.data.shape)
                    for a in axes:
                        shape[a] = 1
                    self.grad += np.broadcast_to(g.reshape(shape), self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """mean = sum / N, composed from sum and division."""
        if axis is None:
            n = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            n = 1
            for a in axes:
                n *= self.data.shape[a]
        return self.sum(axis=axis, keepdims=keepdims) / n

    def transpose(self, *axes):
        """No args: reverse the axes (for 2D this swaps rows and columns)."""
        if not axes:
            axes = tuple(reversed(range(self.data.ndim)))
        out = Tensor(self.data.transpose(*axes), (self,), "transpose")
        inv = tuple(np.argsort(axes))

        def _backward():
            self.grad += out.grad.transpose(*inv)

        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def __getitem__(self, key):
        """Row lookup, slicing, or fancy indexing.

        Backward scatters the output grad back into the parent's shape
        with np.add.at, so repeated indices accumulate (the same row used
        twice gets both gradients — exactly like the scalar engine, where
        the same Value appears twice and its grad is added twice).
        """
        out = Tensor(self.data[key], (self,), "getitem")

        def _backward():
            g = np.zeros_like(self.data)
            np.add.at(g, key, out.grad)
            self.grad += g

        out._backward = _backward
        return out

    # ---- the sweep (mirrors Value.backward) ----

    def backward(self):
        """Reverse-mode differentiation over the graph ending at this node.

        Zero every grad in the graph, set this node's grad to ones, then
        walk nodes in reverse topological order calling each _backward
        closure. Iterative topo sort with a visited set, same as the
        scalar engine's Step 0 fix.
        """
        topo = []
        worklist = [(self, False)]
        visited = set()

        while worklist:
            node, processed = worklist.pop()
            if not processed:
                if node in visited:
                    continue
                visited.add(node)
                node.grad = np.zeros_like(node.data)
                parents = node._prev
                worklist.append((node, True))
                if parents:
                    worklist.extend((x, False) for x in parents)
            else:
                topo.append(node)

        topo.reverse()

        self.grad = np.ones_like(self.data)
        for node in topo:
            node._backward()

    def __repr__(self):
        return f"Tensor(shape={self.data.shape})"
