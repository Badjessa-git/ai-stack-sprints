"""A NumPy Tensor with reverse-mode autograd: Sprint 2's Value, one array
at a time instead of one number at a time.

Sprint 2's Value held one float and one grad. A Tensor holds an array and
an array of grads, the same shape. Every op does the same three things a
Value op did:

1. compute the forward result with NumPy,
2. remember its inputs (`_prev`),
3. set `out._backward`, a closure that adds this op's share of the
   gradient into each input's `.grad`.

`backward()` is unchanged from Sprint 2: topological order, then call
every `_backward` in reverse.

The two things that are new, and the reason this sprint exists:

- **Broadcasting.** `x + b` with x (32, 64) and b (64,) is legal NumPy:
  b is reused for every row. So in the backward pass, b's gradient is the
  sum over all 32 rows it was reused for. `unbroadcast` does that, and
  almost every op needs it.
- **Matrix multiply.** For C = A @ B: dA = dC @ Bᵀ and dB = Aᵀ @ dC.
  With batches, "transpose" means swapping the last two axes.

Order of work (each has a test class in tests/test_tensor.py):
    1. unbroadcast                      TestUnbroadcast
    2. __mul__, __pow__, __neg__        TestElementwise (add is done: read it first)
    3. exp, log, tanh, relu             TestElementwise
    4. sum, mean                        TestReductions
    5. reshape, transpose               TestShape
    6. __matmul__                       TestMatmul
    7. softmax, cross_entropy           TestSoftmaxCrossEntropy
    8. backward                         TestBackward
Then TestAcceptance: attention and a 2-layer MLP, gradient-checked.
"""

import numpy as np


def unbroadcast(grad, shape):
    """Sum `grad` down to `shape`, undoing NumPy broadcasting.

    When NumPy broadcasts an input of `shape` up to `grad.shape`, it does
    two things: it adds missing leading axes, and it stretches size-1 axes.
    Undo both, in that order:

    - while grad has more axes than shape: sum over axis 0
    - then for every axis where shape has 1 but grad doesn't: sum that
      axis with keepdims=True

        unbroadcast(np.ones((2, 3)), (3,))     -> [2., 2., 2.]
        unbroadcast(np.ones((2, 3)), (2, 1))   -> [[3.], [3.]]
        unbroadcast(np.ones((2, 3)), ())       -> 6.0
    """
    raise NotImplementedError("unbroadcast (TestUnbroadcast)")


def _t(x):
    return x if isinstance(x, Tensor) else Tensor(x)


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op!r})"

    # ---- elementwise --------------------------------------------------

    def __add__(self, other):
        # The worked example. Every binary op follows this shape.
        other = _t(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a + b)/da = 1, so a's grad is out.grad, summed over any
            # axes a was broadcast along. Same for b.
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        """d(a*b)/da = b, d(a*b)/db = a. Don't forget unbroadcast."""
        other = _t(other)
        out = Tensor(self.data * other.data, (self, other), "*")
        raise NotImplementedError("__mul__ backward (TestElementwise)")
        return out

    def __pow__(self, k):
        """x ** k for a plain-number exponent k. d/dx = k * x**(k-1)."""
        assert isinstance(k, (int, float)), "only plain-number exponents"
        out = Tensor(self.data**k, (self,), f"**{k}")
        raise NotImplementedError("__pow__ backward (TestElementwise)")
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-_t(other))

    def __truediv__(self, other):
        return self * _t(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return _t(other) - self

    def __rtruediv__(self, other):
        return _t(other) / self

    def exp(self):
        """d/dx e^x = e^x (reuse out.data)."""
        out = Tensor(np.exp(self.data), (self,), "exp")
        raise NotImplementedError("exp backward (TestElementwise)")
        return out

    def log(self):
        """d/dx ln x = 1/x."""
        out = Tensor(np.log(self.data), (self,), "log")
        raise NotImplementedError("log backward (TestElementwise)")
        return out

    def tanh(self):
        """d/dx tanh x = 1 - tanh(x)^2."""
        out = Tensor(np.tanh(self.data), (self,), "tanh")
        raise NotImplementedError("tanh backward (TestElementwise)")
        return out

    def relu(self):
        """Gradient 1 where x > 0, else 0."""
        out = Tensor(np.maximum(self.data, 0), (self,), "relu")
        raise NotImplementedError("relu backward (TestElementwise)")
        return out

    # ---- reductions ---------------------------------------------------

    def sum(self, axis=None, keepdims=False):
        """Every input element contributed once to its sum, so its grad is
        the out.grad of the sum it went into. If keepdims was False, put the
        summed axes back first (np.expand_dims), then np.broadcast_to the
        input shape. axis may be None, an int, or a tuple of ints."""
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")
        raise NotImplementedError("sum backward (TestReductions)")
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod(
            [self.shape[a] for a in ((axis,) if isinstance(axis, int) else axis)]
        )
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- shape --------------------------------------------------------

    def reshape(self, *shape):
        """The gradient has the input's shape: reshape it back."""
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")
        raise NotImplementedError("reshape backward (TestShape)")
        return out

    def transpose(self, *axes):
        """Permute axes (like np.transpose). The backward permutes back:
        np.argsort(axes) is the inverse permutation."""
        axes = axes or tuple(reversed(range(self.data.ndim)))
        out = Tensor(self.data.transpose(axes), (self,), "transpose")
        raise NotImplementedError("transpose backward (TestShape)")
        return out

    def swap_last(self):
        """Swap the last two axes: the batched matrix transpose."""
        axes = list(range(self.data.ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        return self.transpose(*axes)

    # ---- matmul -------------------------------------------------------

    def __matmul__(self, other):
        """Batched matrix multiply: (..., n, k) @ (..., k, m) -> (..., n, m).
        Both inputs need at least 2 axes. Batch axes broadcast like any
        other NumPy op.

        For C = A @ B:  dA = dC @ Bᵀ,  dB = Aᵀ @ dC  (ᵀ = swap the last two
        axes, np.swapaxes(x, -1, -2)). Then unbroadcast, because a 2-D B
        used against a batch of A's got its gradient from every batch."""
        other = _t(other)
        assert self.data.ndim >= 2 and other.data.ndim >= 2, "matmul needs >= 2-D"
        out = Tensor(self.data @ other.data, (self, other), "@")
        raise NotImplementedError("__matmul__ backward (TestMatmul)")
        return out

    # ---- softmax ------------------------------------------------------

    def softmax(self, axis=-1):
        """Softmax along `axis`, as ONE op with a hand-derived backward
        (not built from exp/sum/div: that works but stores 3 graphs).

        Forward: subtract the max first (stability), exp, divide by sum.
        Backward, with s = out.data and g = out.grad:
            dx = s * (g - sum(g * s, axis, keepdims=True))
        Derive it once on paper for a 2-element softmax before coding it."""
        raise NotImplementedError("softmax forward + backward (TestSoftmaxCrossEntropy)")

    # ---- autograd -----------------------------------------------------

    def backward(self):
        """Sprint 2's backward, unchanged in spirit: build a topological
        order with a visited set, set this tensor's grad to ones, then call
        _backward on every node in reverse order."""
        raise NotImplementedError("backward (TestBackward)")


def cross_entropy(logits, targets):
    """Mean cross-entropy of `logits` (N, C) against integer `targets` (N,),
    as ONE op.

    Forward: log-softmax (x - max - log(sum(exp(x - max)))), take
    -logp[i, targets[i]] for each row, average.
    Backward: (softmax(logits) - onehot(targets)) / N, times out.grad.
    This is the gradient every LM training step starts from."""
    targets = np.asarray(targets)
    raise NotImplementedError("cross_entropy (TestSoftmaxCrossEntropy)")
