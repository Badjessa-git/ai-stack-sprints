"""NumPy transformer — the same model as src/scalar/transformer.py, built on the
NumPy engine (src/numpy/engine.py) instead of the scalar Value engine.

Same classes, same __init__ signatures, same method names, same weight
init (random.uniform draws in the same order, so the same seed gives the
same weights). The only difference is the representation:

- Scalar: a vector is a list of Values; a sequence is a list of vectors.
- NumPy: a vector is one Tensor of shape (d,); a whole sequence is one
  Tensor of shape (T, d).

That is the whole point of the rewrite. Every per-number Python
operation in the scalar version becomes one array operation here:
the Q/K/V projections are three matrix multiplies instead of T x 3 x
head_dim dot products; the feedforward is two matrix multiplies instead
of T x (4d + d) dot products; LayerNorm works on the whole sequence at
once. Same math, same gradients, far fewer Python operations.

Two places where the array version differs in shape from the scalar one:
- Attention is causal by masking: the full (T, T) score matrix is
  computed, then positions ahead of t get -1e9 before softmax, so they
  get weight 0. The scalar version never computes those scores at all;
  the result is identical (exp(-1e9) is 0).
- __call__ returns one Tensor of shape (T, vocab) instead of a list of
  T logit vectors. loss() and generate() use that shape directly.

Switch engines in benchmarks/train.py with --engine {scalar,numpy}.
"""

import math
import random

import numpy as np

from src.numpy.engine import Tensor, concat
from src.numpy.nn import Linear, Module


def softmax(xs, axis=-1):
    """xs: Tensor. Returns a Tensor of the same shape, summing to 1 along
    `axis`. The max is subtracted first (a plain number, not part of the
    graph) so exp() never overflows — same trick as the scalar softmax.
    """
    m = xs.data.max(axis=axis, keepdims=True)
    e = (xs - m).exp()
    s = e.sum(axis=axis, keepdims=True)
    return e / s


def gelu(x):
    """Gaussian Error Linear Unit, tanh approximation — same formula as
    the scalar gelu, on whole arrays.

    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    """
    return 0.5 * x * (1 + (math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)).tanh())


class CausalSelfAttention(Module):
    """One attention head.

    x: Tensor of shape (T, d_model). Returns Tensor (T, head_dim).
    Position t may only look at positions 0..t, never ahead.
    """

    def __init__(self, d_model, head_dim):
        self.d_model = d_model
        self.head_dim = head_dim
        self.Wq = Linear(d_model, head_dim)
        self.Wk = Linear(d_model, head_dim)
        self.Wv = Linear(d_model, head_dim)

    def __call__(self, x):
        T = x.data.shape[0]
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        scores = (q @ k.transpose()) / math.sqrt(self.head_dim)
        # causal mask: ahead-of-t positions get -1e9, so softmax gives
        # them weight 0. The diagonal and below stay 0 (unchanged).
        mask = np.triu(np.full((T, T), -1e9), k=1)
        weights = softmax(scores + mask, axis=1)
        return weights @ v

    def parameters(self):
        return self.Wq.parameters() + self.Wk.parameters() + self.Wv.parameters()


class MultiHeadAttention(Module):
    """n_head heads in parallel, concatenated, one output projection.

    x: Tensor (T, d_model). Returns Tensor (T, d_model).
    """

    def __init__(self, d_model, n_head):
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.heads = [
            CausalSelfAttention(d_model, self.head_dim) for _ in range(n_head)
        ]
        self.Wout = Linear(d_model, d_model)

    def __call__(self, x):
        outs = [head(x) for head in self.heads]
        concatted = concat(outs, axis=1)
        return self.Wout(concatted)

    def parameters(self):
        ps = []
        for h in self.heads:
            ps.extend(h.parameters())
        return ps + self.Wout.parameters()


class LayerNorm(Module):
    """Normalize each row to mean 0 and variance 1, then scale + shift.

    x: Tensor (T, d_model). gamma starts at 1, beta at 0, so a fresh
    LayerNorm passes its input through almost unchanged.
    """

    def __init__(self, d_model, eps=1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = Tensor(np.ones(d_model))
        self.beta = Tensor(np.zeros(d_model))

    def __call__(self, x):
        mean = x.mean(axis=1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
        std = (var + self.eps) ** 0.5
        out = (x - mean) / std
        return out * self.gamma + self.beta

    def parameters(self):
        return [self.gamma, self.beta]


class FeedForward(Module):
    """Linear -> GELU -> Linear. Hidden width defaults to 4x d_model."""

    def __init__(self, d_model, hidden=None):
        self.d_model = d_model
        self.hidden = hidden or 4 * d_model
        self.fc1 = Linear(d_model, self.hidden)
        self.fc2 = Linear(self.hidden, d_model)

    def __call__(self, x):
        return self.fc2(gelu(self.fc1(x)))

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()


class Block(Module):
    """One transformer block, pre-norm.

    x -> x + attn(LN(x)), then x -> x + mlp(LN(x)). x is a Tensor of
    shape (T, d_model); the return is the same shape.
    """

    def __init__(self, d_model, n_head):
        self.d_model = d_model
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head)
        self.ln2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def parameters(self):
        return (
            self.ln1.parameters()
            + self.attn.parameters()
            + self.ln2.parameters()
            + self.mlp.parameters()
        )


class Embedding(Module):
    """A lookup table: integer indices -> rows of a (n, d_model) array.

    __call__ takes a list of indices and returns a Tensor of shape
    (len(idx), d_model). The backward scatters the gradient back into
    exactly the rows that were used (np.add.at accumulates, so a token
    that appears twice gets both gradients).
    """

    def __init__(self, n, d_model):
        self.n = n
        self.d_model = d_model
        self.weight = Tensor(
            np.array(
                [[random.uniform(-1, 1) for _ in range(d_model)] for _ in range(n)]
            )
        )

    def __call__(self, idx):
        return self.weight[idx]

    def parameters(self):
        return [self.weight]


class Transformer(Module):
    """Tiny character-level GPT: embeddings + N blocks + final norm + head.

    vocab_size: number of distinct characters. d_model: width of every
    vector in the model. n_head: attention heads per block. n_layer:
    blocks stacked. block_size: longest context the model ever sees.
    """

    def __init__(self, vocab_size, d_model=16, n_head=2, n_layer=1, block_size=8):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.tok_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(block_size, d_model)
        self.blocks = [Block(d_model, n_head) for _ in range(n_layer)]
        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def __call__(self, idx):
        """idx: list of token ids, length T. Returns Tensor (T, vocab)."""
        T = len(idx)
        x = self.tok_emb(idx) + self.pos_emb(list(range(T)))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def loss(self, idx):
        """Mean cross-entropy: logits at position t predict token t+1.

        Returns one scalar Tensor. loss = mean over t of
        -log(softmax(logits[t])[idx[t+1]]).
        """
        logits = self(idx)
        probs = softmax(logits[:-1], axis=1)
        rows = np.arange(len(idx) - 1)
        targets = np.array(idx[1:], dtype=np.int64)
        target_probs = probs[(rows, targets)]
        return -target_probs.log().mean()

    def generate(self, idx, n_new, temperature=1.0, greedy=False):
        """Extend idx by n_new tokens. Greedy takes the argmax; otherwise
        sample from softmax(logits / temperature). Returns the full list."""
        for _ in range(n_new):
            ctx = idx[-self.block_size :]
            last = self(ctx)[-1]
            if greedy:
                nxt = int(np.argmax(last.data))
            else:
                probs = softmax(last / temperature)
                r = random.random()
                total = 0.0
                for i, p in enumerate(probs.data):
                    total += p
                    if r < total:
                        nxt = i
                        break
            idx = idx + [nxt]
        return idx

    def parameters(self):
        ps = self.tok_emb.parameters() + self.pos_emb.parameters()
        for b in self.blocks:
            ps.extend(b.parameters())
        return ps + self.ln_f.parameters() + self.lm_head.parameters()
