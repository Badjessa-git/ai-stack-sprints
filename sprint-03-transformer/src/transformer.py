"""Transformer block + tiny character LM — SCAFFOLD.

You implement everything marked NotImplementedError. Your Sprint 2 engine
(src/engine.py, a frozen copy of your merged work) and nn library
(src/nn.py) are already here and complete, except for two new ops you add
in Step 0: Value.exp() and Value.log().

How the pieces fit (read the README first — it teaches each one in plain
words before you code it):
    softmax -> CausalSelfAttention -> MultiHeadAttention
    LayerNorm -> FeedForward (GELU) -> Block -> Embedding -> Transformer

Conventions, same as Sprint 2:
- Everything is a list of engine.Value objects. A vector is a list of
  Values; a sequence is a list of vectors.
- Layers from src.nn return a single Value when n_out == 1, else a list.
- Every module subclasses Module, implements parameters(), and gets
  zero_grad() for free.
- Forward builds the graph; loss.backward() differentiates it. A forward
  test passing means the backward closure is installed, or the gradcheck
  tests will catch it.
"""

import math
import random

from src.engine import Value
from src.nn import Layer, Module


def softmax(xs):
    """Turn a list of Values into weights that sum to 1.

    xs: list of Values (the scores). Returns a list of Values the same
    length, each between 0 and 1, summing to 1. Subtract the largest
    score first so exp() never overflows — the weights come out the same.
    """
    raise NotImplementedError


def gelu(x):
    """Gaussian Error Linear Unit, tanh approximation (one Value in/out).

    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    It behaves like ReLU for large |x| but lets a little signal through
    near zero, which is why transformers use it instead of ReLU.
    """
    raise NotImplementedError


def _dot(a, b):
    """Dot product of two equal-length lists of Values -> one Value."""
    raise NotImplementedError


def _vec_add(a, b):
    """Element-wise sum of two equal-length lists of Values."""
    raise NotImplementedError


class CausalSelfAttention(Module):
    """One attention head.

    x: a list of T vectors, each a list of d_model Values (one vector per
    token position). Returns a list of T vectors of head_dim Values.
    Position t may only look at positions 0..t, never ahead — that is the
    causal mask, and it is what makes the model a language model instead
    of a fill-in-the-blank model.
    """

    def __init__(self, d_model, head_dim):
        self.d_model = d_model
        self.head_dim = head_dim
        self.Wq = Layer(d_model, head_dim, nonlin=False)
        self.Wk = Layer(d_model, head_dim, nonlin=False)
        self.Wv = Layer(d_model, head_dim, nonlin=False)

    def __call__(self, x):
        raise NotImplementedError

    def parameters(self):
        return self.Wq.parameters() + self.Wk.parameters() + self.Wv.parameters()


class MultiHeadAttention(Module):
    """n_head heads in parallel, concatenated, one output projection.

    x: list of T vectors of d_model Values. Returns the same shape.
    Splitting d_model across heads lets each head watch a different kind
    of relationship (one head may watch the previous token, another the
    start of the sentence) while the total width stays d_model.
    """

    def __init__(self, d_model, n_head):
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.heads = [
            CausalSelfAttention(d_model, self.head_dim) for _ in range(n_head)
        ]
        self.Wout = Layer(d_model, d_model, nonlin=False)

    def __call__(self, x):
        raise NotImplementedError

    def parameters(self):
        ps = []
        for h in self.heads:
            ps.extend(h.parameters())
        return ps + self.Wout.parameters()


class LayerNorm(Module):
    """Normalize one vector to mean 0 and variance 1, then scale + shift.

    x: one vector (list of Values). gamma starts at 1, beta at 0, so a
    fresh LayerNorm passes its input through almost unchanged and the
    network learns how much normalization it wants.
    """

    def __init__(self, d_model, eps=1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = [Value(1.0) for _ in range(d_model)]
        self.beta = [Value(0.0) for _ in range(d_model)]

    def __call__(self, x):
        raise NotImplementedError

    def parameters(self):
        return self.gamma + self.beta


class FeedForward(Module):
    """Linear -> GELU -> Linear. Hidden width defaults to 4x d_model.

    This is where most of a transformer's parameters live, and where it
    does its per-position thinking — attention moves information between
    positions, the feedforward digests it at each position alone.
    """

    def __init__(self, d_model, hidden=None):
        self.d_model = d_model
        self.hidden = hidden or 4 * d_model
        self.fc1 = Layer(d_model, self.hidden, nonlin=False)
        self.fc2 = Layer(self.hidden, d_model, nonlin=False)

    def __call__(self, x):
        raise NotImplementedError

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()


class Block(Module):
    """One transformer block, pre-norm.

    x -> x + attn(LN(x)), then x -> x + mlp(LN(x)). The residual (+ x)
    is what lets gradients flow through a deep stack without vanishing;
    the LayerNorm before each sublayer (pre-norm) is what keeps training
    stable. x is a list of T vectors; the return is the same shape.
    """

    def __init__(self, d_model, n_head):
        self.d_model = d_model
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head)
        self.ln2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def __call__(self, x):
        raise NotImplementedError

    def parameters(self):
        return (
            self.ln1.parameters()
            + self.attn.parameters()
            + self.ln2.parameters()
            + self.mlp.parameters()
        )


class Embedding(Module):
    """A lookup table: an integer index -> a list of d_model Values.

    The row is returned as-is (a list of the actual parameter Values),
    so a forward pass builds a graph through the embedding rows and
    training updates exactly the rows that were used.
    """

    def __init__(self, n, d_model):
        self.n = n
        self.d_model = d_model
        self.weight = [
            [Value(random.uniform(-1, 1)) for _ in range(d_model)] for _ in range(n)
        ]

    def __call__(self, idx):
        return list(self.weight[idx])

    def parameters(self):
        return [v for row in self.weight for v in row]


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
        self.lm_head = Layer(d_model, vocab_size, nonlin=False)

    def __call__(self, idx):
        """idx: list of token ids, length T. Returns T logit vectors."""
        raise NotImplementedError

    def loss(self, idx):
        """Mean cross-entropy: logits at position t predict token t+1.

        Returns one scalar Value. loss = mean over t of
        -log(softmax(logits[t])[idx[t+1]]).
        """
        raise NotImplementedError

    def generate(self, idx, n_new, temperature=1.0, greedy=False):
        """Extend idx by n_new tokens. Greedy takes the argmax; otherwise
        sample from softmax(logits / temperature). Returns the full list."""
        raise NotImplementedError

    def parameters(self):
        ps = self.tok_emb.parameters() + self.pos_emb.parameters()
        for b in self.blocks:
            ps.extend(b.parameters())
        return ps + self.ln_f.parameters() + self.lm_head.parameters()
