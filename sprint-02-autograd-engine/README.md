# Sprint 02 — Build a Scalar Autograd Engine From Scratch

Every other weekend · 48-hour manual coding sprint · Sprint 2 of "Rebuild the Modern AI Stack"

## Why this sprint

Every neural network ever trained runs on one algorithm: backpropagation.
It is ~100 lines of code, and after this weekend it will not be magic
anymore — it will be a concrete program you wrote, where you can point at
any line and say what it does. That understanding compounds: Sprint 3's
transformer trains with exactly this engine's ideas, and the "never
recompute what didn't change" lesson from your Sprint 1 Part 2 optimization
shows up here again (in how backward() walks the graph — and in Sunday's
optional stretch goal).

## Series context

The roadmap, with this weekend highlighted:

1. BPE tokenizer from scratch ✅ (Oct 10–11)
2. **Scalar autograd engine + tiny MLP** ← you are here (Oct 24–25)
3. Transformer block + tiny character LM — attention, LayerNorm, sampling
4. Vector search engine — exact kNN, heaps, then HNSW/IVF
5. Hybrid retrieval for RAG — BM25 + vectors, rank fusion (reuses 1 + 4)
6. KV cache + paged inference simulator — why inference is memory-bound
7. LSM-tree key-value store — memtable, SSTables, compaction, Bloom filters
8. Continuous batching scheduler — capstone simulator (ties 3 + 6 together)

This sprint feeds Sprint 3 directly: the transformer is just a bigger graph
of the same Values, trained by the same backward(). Get the scalar case
exactly right this weekend and the transformer is "more of the same, with
attention on top."

## Learning goals

By Sunday night you should be able to explain, from having built it:

- what a computational graph is, and why the forward pass builds one as a
  side effect
- the chain rule in plain words, and why reverse mode computes every
  parameter's gradient in a single backward sweep
- why backward() must visit nodes in reverse topological order
- why gradients accumulate with `+=` (and what breaks if you use `=`)
- why a 200-line scalar engine trains a real classifier on real data

## Discussion: the core concepts (read before you build)

### 1. Why scalars first

A neural net looks like it runs on tensors, but every tensor op is the same
scalar op applied thousands of times. If you can differentiate one scalar
expression by hand-tracking its graph, you understand what PyTorch does —
the tensor version is the same idea with batching. Scalars keep every number
inspectable: you can print any node's data and grad while debugging.

### 2. The computational graph

Every time you write `c = a + b`, two things happen: the arithmetic
(forward), and — invisibly — a record that "c came from a and b via +".
The full record of a forward pass is a directed acyclic graph: nodes are
Values, edges point from inputs to the op that consumed them. The graph is
the program's memory of what it computed. (This is the graph view from your
Wednesday worksheet — same picture, now executable.)

### 3. Reverse-mode differentiation (the chain rule, backwards)

Training needs one number per parameter: "if I nudge this weight a tiny
amount, how much does the loss move?" That is a derivative, dL/dw. The chain
rule says you can factor it through every intermediate step:
dL/dw = (dL/dy) × (dy/dw). Reverse mode starts at the loss — where dL/dL = 1
is trivially true — and walks the graph backwards, multiplying each op's
*local* derivative by the gradient flowing into it. One backward sweep
yields the gradient for EVERY parameter at once.

That last sentence is the whole trick, and it is why reverse mode wins for
neural nets: millions of parameters, one loss. Forward-mode differentiation
would need one sweep *per parameter*. Reverse mode needs one sweep, total.

In plain words — this is your capture-sheet prompt 3 from the Oct 18
Karpathy deep dive (one phrase for how he describes gradient flow; the sheet
is at `karpathy-deep-dive-capture-2026-10-18.md` in the goal files — if that
line is still blank, fill it in tonight and open Saturday with it): each
node, on the backward pass, is answering "how much did I contribute to the
final answer?" and handing that answer to whoever made it.

### 4. Topological ordering

You cannot push gradients backwards in arbitrary order: a node's grad must
be *final* before you propagate it to its children. So backward() first
sorts the graph topologically — every node ordered after the nodes that
produced it — then walks that order in reverse. The sort must be iterative
(explicit stack): the benchmark builds graphs deeper than Python's recursion
limit, and finding that out via RecursionError is a rite of passage you
only need once.

### 5. Gradient accumulation

If one Value feeds two ops, nudging it moves the loss through *both* paths —
so its grad is the SUM of both contributions. In code that is `+=`, never
`=`; the `test_gradient_accumulation` and `test_diamond` tests are hunting
exactly this bug. Related: backward() zeroes every grad before each sweep,
or repeated calls (like every training step) pile stale gradients on top of
fresh ones — `test_backward_twice_resets_grads` pins that behavior.

## The standard feasible representation (start here)

This is the recommended shape — it is also the entire contents of the
already-written `Value.__init__`, so you are not discovering it, you are
filling it in:

```python
class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data          # the scalar number itself
        self.grad = 0.0           # dL/d(self); filled in by backward()
        self._prev = set(_children)  # which Values produced me
        self._op = _op             # which op produced me: '+', '*', 'tanh', ...
        self._backward = lambda: None  # how I push my grad to my children
```

Each op (say `__add__`) does four things: compute the output data, wrap it
in a Value with `_children=(self, other)` and `_op='+'`, and install a
`_backward` closure that captures the *local* derivatives and does the `+=`
into the children's grads. `backward()` is then just: topo-sort, zero grads,
set output grad to 1.0, walk in reverse calling each `_backward`.

Alternative route, named up front: instead of per-op closures, you can store
`(op, children)` and dispatch through a table of derivative rules inside
backward(). Equivalent power, arguably clearer to read, more code. The
closure version is recommended — it matches the micrograd reference and is
less code to get right.

## The spec

- `src/engine.py`: `Value` with `__add__`, `__radd__`, `__mul__`, `__rmul__`,
  `__pow__`, `__neg__`, `__sub__`, `__rsub__`, `__truediv__`, `relu()`,
  `tanh()`, `backward()`. Binary ops accept Value, int, or float (`**`
  takes int/float exponents). `__init__` is already written; everything else
  raises NotImplementedError until you write it.
- `src/nn.py`: `Module` (parameters/zero_grad), `Neuron(n_in, nonlin=True)`
  with weights+bias from `random.uniform(-1, 1)`, `Layer`, `MLP(n_in, n_outs)`.
  `__call__` returns a single Value when the output size is 1, else a list.
- `tests/`: 16 engine tests + 6 nn tests. They fail on the stubs as designed.
  Get them green in file order; `test_numerical_gradient_check` is the gate
  that says your chain rule is actually right, and the last nn test trains a
  tiny MLP end to end.
- `benchmarks/bench.py`: graph-size sweep (forward vs backward time) plus an
  MLP micro-benchmark. `benchmarks/train.py`: trains MLP(2,[16,16,1]) on
  `data/moons.csv`, prints the loss curve. Both run after tests pass.

## The rules (agreed, same as Sprint 1)

1. **You write the core implementation** — everything raising
   NotImplementedError in `src/`.
2. AI may scaffold, explain concepts, review your code, and ask debugging
   questions. It does not paste the core solution.
3. Turn off / don't use autocomplete AI for `src/`. Struggling with the
   `_backward` closure *is* the workout.
4. Tests are the spec. If a test and this README disagree, the test wins —
   flag it rather than silently changing the test.

## Out of scope (don't build these)

- No numpy, no torch, no pip installs: pure Python stdlib, like Sprint 1.
- No tensors, no batching, no GPU. Scalar-only is the point this weekend.
- No peeking at micrograd's `engine.py` before your backward() passes the
  numerical gradient check — the lecture is for concepts, the code is sealed
  until your tests are green. Peeking early trades away the rep.

## Stretch goals — difficulty and alternatives, BEFORE you sink hours

Rule: v1 (all tests green + `train.py` loss falling on moons) comes first.
No stretch goal before that. Then pick at most one, only if measured.

- **Stretch A — topological-order caching (medium).** In training you call
  backward() hundreds of times on same-shaped graphs; caching the topo order
  across calls skips re-sorting. This is the "never recompute what didn't
  change" pattern from your Part 2 optimization, recurring on schedule.
  *Alternative / honest caveat:* measure first with bench.py — if backward()
  isn't the bottleneck, this buys ~nothing and you should say so in the post
  with numbers. A measured "not worth it" is a result.
- **Stretch B — higher-order gradients (hard).** Grad-of-grad: making
  backward() itself build a graph, so every `_backward` closure becomes
  differentiable. Real, but fiddly — closure-captured floats need to become
  Values. *Alternative:* leave it as a discussion-only item; during review we
  compare with PyTorch's `create_graph=True` and you write the post section
  from understanding, not from a second implementation.
- **Stretch C — vectorize the hot loop (medium-hard).** The scalar engine is
  slow; production engines batch ops over tensors. *Alternative (recommended):*
  keep scalar-only and put the speed story in the post's "what production
  does differently" section — that section is explicitly held for the review
  discussion (see POST.md), not for weekend code.

## Weekend plan

- **Fri (tonight):** read the Discussion section above; fill in capture-sheet
  prompt 3 if blank; skim the micrograd lecture's first half for concepts
  only (code sealed per Out of scope).
- **Sat:** implement the Value ops + backward(); tests green in file order;
  run `make benchmark`, record v1 numbers.
- **Sun:** run `make train` on moons; compare against Saturday; at most one
  measured stretch; draft the post from POST.md.

## Resources

- Karpathy, "The spelled-out intro to neural networks and backpropagation:
  building micrograd" (YouTube lecture) and https://github.com/karpathy/micrograd
  — concepts any time; code sealed until your numerical gradient check passes.
- 3Blue1Brown, Neural Networks Ch. 2 — gradient descent and where backprop fits.
- Christopher Olah, "Calculus on Computational Graphs: Backpropagation" —
  the graph view; pairs with your Wednesday worksheet.
- Your capture sheet (`karpathy-deep-dive-capture-2026-10-18.md`, goal files),
  prompt 3 — the one-line plain-words phrase; tonight's warm-up.
- Your prep worksheets: `sprint2-wed-graph-view-worksheet-2026-10-21.md`,
  `sprint2-thu-fri-micrograd-worksheet-2026-10-22.md`,
  `sprint2-evening-prep-2026-10-20.md` (goal files).

## Layout & commands

```
sprint-02-autograd-engine/
  README.md  POST.md  Makefile
  src/engine.py  src/nn.py     <- you implement the NotImplementedErrors
  tests/test_engine.py  tests/test_nn.py
  benchmarks/bench.py  benchmarks/train.py
  data/moons.csv
```

```bash
make test       # 22 tests, fail on stubs as designed
make benchmark  # scaling sweep, after tests pass
make train      # MLP on moons, after tests pass
```
