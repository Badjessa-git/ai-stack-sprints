# Sprint 3.5: Tensor Autograd (optional)

## Why this sprint

Sprint 2's `Value` does backprop one number at a time. Sprint 3 trained
its real models on `src/numpy_v2/`, a NumPy engine that does the same
thing one *array* at a time. That engine was scaffolded, so you never
wrote it. This sprint closes the gap. You rebuild it from Sprint 2's
idea, and afterwards every line under the models in this series is yours.

It's the same algorithm. Each op computes a forward result, remembers its
inputs, and leaves a closure that pushes gradient back into them.
`backward()` walks the graph in reverse topological order. Two things are
genuinely new:

- **Broadcasting.** In `x + b`, with x shaped (32, 64) and b shaped
  (64,), NumPy reuses b for every row. Something reused 32 times gets 32
  gradient contributions, so b's gradient is the *sum* over the rows.
  `unbroadcast` does that summing, and nearly every op calls it. If you
  get it wrong, shapes still line up and the gradients are silently
  wrong.
- **Matrix multiply.** For C = A @ B, the gradient of A is dC @ Bᵀ and
  the gradient of B is Aᵀ @ dC. Work it out once with 2×2 matrices on
  paper. With a batch dimension, "transpose" means swapping the last two
  axes.

Learning goals:

1. Explain in plain words why a broadcast input's gradient is a sum, and
   which axes get summed.
2. Derive the matmul gradients and the softmax gradient on paper, then
   implement them.
3. Know why gradient checking proves a backward pass correct whatever
   the code looks like inside.
4. Compare your engine's speed with Sprint 3's on the same shapes.

## The plan

Everything is in `src/tensor.py`. `__add__` is done as the worked
example, so read it first. A step is done when its test class passes.

| Step | What | Tests |
|------|------|-------|
| 1 | `unbroadcast` | `TestUnbroadcast`, then `test_add_broadcast` |
| 2 | `__mul__`, `__pow__` backward (`-`, `/` are built from them) | `TestElementwise` |
| 3 | `exp`, `log`, `tanh`, `relu` backward | `TestElementwise` |
| 4 | `sum` backward (axis None / int / tuple, keepdims) | `TestReductions` |
| 5 | `reshape`, `transpose` backward | `TestShape` |
| 6 | `__matmul__` backward, batched and broadcast | `TestMatmul` |
| 7 | `softmax` (one op), `cross_entropy` (one op) | `TestSoftmaxCrossEntropy` |
| 8 | `backward()` with a visited set; must survive a 2,000-op chain | `TestBackward` |
| 9 | Acceptance: an attention head and an MLP, end to end | `TestAcceptance` |

Step 8 is listed late but every test needs it. Port it from Sprint 2
first, then work through steps 1–7. `test_deep_chain_runs` is the one
new requirement.

## Run it

```bash
make test        # 33 tests; the stubs fail all but a few
make benchmark   # forward+backward ms at growing sizes
```

## How the tests work: gradient checking

For every input number, the test nudges it by ±1e-6, reruns the forward
pass, and measures how much the output moved. That's the gradient,
measured instead of derived. Your backward pass has to match it to about
1e-5. It's slow (two forward passes per input number), so the tests use
tiny shapes, but it is the strongest test you can write for autograd.
When one fails, the error message names the input that disagrees.

## Results

Reference timings on the dev Mac, NumPy 2.0, 2 threads. Yours should be
in the same range:

| Op | Shape | fwd+bwd |
|----|-------|---------|
| attention | B16 T64 d64 | ~3 ms |
| attention | B16 T128 d512 | ~220 ms |
| MLP + cross-entropy | N1024 d512 C4096 | ~600 ms |

## Before writing the post

- **PyTorch is this, plus kernels.** The tape-and-closures design is the
  same one PyTorch uses. The difference is that each op is a fused
  C++/CUDA kernel and memory is managed for you.
- **Fused ops.** softmax and cross-entropy are single ops here for the
  same reason production fuses them: built from `exp`/`sum`/`div`, they
  store three intermediate arrays and lose precision.
- **What's missing:** in-place ops, views, dtype control (everything is
  float64 here), GPU, and higher-order gradients.

## What AI wrote

- **Scaffold:** this README, the tests (including the gradient checker),
  `benchmarks/bench.py`, the Makefile, `POST.md`, and in `src/tensor.py`
  the class skeleton, the forward passes, `__add__` as the worked
  example, and the `-`, `/`, `mean`, `swap_last` helpers built from the
  other ops.
- **Hand-written:** every backward closure, `unbroadcast`, `softmax`,
  `cross_entropy`, and `backward()`.

## Post status

Not started. Outline in `POST.md`.
