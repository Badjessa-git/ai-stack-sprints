# Sprint 3.5 post outline: tensor autograd

Working title: **"The engine under my transformer, rewritten by hand"**

Plain first, term second, proof third. Lead with an artifact.

## 1. The honest gap

- Sprint 3's models trained on an engine I didn't write. Say so plainly,
  and say why it mattered to you.
- The artifact: the same attention head, gradient-checked, running on
  the engine you just wrote.

## 2. One idea, now with arrays

- The plain version: every op remembers where its inputs came from and
  how to send blame back to them. That's Sprint 2, unchanged.
- What changes with arrays: one paragraph, with the `x + b` example.

## 3. The bug class broadcasting creates

- A broadcast input reused N times gets N gradient contributions.
- Show the wrong version: forget to sum and the shapes are still fine,
  and the gradients are silently wrong. Show the gradient check catching it.
- If you hit a real bug here, tell it with real numbers.

## 4. Matmul and softmax on paper

- The 2×2 derivation of dA = dC @ Bᵀ. A photo of the paper is fine.
- The softmax backward, and why it's one fused op.

## 5. Numbers

- The benchmark table, next to Sprint 3's engine on the same shapes.

## 6. What PyTorch does differently

- Only what the README's checklist covers: kernels, fusion, views,
  dtypes, GPU.
