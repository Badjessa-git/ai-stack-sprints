# Sprint 08 post outline: from GPT-2 to a modern model

Working title: **"Five years of LLM progress is four small swaps"**

## 1. The artifact

- "The capital of France is Paris." from SmolLM-135M on your code,
  served by your server.

## 2. The four swaps

- One section each, plain first: RMSNorm, RoPE, SwiGLU, GQA.
- For each: the GPT-2 version, the new version, and the one-line reason.

## 3. RoPE, properly

- The rotation picture. The test showing only relative position matters.
- The "rotate half" trap: the paper's layout vs Hugging Face's.

## 4. Same size, five years apart

- The comparison table. The honest finding: SmolLM wins on prose, ties
  on WikiText. What that says about training data.

## 5. Why GQA is a memory story

- KV bytes per token, and why that matters more than FLOPs for decode.
