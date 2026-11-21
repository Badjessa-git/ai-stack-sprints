# Sprint 04 post outline: GPT-2 in your own code

Working title: **"I ran GPT-2 on code I wrote, and it was mostly checking"**

## 1. The artifact first

- The greedy completion of "Albert Einstein was a", from your code.
- Next to it, the same prompt from your best Sprint 3 model.
- One sentence: same architecture; the difference is OpenAI's training.

## 2. A weight file is a header and a pile of bytes

- The format in five lines. Show your `read_header`.
- Why it replaced pickle (it can't run code).
- bfloat16 in one paragraph: the top half of a float32.

## 3. The tokenizer you already built

- GPT-2's tokenizer is Part 3's algorithm. Show the 'Ġ' table and explain it.
- Rank isn't id: the one assumption from Part 3 that broke.

## 4. The details that fail silently

- Conv1D layout, the tied head, tanh GELU: the model runs either way.
- If you hit one, tell it with numbers: the loss when it was wrong vs right.

## 5. The table

- GPT-2 1.12 vs your runs on bits per byte. Discuss what the gap is
  (scale) and what it isn't (code).

## 6. What's slow, and next

- tokens/sec with no cache. The bridge to Sprint 5.
