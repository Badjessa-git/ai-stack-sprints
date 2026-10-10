# Post draft — Sprint 01

Working title: "Why LLMs can't count the r's in strawberry — a tokenizer, built by hand"
Target: 900–1,500 words. Your words first; editing comes after a full draft exists.

## Prompts (delete as you fill each section)

1. **The problem, in one paragraph.** What a tokenizer does, and the moment
   you realised a model never actually sees letters or words.

2. **My mental model before I started.** How did you *think* tokenization
   worked on Friday? Be specific — this section is only honest if you write
   it before Sunday's version of you rewrites history.

3. **The naive build.** Pair counting → merge → repeat. Where did your first
   version's logic live, and what did the code look like at its simplest?

4. **Where it broke.** Paste a real failure: a failing test, a wrong merge
   order, a roundtrip that mangled an emoji. What was the actual bug?

5. **The numbers.** Quote `make benchmark`: tokens vs bytes at each vocab
   size. What surprised you about the compression curve?

6. **Why this data structure.** Merge rank as dict insertion order; re-counting
   pairs from scratch every round vs incremental updates — cost vs simplicity.
   What do production tokenizers do instead, and why can they get away with it?

7. **What I misunderstood on Friday vs Sunday.** One concrete thing, not a
   platitude. If you can't name one, you stopped too early.

## Before publishing checklist

- [ ] Every number in the post came from a benchmark run I can reproduce
- [ ] One code snippet maximum per section, and I can explain each line
- [ ] No sentence I couldn't defend out loud to another engineer
- [ ] I read it out loud once (writing in general improves out loud)
