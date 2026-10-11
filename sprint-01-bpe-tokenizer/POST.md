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

   The first step we built here was the tokenizer. Computers don't actually see the letters that we type into the GPT prompt. The text is converted into tokens that is then shared with the model. The task of the day was completing the build going from text -> tokens -> original text. I started by implementing get_stats and . get_stats enabled us to extract how ofen adjacent pairs occurs in a list of tokens. The next one was implementing merge pair which allowed us to replace a specific pair with a new id. As a result of the two helper function we were able to understand the count of pairs in a text and compress them into new ids (>=256) to represent the text inputed for our model to understand.

   Now we move on to the training loop, our objective here was to build the an in memory encoding of texts so that whenever new text is introduced the token_id list generated remains stable making it easier for models to undertand the text that is sent. The logic is as follow, first transform the text into bytes. Our base vocab is the 256 possible byes from regular text encoding, we want any additional encoding to have id 256 and up. We first checked that the vocab size was larger than 255 to make sure that we can encoe more complex text without disrupting ACII mapping. We iteratively get the overlapping pairs in the text, we pick the most frequent pair record it in our in memory map as (left, right) -> new_id (256+iteration count starting from 0) and then merge the pair back into the original token list. We did it vocab_size - 256 times to stay in the input bounds (Note: why does vocab size matter?). At the end of the training model, we have a list of pairs in memory that represents the text. In both encoding and decoding we leverage the in memory map that has been built to turn new text in tokens and the reverse back into the original strings. 

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
