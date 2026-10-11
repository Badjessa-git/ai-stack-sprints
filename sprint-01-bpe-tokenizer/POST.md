# Post draft — Sprint 01

Working title: "Why LLMs can't count the r's in strawberry — a tokenizer, built by hand"
Target: 900–1,500 words. Your words first; editing comes after a full draft exists.

## Prompts (delete as you fill each section)

1. **The problem, in one paragraph.** What a tokenizer does, and the moment
   I've realized now that I've not manually coded in a while as AI Driven Development has allowed me to move as a much faster pace than I had before. However I still thinking being able to code by hand, understand systems is still a valuable skill for a software engineer and I could feel my skills dulling as AI starts taking a lot of dev work away from you. This prompted me to look for a way to still keep up with my coding skills while keeping up with the industry. I realized that I've been using LLMs for a while now without doing a deep dive into them and I wanted to undestand how it works. That's why I'm workin on this series with the goal of rebuilding the model AI stack on my mac mini. I don't need to built GPT 5 but I aim to end up with a somthing that I can ping and get a response from.

   The rules are simple
      - Every two weeks (or earlier if i have the time) I will take on a sprint and work on this and publish a post of my experience.
      - I'm using AI to set up the scaffolding, benchmark and tests for each of the different sprint since I will only be doing this on weekends.
      - All the code has to be written manually, no auto-complete, no AI. 
      - I can use the web to look up syntax but I will try to solve all the problems first before looking outwards for a solution.
   
   This is the first sprint and today we are building a tokenizer, LLMs don't see letters when you type into a composer, they use tokenizer to compress the text into a language that they can understand and leverage the same tokenizer to talk to us.


2. **My mental model before I started.** How did you *think* tokenization
   worked on Friday? Be specific — this section is only honest if you write
   it before Sunday's version of you rewrites history.

3. **The naive build.** Pair counting → merge → repeat. Where did your first
   version's logic live, and what did the code look like at its simplest?

   The first step we built here was the tokenizer. Computers don't actually see the letters that we type into the GPT prompt. The text is converted into tokens that is then shared with the model. The task of the day was completing the build going from text -> tokens -> original text. I started by implementing get_stats and . get_stats enabled us to extract how ofen adjacent pairs occurs in a list of tokens. The next one was implementing merge pair which allowed us to replace a specific pair with a new id. As a result of the two helper function we were able to understand the count of pairs in a text and compress them into new ids (>=256) to represent the text inputed for our model to understand.

   Now we move on to the training loop, our objective here was to build the an in memory encoding of texts so that whenever new text is introduced the token_id list generated remains stable making it easier for models to undertand the text that is sent. The logic is as follow, first transform the text into bytes. Our base vocab is the 256 possible byes from regular text encoding, we want any additional encoding to have id 256 and up. We first checked that the vocab size was larger than 255 to make sure that we can encoe more complex text without disrupting ACII mapping. We iteratively get the overlapping pairs in the text, we pick the most frequent pair record it in our in memory map as (left, right) -> new_id (256+iteration count starting from 0) and then merge the pair back into the original token list. We did it vocab_size - 256 times to stay in the input bounds (Note: why does vocab size matter?). At the end of the training model, we have a list of pairs in memory that represents the text. In both encoding and decoding we leverage the in memory map that has been built to turn new text in tokens and the reverse back into the original strings. 

4. **Where it broke.** Paste a real failure: a failing test, a wrong merge
   order, a roundtrip that mangled an emoji. What was the actual bug?

   The piece that was the most challenging in implementing this tokenizer is the decode function. Encoding was pretty straight forward, use and iterator to find the lowest rank (earliest insertion which python3 ensures on dict), call merge_pair for that pair and repeat the process for the all of the in-memory pairs. 
   
   The reverse was a lot more challenging. My first attempt was to do this iteratively, for each token in the input check if it exists in the in-memory map, find the tuple key and replace it with the left and right pair in the token list, turn them into a bytearray and call decode to get the original string. The thing I failed to notice is that the pair could also be the result of an earlier pair encoding. Here's a visual to make it easy to understand

   Lets look at an example to make it easy, lets assume this setup
   ```python
   tok = BPETokenizer()
   tok.train("the quick brown fox jumps over the lazy dog. " * 30, vocab_size=320)
   encoding = tok.encode("the quick brown fox jumps over the lazy dog")
   print("encoding", encoding)
   ```

   ```bash
   Merges: {(116, 104): 256, (256, 101): 257, (257, 32): 258, (32, 258): 259, (113, 117): 260, (260, 105): 261, (261, 99): 262, (262, 107): 263, (263, 32): 264, (264, 98): 265, (265, 114): 266, (266, 111): 267, (267, 119): 268, (268, 110): 269, (269, 32): 270, (270, 102): 271, (271, 111): 272, (272, 120): 273, (273, 32): 274, (274, 106): 275, (275, 117): 276, (276, 109): 277, (277, 112): 278, (278, 115): 279, (279, 32): 280, (280, 111): 281, (281, 118): 282, (282, 101): 283, (283, 114): 284, (284, 259): 285, (285, 108): 286, (286, 97): 287, (287, 122): 288, (288, 121): 289, (289, 32): 290, (290, 100): 291, (291, 111): 292, (292, 103): 293, (293, 46): 294, (294, 259): 295, (295, 295): 296, (296, 296): 297, (297, 297): 298, (298, 298): 299}

   Encoding: [258, 293]
   ```

   The issue with the approach described is that our output returned would be
   ```bash
   [257, 32, 292, 103]
   ```

   257 and 292 were additional encoding since we do multiple passes when creating our merge (the train loop runs vocab_size - 256 times) so when turning this into a string we get an empty string.

   ![Alt Text](/sprint-01-bpe-tokenizer/assets/decode-before.gif)

   Enters Recursion, the issue here is that we did this in one pass, we need to recursively decode the pairs until we get values that are < 256. Our new algorithm went as follow, for each token id call a private decode function that takes a single token, if the token exists in our values then get the tuple key. Check that each left and right values are below 256. If they're not then recursively call decode again until we get a number that is below 256. Return an array with the token_id and then build the array that way which will from which we can build the initial token_ids list and turn it into a string.

   With recursion, we can keep breaking down each encoding until we get the value it maps to in the base vocab. We can then decode base on that value and get our original string

   ```bash
   [116, 104, 101, 32, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102, 111, 120, 32, 106, 117, 109, 112, 115, 32, 111, 118, 101, 114, 32, 116, 104, 101, 32, 108, 97, 122, 121, 32, 100, 111, 103]
   ```

   ![Alt Text](/sprint-01-bpe-tokenizer/assets/decode-tree.gif)

5. **The numbers.** Quote `make benchmark`: tokens vs bytes at each vocab
   size. What surprised you about the compression curve?
   
   I also ran some benchmarks with various vocab sizes on the same corpus
   Note: The corpus is quite small so this doesn't tell us if this approach will work well at scale, I'm mostly curious here about token the size in relation to vocab increase.

   corpus: 4,644 chars / 4,646 bytes
   vocab= 256  merges=  0  tokens= 4,646  ratio=100.00%  bytes/token=1.00  train=0.00s  encode=0.000s
   vocab= 384  merges=128  tokens= 2,346  ratio=50.50%  bytes/token=1.98  train=0.10s  encode=0.041s
   ocab= 512  merges=256  tokens= 1,882  ratio=40.51%  bytes/token=2.47  train=0.18s  encode=0.070s

   At 256 no merges occurs and the number of tokens is the same as the input size which is expected. This is the possible bytes from regular text encoding so no training truly occurs there. 

   We observe that the number of merges increases as the vocab sizes increases but the the number of tokens generated decreases. At size 512 we see a 59.5% reduction in the number of bytes returned, a valuable learnings here is that vocab increase reduces number of tokens which means shorter sequences and less processing per request but it is at a cost of time spent on training and number of iteration. At a truly large value, we would see the number of tokens be lower but the time to train would take a lot longer because of the iteration process.

   Another interesting observation is the number of tokens cut down as vocab size increases decreases 256 -> 384 => 2300 and 384 -> 512 => 464. This is because the most frequent pairs are eaten up first and merge_pairs go after rarer pairs, we observer diminishing returns as we increase the sizes so finding the right tradeoffs based on your training set is important to avoid wasting compute

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
