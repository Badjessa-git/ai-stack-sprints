# POST.md — Sprint 4: Fine-Tuning (outline)

Post in the series register: plain first, term second, proof third.
Plain-words explanation, then the technical term, then a real artifact
(a short code snippet with a "look at this" cue, or a table). Same
pattern as Parts 1-2.

## 1. The hook (plain, specific, no formula)

The Sprint 3 model cannot write a sentence. But it can tell you which
play a quote comes from. That gap — bad at generating, good at
recognizing — is the whole idea of fine-tuning, and it is why
fine-tuning is the most useful thing you can do with a small model.

Lead with the concrete result: the accuracy number, and the comparison
with GPT-2. No threadbro hook, no vague promise. The numbers are the
hook.

## 2. What fine-tuning is (plain first)

Pre-training taught the model to predict the next token. That is a
hard, open-ended job. Fine-tuning takes the same model and teaches it
one narrow job instead — here, "which play is this quote from?" The
model already knows the language from pre-training; fine-tuning just
points that knowledge at a specific question.

Then the term: this is called fine-tuning (or transfer learning). The
pretrained weights are the transfer — knowledge from one task reused
for another.

Proof: a short snippet of the classifier — the head (one Linear layer)
and the forward pass (transformer, last position, head). Show how
little code it takes to repurpose the whole model.

## 3. The classification head (the one new piece)

Plain: the model already produces a vector for each token. The head is
a single layer that turns the last token's vector into four numbers,
one per play. The highest number wins.

Term: classification head (a linear layer, d_model -> n_plays).

Proof: the `logits` code, with a "look at this" cue on the last
position (`x[-1:]`) and why it is the only position that has seen the
whole quote. Mention the 2-D matmul sharp edge — the engine only does
2-D matmul, so the vector stays a row.

## 4. The loss and accuracy (same math, different classes)

Plain: the loss is "how surprised was the model by the right answer."
If it gave the right play 90%, the loss is small. If it gave it 1%,
the loss is large. Accuracy is just "how often was it right."

Term: cross-entropy loss, accuracy.

Proof: the loss is the same formula as Sprint 3's next-token loss —
the only difference is the classes (4 plays instead of a 512-token
vocabulary). Show the two side by side if it fits. The accuracy table
(train/val/test) is the artifact here.

## 5. The comparison (the story)

This is the section that carries the post. Two tracks, same task, same
data:

| Track | Model | Pretraining | Test accuracy |
|-------|-------|-------------|---------------|
| 1 | d256/n2 BPE (hand-built) | 8M chars, from scratch | ? |
| 2 | GPT-2 124M (Hugging Face) | 40GB, OpenAI | ? |

Plain: the gap between the two numbers is what 40GB of pretraining
buys you. Also record how much task data each one needed — the
pretrained model should need far less.

Proof: the accuracy table, plus the per-play breakdown if it is
interesting (which plays get confused with each other — Hamlet and
Macbeth are both tragedies, so they are the likely confusion).

## 6. What did not work / what the ceiling is

Plain: the model still cannot write. Fine-tuning did not fix
generation — it was never supposed to. It gave the model a job it can
actually do. That is the honest scope: fine-tuning repurposes, it does
not upgrade.

Also: the dataset is small and the classes are imbalanced (Hamlet has
the most dialogue). Say so. The accuracy number is real but the task
is a toy.

## 7. What is next

Sprint 5 is vector search — the lookup mechanism the transformer
lacks. The fine-tuned classifier and the vector search both answer
"what is this text like?" in different ways, and Sprint 6 (hybrid
retrieval) will combine them.

## Snippets to include (verbatim, from the merged code)

- The classification head creation (`Linear(model.d_model, n_plays)`).
- The `logits` forward pass (transformer, last position, head).
- The loss (one line: softmax, negative log of the true label).
- The accuracy function (argmax == label, averaged).
- The comparison table (Track 1 vs Track 2 test accuracy).
