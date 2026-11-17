"""A text classifier built on the pretrained Sprint 3 transformer.

This is the core of Sprint 4, Track 1. The idea is small, and it is
the whole point of fine-tuning:

    Pre-training (Sprint 3) taught the transformer to predict the next
    token. That is a hard, open-ended job, and the model is not good at
    it — it writes word-shaped chunks, not sentences. But in learning
    to predict, it built up useful internal features: it knows
    Shakespearean vocabulary, it can tell a philosophical passage from
    a violent one, it recognizes the rhythm of verse. Those features
    live in the weights.

    Fine-tuning reuses those features for a narrower job: read a quote
    and say which of four plays it comes from. That is a much easier
    decision than generating text — one label out of four, instead of
    one token out of 512, over and over. A model that cannot write can
    still be good at recognizing.

How it works:
- The transformer is loaded with its Sprint 3 weights (d256/n2 BPE).
  It is NOT re-initialized. That is the "pre-trained" part.
- A new classification head is added on top: one Linear layer,
  d_model -> n_plays. It maps the final position's hidden state to
  one logit per play. The head starts random — it is the only part
  that has never seen data.
- The quote is run through the transformer exactly as in Sprint 3
  (embeddings, blocks, final LayerNorm). The lm_head is NOT used —
  instead, the final position's hidden state (the last token's vector,
  after ln_f) goes through the classification head.
- The loss is cross-entropy over the play labels: softmax the logits,
  take the negative log probability of the true play. Same math as the
  next-token loss in Sprint 3, but over 4 classes instead of a 512-token
  vocabulary.
- Training updates the WHOLE model (head + transformer) with a small
  learning rate (Adam, lr 0.001). The model is small, so there is no
  need to freeze anything — freezing is for big models, where you want
  to keep the pretrained features intact and avoid catastrophic
  forgetting (see the README).

The exercises in this file (all marked TODO):
1. __init__: create the classification head.
2. logits: forward pass — transformer, last position, head.
3. loss: cross-entropy over the play labels.
4. predict: argmax of the logits (inference, no gradients).
5. accuracy: fraction of examples predicted correctly.
6. parameters: the transformer's parameters plus the head's.
"""

import os
import sys

# The transformer lives in Sprint 3. Add its root to the path so that
# `src.numpy` resolves to sprint-03-transformer/src, not to this
# folder's src/ (which holds only this file). This import must happen
# before anything imports this folder's `src` as a package.
_SPRINT3_ROOT = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
    "sprint-03-transformer",
)
if _SPRINT3_ROOT not in sys.path:
    sys.path.insert(0, _SPRINT3_ROOT)

from src.numpy.engine import Tensor  # noqa: E402
from src.numpy.nn import Linear, Module  # noqa: E402
from src.numpy.transformer import Transformer, softmax  # noqa: E402

import numpy as np  # noqa: E402


class Classifier(Module):
    """Play classifier on top of a pretrained Transformer.

    model: a Transformer instance whose weights are already loaded
        (from Sprint 3, e.g. weights-numpy-d256-n2-bpe.json). The
        classifier does not build or initialize the transformer — the
        caller does that, so the same pretrained weights can be reused.
    n_plays: number of classes (4 for this sprint).
    """

    def __init__(self, model, n_plays):
        # TODO (exercise 1): store the model, and create the
        # classification head: a Linear layer from the model's hidden
        # size (model.d_model) to n_plays. It maps one hidden vector
        # to one logit per play. The head starts random (Linear's
        # default init) — it has never seen data, and training is
        # what teaches it.
        raise NotImplementedError

    def logits(self, idx, training=True):
        """Run the quote through the transformer and the head.

        idx: list of token ids (the BPE-encoded quote).
        Returns a Tensor of shape (n_plays,) — one logit per play.

        The forward pass is the transformer's, minus the lm_head:
        token + positional embeddings, then each block, then the final
        LayerNorm. Take the LAST position's hidden state (the vector
        for the final token, after ln_f) and feed it through the head.
        Why the last position: attention is causal, so the last
        position is the only one that has seen the whole quote.
        training is passed through to the blocks (dropout on/off).
        """
        # TODO (exercise 2): replicate the transformer's forward pass
        # up to ln_f (see Transformer.__call__ in
        # sprint-03-transformer/src/numpy/transformer.py — same steps,
        # but stop before lm_head), take the last row, apply the head.
        raise NotImplementedError

    def loss(self, idx, label):
        """Cross-entropy over the play labels. Returns a scalar Tensor.

        idx: list of token ids (the BPE-encoded quote).
        label: int, the true play index (0..n_plays-1).

        Softmax the logits so they sum to 1 (a probability per play),
        then take the negative log of the probability assigned to the
        true play. If the model gave the right play 90%, the loss is
        small (-log 0.9 = 0.11). If it gave it 1%, the loss is large
        (-log 0.01 = 4.6). Same math as the next-token loss — the only
        difference is what the classes are (plays, not tokens).
        """
        # TODO (exercise 3): logits = self.logits(idx), then
        # probs = softmax(logits), then -probs[label].log(). The
        # result must be a scalar Tensor so loss.backward() works.
        raise NotImplementedError

    def predict(self, idx):
        """The model's single best guess: the play index with the
        highest logit. Returns a plain int. No gradients — this is
        inference, not training, so run it with training=False.
        """
        # TODO (exercise 4): logits = self.logits(idx, training=False),
        # return int(np.argmax(logits.data)). np.argmax reads the
        # numbers out of the Tensor; .data is the plain array.
        raise NotImplementedError

    def accuracy(self, examples):
        """Fraction of examples predicted correctly. Returns a float.

        examples: a list of (idx, label) pairs — idx is a list of
        token ids, label is the true play index. Runs predict() on
        each one and counts the matches. 0.25 is chance (4 plays);
        1.0 is perfect. This is the number that matters for the
        Sprint 4 comparison, not the loss.
        """
        # TODO (exercise 5): loop over examples, count
        # self.predict(idx) == label, return correct / len(examples).
        # Guard the empty case (return 0.0 if there are no examples).
        raise NotImplementedError

    def parameters(self):
        """All trainable parameters: the transformer's plus the head's.

        The training loop (benchmarks/finetune.py) calls this to get
        the full list for zero_grad, backward, and the Adam update.
        The head's parameters must be included, or the head never
        learns — the most common bug in this file.
        """
        # TODO (exercise 6): return the model's parameters plus the
        # head's. model.parameters() gives the transformer's list;
        # self.head.parameters() gives the head's.
        raise NotImplementedError
