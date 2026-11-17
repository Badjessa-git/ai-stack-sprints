"""Tests for the Sprint 4 classifier (Track 1).

These tests define the contract for src/classifier.py. They FAIL on
the stubs (NotImplementedError) and PASS once the exercises are done.

How the classifier is imported: src/classifier.py is loaded by file
path (importlib), not as `from src.classifier import ...`. That is
deliberate — classifier.py imports `src.numpy` from Sprint 3, and this
folder also has a `src/` directory, so a plain package import would
resolve `src` to the wrong one. Loading by file path keeps the two
`src` packages from colliding. The loader below is the pattern
benchmarks/finetune.py uses too.

The tests use a tiny random Transformer (not the pretrained weights)
so they are fast and self-contained. The point here is the classifier
machinery — can it produce the right shapes, can its loss go down, can
it learn a trivially separable toy set — not the pretrained model's
quality. The pretrained weights enter in benchmarks/finetune.py.
"""

import importlib.util
import os
import random
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "..", "sprint-03-transformer"))

from src.numpy.transformer import Transformer  # noqa: E402


def _load_classifier():
    """Load src/classifier.py by file path (see the module docstring)."""
    path = os.path.join(ROOT, "src", "classifier.py")
    spec = importlib.util.spec_from_file_location("classifier", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


classifier_mod = _load_classifier()
Classifier = classifier_mod.Classifier


def _adam_update(params, state, lr, t):
    """One Adam update. state is a list of [m, v] arrays, mutated in place."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for i, p in enumerate(params):
        m, v = state[i]
        g = p.grad
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        state[i] = [m, v]
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p.data -= lr * m_hat / (v_hat ** 0.5 + eps)


def _toy_model(seed=1, vocab_size=4, d_model=8, n_layer=1):
    """A tiny Transformer for the tests. Random init, fixed seed."""
    random.seed(seed)
    np.random.seed(seed)
    return Transformer(
        vocab_size, d_model=d_model, n_head=2, n_layer=n_layer, block_size=8
    )


def _separable_examples():
    """A trivially separable toy set: 2 plays, 2 distinct inputs.

    Play 0 is always the token 0 repeated; play 1 is always token 1
    repeated. The only thing that differs is the token, so the model
    has to learn "token 0 -> play 0, token 1 -> play 1". Any working
    classifier gets this to 100% — if it cannot, the machinery is
    broken (head not connected, gradients not flowing, loss wrong).
    """
    examples = []
    for _ in range(4):
        examples.append(([0, 0, 0, 0], 0))
        examples.append(([1, 1, 1, 1], 1))
    return examples


def _train_on(clf, examples, steps=100, lr=0.01, seed=1):
    """Train the classifier on examples with Adam. Returns the losses."""
    rng = random.Random(seed)
    params = clf.parameters()
    state = [[np.zeros_like(p.data), np.zeros_like(p.data)] for p in params]
    losses = []
    for t in range(1, steps + 1):
        idx, label = rng.choice(examples)
        loss = clf.loss(idx, label)
        # Zero the grads before backward (same as train.py's loop).
        for p in params:
            p.grad = np.zeros_like(p.data)
        loss.backward()
        _adam_update(params, state, lr, t)
        losses.append(float(loss.data))
    return losses


class TestHeadShape(unittest.TestCase):
    def test_logits_shape(self):
        """logits() returns one logit per play, shape (n_plays,)."""
        model = _toy_model()
        clf = Classifier(model, n_plays=4)
        out = clf.logits([1, 2, 3])
        self.assertEqual(tuple(out.data.shape), (4,))

    def test_logits_is_tensor(self):
        """logits() returns a Tensor (it must carry gradients)."""
        from src.numpy.engine import Tensor

        model = _toy_model()
        clf = Classifier(model, n_plays=4)
        out = clf.logits([1, 2, 3])
        self.assertIsInstance(out, Tensor)


class TestLoss(unittest.TestCase):
    def test_loss_is_scalar(self):
        """loss() returns a single number (a scalar Tensor)."""
        model = _toy_model()
        clf = Classifier(model, n_plays=4)
        loss = clf.loss([1, 2, 3], 1)
        self.assertEqual(float(loss.data), float(loss.data))  # finite
        self.assertEqual(np.size(loss.data), 1)

    def test_loss_positive(self):
        """Cross-entropy is -log(prob), and prob <= 1, so loss >= 0."""
        model = _toy_model()
        clf = Classifier(model, n_plays=4)
        loss = clf.loss([1, 2, 3], 1)
        self.assertGreaterEqual(float(loss.data), 0.0)

    def test_loss_decreases(self):
        """Training on the separable toy set drives the loss down."""
        model = _toy_model()
        clf = Classifier(model, n_plays=2)
        examples = _separable_examples()
        avg_before = sum(
            float(clf.loss(idx, label).data) for idx, label in examples
        ) / len(examples)
        _train_on(clf, examples, steps=100, lr=0.01)
        avg_after = sum(
            float(clf.loss(idx, label).data) for idx, label in examples
        ) / len(examples)
        self.assertLess(avg_after, avg_before)


class TestPredict(unittest.TestCase):
    def test_predict_returns_label(self):
        """predict() returns a plain int in the label range."""
        model = _toy_model()
        clf = Classifier(model, n_plays=4)
        pred = clf.predict([1, 2, 3])
        self.assertIsInstance(pred, int)
        self.assertGreaterEqual(pred, 0)
        self.assertLess(pred, 4)


class TestAccuracy(unittest.TestCase):
    def test_accuracy_on_separable_set(self):
        """After training on the trivially separable set, accuracy is 1.0."""
        model = _toy_model()
        clf = Classifier(model, n_plays=2)
        examples = _separable_examples()
        _train_on(clf, examples, steps=100, lr=0.01)
        self.assertEqual(clf.accuracy(examples), 1.0)

    def test_accuracy_range(self):
        """Accuracy is a float between 0 and 1, even untrained."""
        model = _toy_model()
        clf = Classifier(model, n_plays=2)
        acc = clf.accuracy(_separable_examples())
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_accuracy_empty(self):
        """Accuracy of no examples is 0.0 (not a crash or NaN)."""
        model = _toy_model()
        clf = Classifier(model, n_plays=2)
        self.assertEqual(clf.accuracy([]), 0.0)


class TestParameters(unittest.TestCase):
    def test_includes_head(self):
        """parameters() = the transformer's params + the head's (weight+bias)."""
        model = _toy_model()
        clf = Classifier(model, n_plays=4)
        self.assertEqual(
            len(clf.parameters()), len(model.parameters()) + 2
        )

    def test_head_weight_shape(self):
        """The head's weight is (n_plays, d_model): one row per play."""
        model = _toy_model()
        clf = Classifier(model, n_plays=4)
        # The head is the last two parameters (weight, then bias).
        w = clf.parameters()[-2]
        self.assertEqual(tuple(w.data.shape), (4, 8))


class TestGPT2Script(unittest.TestCase):
    """Track 2: the GPT-2 fine-tuning script has the right structure.

    It is a skeleton (torch is not installed here), so this test reads
    the source and checks for the pieces the script must contain,
    rather than importing it.
    """

    def _source(self):
        path = os.path.join(ROOT, "benchmarks", "finetune_gpt2.py")
        with open(path) as f:
            return f.read()

    def test_loads_pretrained_gpt2(self):
        src = self._source()
        self.assertIn("AutoModelForSequenceClassification", src)
        self.assertIn("gpt2", src)

    def test_uses_hf_trainer(self):
        src = self._source()
        self.assertIn("Trainer", src)
        self.assertIn("TrainingArguments", src)

    def test_loads_play_dataset(self):
        src = self._source()
        self.assertIn("play_quotes_train.jsonl", src)

    def test_tokenizes_and_evaluates(self):
        src = self._source()
        self.assertIn("AutoTokenizer", src)
        self.assertIn("accuracy", src.lower())


if __name__ == "__main__":
    unittest.main()
