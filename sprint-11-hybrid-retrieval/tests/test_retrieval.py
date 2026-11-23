import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import (  # noqa: E402
    InvertedIndex, SIFEmbedder, hybrid_search, mrr, recall_at_k, rrf, tokenize,
)

DOCS = {
    0: "the cat sat on the mat",
    1: "the dog chased the cat around the garden",
    2: "stock markets fell sharply on monday",
}


def index():
    ix = InvertedIndex()
    for d, t in DOCS.items():
        ix.add(d, t)
    return ix


class TestBM25(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("The Cat's 3 lives, (2024)!"), ["the", "cat", "s", "3", "lives", "2024"])

    def test_postings(self):
        ix = index()
        self.assertEqual(ix.postings["the"], {0: 2, 1: 3})
        self.assertEqual(ix.doc_len, {0: 6, 1: 8, 2: 6})

    def test_idf(self):
        ix = index()
        self.assertAlmostEqual(ix.idf("cat"), math.log(1 + (3 - 2 + 0.5) / (2 + 0.5)))
        self.assertAlmostEqual(ix.idf("stock"), math.log(1 + (3 - 1 + 0.5) / (1 + 0.5)))
        self.assertGreater(ix.idf("unseen"), ix.idf("stock"))

    def test_score_by_hand(self):
        ix = index()
        avg = (6 + 8 + 6) / 3
        tf, dl = 1, 6
        expected = ix.idf("mat") * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avg))
        self.assertAlmostEqual(ix.bm25_scores("mat")[0], expected)

    def test_only_matching_docs(self):
        self.assertEqual(set(index().bm25_scores("cat")), {0, 1})
        self.assertEqual(index().bm25_scores("zebra"), {})
        self.assertEqual(index().bm25_scores(""), {})

    def test_shorter_doc_wins_on_equal_tf(self):
        # "cat" appears once in both; doc 0 is shorter, so it ranks first
        self.assertEqual(index().search("cat"), [0, 1])

    def test_rare_term_beats_common(self):
        self.assertEqual(index().search("the markets")[0], 2)


class FakeTok:
    vocab = {"cat": 0, "dog": 1, "car": 2, "auto": 3, "the": 4}

    def encode(self, text):
        return [self.vocab[w] for w in text.split() if w in self.vocab]


E = np.array([
    [1.0, 0.1, 0.0, 0.5],   # cat
    [0.9, 0.2, 0.0, 0.5],   # dog: near cat
    [0.0, 0.1, 1.0, 0.5],   # car
    [0.1, 0.0, 0.9, 0.5],   # auto: near car
    [0.3, 0.3, 0.3, 3.0],   # the: frequent, mostly "the shared direction"
], np.float32)


class TestSIF(unittest.TestCase):
    def setUp(self):
        self.sif = SIFEmbedder(E, FakeTok().encode, a=0.1)
        self.texts = ["the cat", "the dog", "the car", "the auto", "the the cat"]
        self.V = self.sif.fit(self.texts)

    def test_shapes_and_unit_length(self):
        self.assertEqual(self.V.shape, (5, 4))
        np.testing.assert_allclose(np.linalg.norm(self.V, axis=1), 1.0, atol=1e-5)

    def test_frequent_tokens_weigh_less(self):
        self.assertLess(self.sif.weights[4], self.sif.weights[0])
        self.assertEqual(self.sif.weights[1], self.sif.weights[2])  # same count
        self.assertGreater(len(self.sif.weights), 0)

    def test_common_direction_removed(self):
        np.testing.assert_allclose(self.V @ self.sif.pc, 0.0, atol=1e-5)

    def test_meaning_not_spelling(self):
        sim = lambda a, b: float(self.sif.embed(a) @ self.sif.embed(b))
        self.assertGreater(sim("the car", "the auto"), sim("the car", "the cat"))

    def test_empty_text(self):
        np.testing.assert_array_equal(self.sif.embed("zebra"), np.zeros(4))


class TestFusion(unittest.TestCase):
    def test_rrf_values(self):
        out = rrf([[1, 2, 3], [3, 1]], k=60)
        self.assertEqual(out[0], 1)  # 1/61 + 1/62 beats 3's 1/63 + 1/61
        self.assertEqual(out, [1, 3, 2])

    def test_rrf_weights(self):
        self.assertEqual(rrf([[1, 2], [2, 1]], weights=[3.0, 1.0])[0], 1)
        self.assertEqual(rrf([[1, 2], [2, 1]], weights=[1.0, 3.0])[0], 2)

    def test_rrf_ties_by_id(self):
        self.assertEqual(rrf([[5], [4]]), [4, 5])

    def test_hybrid(self):
        ix = index()
        dense = lambda q, k: [2, 0, 1][:k]
        out = hybrid_search(ix, dense, "cat", k=3)
        self.assertEqual(set(out), {0, 1, 2})
        self.assertEqual(out[0], 0)  # BM25 #1 and dense #2


class TestMetrics(unittest.TestCase):
    def test_recall(self):
        self.assertEqual(recall_at_k([1, 2, 3, 4], [2, 9], 3), 0.5)
        self.assertEqual(recall_at_k([1, 2, 3, 9], [2, 9], 3), 0.5)  # 9 is past k

    def test_mrr(self):
        self.assertEqual(mrr([5, 6, 7], [7]), 1 / 3)
        self.assertEqual(mrr([5, 6], [7]), 0.0)


if __name__ == "__main__":
    unittest.main()
