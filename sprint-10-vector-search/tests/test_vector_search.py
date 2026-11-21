import functools
import os
import sys
import unittest
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_search import (  # noqa: E402
    HNSW, BruteForce, IVFIndex, clustered, kmeans, normalize, recall_at_k, top_k,
)

X = clustered(2000, 32, clusters=20, seed=0)
Q = clustered(50, 32, clusters=20, seed=0)[:50] + 0.05


@functools.lru_cache(maxsize=None)
def truth(k=10):
    """Exact top-k for every query, from your BruteForce (computed on
    first use, so a stub fails per test, not at import)."""
    bf = BruteForce(X)
    return tuple(tuple(bf.search(q, k)[0]) for q in Q)


def mean_recall(index_search):
    return np.mean([recall_at_k(index_search(q), t) for q, t in zip(Q, truth())])


class TestBruteForce(unittest.TestCase):
    def test_normalize(self):
        n = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
        np.testing.assert_allclose(n, [[0.6, 0.8], [0.0, 0.0]])
        self.assertEqual(n.dtype, np.float32)

    def test_top_k(self):
        np.testing.assert_array_equal(top_k(np.array([0.1, 0.9, 0.5, 0.7]), 2), [1, 3])
        self.assertEqual(len(top_k(np.array([1.0, 2.0]), 5)), 2)

    def test_exact(self):
        bf = BruteForce(X)
        ids, sims = bf.search(Q[0], 5)
        expected = np.argsort(-(normalize(X) @ normalize(Q[0])))[:5]
        np.testing.assert_array_equal(ids, expected)
        self.assertTrue((np.diff(sims) <= 0).all(), "best first")

    def test_finds_itself(self):
        ids, sims = BruteForce(X).search(X[123], 1)
        self.assertEqual(int(ids[0]), 123)
        self.assertAlmostEqual(float(sims[0]), 1.0, places=5)

    def test_recall(self):
        self.assertEqual(recall_at_k([1, 2, 3], [3, 4, 1, 9]), 0.5)


class TestKMeans(unittest.TestCase):
    def test_finds_obvious_clusters(self):
        rng = np.random.default_rng(1)
        a = normalize(np.array([1.0, 0, 0]) + 0.05 * rng.standard_normal((50, 3)))
        b = normalize(np.array([0, 1.0, 0]) + 0.05 * rng.standard_normal((50, 3)))
        C, labels = kmeans(np.vstack([a, b]), 2, seed=0)
        self.assertEqual(len(set(labels[:50])), 1)
        self.assertEqual(len(set(labels[50:])), 1)
        self.assertNotEqual(labels[0], labels[50])
        np.testing.assert_allclose(np.linalg.norm(C, axis=1), 1.0, atol=1e-5)

    def test_deterministic(self):
        a = kmeans(normalize(X[:300]), 8, seed=3)[1]
        b = kmeans(normalize(X[:300]), 8, seed=3)[1]
        np.testing.assert_array_equal(a, b)


class TestIVF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ivf = IVFIndex(n_lists=32, seed=0).build(X)

    def test_every_vector_in_one_list(self):
        ids = np.sort(np.concatenate(self.ivf.lists))
        np.testing.assert_array_equal(ids, np.arange(len(X)))

    def test_probing_everything_is_exact(self):
        self.assertEqual(mean_recall(lambda q: self.ivf.search(q, 10, nprobe=32)[0]), 1.0)

    def test_recall_grows_with_nprobe(self):
        r1 = mean_recall(lambda q: self.ivf.search(q, 10, nprobe=1)[0])
        r8 = mean_recall(lambda q: self.ivf.search(q, 10, nprobe=8)[0])
        self.assertLess(r1, r8)
        self.assertGreater(r8, 0.9)


class TestHNSW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = HNSW(M=12, ef_construction=80, seed=0).build(X)

    def test_layers_thin_out(self):
        sizes = [len(layer) for layer in self.h.graph]
        self.assertEqual(sizes[0], len(X))
        self.assertTrue(all(a > b for a, b in zip(sizes, sizes[1:])), sizes)

    def test_degree_limits(self):
        self.assertTrue(all(len(v) <= self.h.M0 for v in self.h.graph[0].values()))
        for layer in self.h.graph[1:]:
            self.assertTrue(all(len(v) <= self.h.M for v in layer.values()))

    def test_high_recall(self):
        self.assertGreater(mean_recall(lambda q: self.h.search(q, 10, ef=64)[0]), 0.95)

    def test_recall_grows_with_ef(self):
        lo = mean_recall(lambda q: self.h.search(q, 10, ef=10)[0])
        hi = mean_recall(lambda q: self.h.search(q, 10, ef=100)[0])
        self.assertLessEqual(lo, hi)

    def test_far_fewer_comparisons_than_brute_force(self):
        self.h.n_dist = 0
        for q in Q:
            self.h.search(q, 10, ef=64)
        self.assertLess(self.h.n_dist / len(Q), len(X) / 2)

    def test_finds_itself(self):
        self.assertEqual(int(self.h.search(X[77], 1, ef=32)[0][0]), 77)


if __name__ == "__main__":
    unittest.main()
