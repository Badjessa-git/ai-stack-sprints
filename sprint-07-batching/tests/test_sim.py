import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim import (  # noqa: E402
    Batch, ContinuousBatcher, CostModel, Request, StaticBatcher, simulate, summarize, workload,
)

UNIT = CostModel(base=1.0, per_decode=0.0, per_prefill_token=0.0)  # every step takes 1 s


def reqs(*specs):
    """(arrival, output_len) pairs -> requests with 10-token prompts."""
    return [Request(i, a, 10, n) for i, (a, n) in enumerate(specs)]


class TestStatic(unittest.TestCase):
    def test_batch_runs_until_its_longest_member(self):
        rs = simulate(reqs((0, 1), (0, 3)), StaticBatcher(4), UNIT)
        # step 1 (prefill) makes token 1 for both; steps 2-3 decode
        self.assertEqual([r.first_token for r in rs], [1.0, 1.0])
        self.assertEqual([r.finished for r in rs], [1.0, 3.0])

    def test_late_arrival_waits_for_the_whole_batch(self):
        rs = simulate(reqs((0, 4), (0.5, 1)), StaticBatcher(4), UNIT)
        # request 1 arrives during the first batch; it starts at t=4
        self.assertEqual(rs[1].first_token, 5.0)

    def test_max_batch(self):
        rs = simulate(reqs((0, 1), (0, 1), (0, 1)), StaticBatcher(2), UNIT)
        self.assertEqual(sorted(r.finished for r in rs), [1.0, 1.0, 2.0])

    def test_schedule_shapes(self):
        s = StaticBatcher(2)
        for r in reqs((0, 2), (0, 2), (0, 2)):
            s.add(r)
        b = s.schedule(0)
        self.assertEqual((len(b.prefill), len(b.decode)), (2, 0))
        s.complete(b, 1)
        b = s.schedule(1)
        self.assertEqual((len(b.prefill), len(b.decode)), (0, 2))


class TestContinuous(unittest.TestCase):
    def test_late_arrival_joins_next_step(self):
        rs = simulate(reqs((0, 4), (0.5, 1)), ContinuousBatcher(4), UNIT)
        self.assertEqual(rs[1].first_token, 2.0)
        self.assertEqual(rs[0].finished, 4.0)

    def test_finished_leave_immediately(self):
        c = ContinuousBatcher(2)
        for r in reqs((0, 1), (0, 3), (0, 2)):
            c.add(r)
        b = c.schedule(0)
        self.assertEqual(len(b.prefill), 2)
        c.complete(b, 1)  # request 0 is done
        b = c.schedule(1)
        self.assertEqual(([r.id for r in b.decode], [r.id for r in b.prefill]), ([1], [2]))

    def test_never_exceeds_max_batch(self):
        class Watch(ContinuousBatcher):
            biggest = 0

            def schedule(self, now):
                b = super().schedule(now)
                Watch.biggest = max(Watch.biggest, len(b))
                return b

        simulate(workload(100, rate=50, seed=1), Watch(6))
        self.assertEqual(Watch.biggest, 6)


class TestSummary(unittest.TestCase):
    def test_values(self):
        rs = [Request(0, 0.0, 5, 2, generated=2, first_token=1.0, finished=2.0),
              Request(1, 1.0, 5, 2, generated=2, first_token=3.0, finished=5.0)]
        s = summarize(rs)
        self.assertAlmostEqual(s["throughput"], 4 / 5.0)
        self.assertEqual((s["ttft_p50"], s["ttft_p99"]), (2.0, 2.0))
        self.assertEqual((s["latency_p50"], s["latency_p99"]), (4.0, 4.0))

    def test_nearest_rank(self):
        rs = [Request(i, 0.0, 1, 1, generated=1, first_token=float(i), finished=float(i)) for i in range(1, 11)]
        s = summarize(rs)
        self.assertEqual(s["latency_p50"], 6.0)  # sorted[int(0.5 * 10)] = sorted[5]
        self.assertEqual(s["latency_p99"], 10.0)


class TestPolicies(unittest.TestCase):
    """The point of Part A, as numbers."""

    def test_every_request_finishes(self):
        for sched in (StaticBatcher(8), ContinuousBatcher(8)):
            rs = simulate(workload(150, rate=10, seed=2), sched)
            self.assertTrue(all(r.generated == r.output_len for r in rs))
            self.assertTrue(all(r.arrival <= r.first_token <= r.finished for r in rs))

    def test_continuous_beats_static_under_load(self):
        st = summarize(simulate(workload(300, rate=20, seed=0), StaticBatcher(16)))
        co = summarize(simulate(workload(300, rate=20, seed=0), ContinuousBatcher(16)))
        self.assertGreater(co["throughput"], st["throughput"])
        self.assertLess(co["latency_p50"], st["latency_p50"])
        self.assertLess(co["ttft_p99"], st["ttft_p99"])

    def test_batch_of_one_is_sprint6(self):
        # max_batch 1: both policies are the one-at-a-time server
        a = summarize(simulate(workload(50, rate=5, seed=3), StaticBatcher(1)))
        b = summarize(simulate(workload(50, rate=5, seed=3), ContinuousBatcher(1)))
        self.assertAlmostEqual(a["throughput"], b["throughput"])


if __name__ == "__main__":
    unittest.main()
