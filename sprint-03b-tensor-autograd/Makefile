.PHONY: test benchmark

# Gradient checks for every op, then attention + an MLP end to end
test:
	python3 -m unittest discover -s tests -v

# Forward + backward timings at growing sizes (after the tests pass)
benchmark:
	python3 benchmarks/bench.py
