.PHONY: test benchmark train

# Correctness suite: engine first, then nn (last test trains a tiny MLP)
test:
	python3 -m unittest discover -s tests -v

test-backward:
	python3 -m unittest tests.test_engine.TestBackwardClosures

test-forward:
	python3 -m unittest tests.test_engine.TestForward
	
# Graph-size scaling sweep + MLP micro-benchmark (run after tests pass)
benchmark:
	python3 benchmarks/bench.py

# v1 training run on moons (run after tests pass)
train:
	python3 benchmarks/train.py
