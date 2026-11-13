.PHONY: test benchmark train

# Correctness suite: engine extensions first, then ops, then the model
test:
	python3 -m unittest discover -s tests -v

# Step time + throughput (run after tests pass)
benchmark:
	python3 benchmarks/bench.py

# Short training run on the corpus (run after tests pass)
# e.g. make train ARGS="--steps 300 --lr 0.05"
train:
	python3 benchmarks/train.py $(ARGS)
