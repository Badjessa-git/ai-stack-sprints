.PHONY: test benchmark train generate

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

# Generate text from a trained NumPy model. Just give it the prompt text,
# and optionally which weights file (and its architecture) to use:
#   make generate TEXT="To be, or not to be"
#   make generate TEXT="Once upon a time" WEIGHTS=weights-numpy-d64.json DMODEL=64 BLOCK=32
CHARS ?= 200
TEMP ?= 0.8
WEIGHTS ?= weights-numpy-d128.json
DMODEL ?= 128
NLAYER ?= 2
BLOCK ?= 64
CORPUS ?= data/corpus-combined.txt
GREEDY ?=
generate:
	@if [ -z "$(TEXT)" ]; then echo 'Usage: make generate TEXT="your prompt here"'; exit 1; fi
	python3 benchmarks/generate.py --engine numpy \
		--load $(WEIGHTS) \
		--d-model $(DMODEL) --n-layer $(NLAYER) --block-size $(BLOCK) \
		--corpus $(CORPUS) \
		--prompt "$(TEXT)" --chars $(CHARS) --temperature $(TEMP) $(GREEDY)


