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

# Generate text from the trained NumPy model (d128, 2 layers, block 64,
# combined corpus). Just give it the prompt text:
#   make generate TEXT="To be, or not to be"
#   make generate TEXT="Once upon a time" CHARS=300 TEMP=1.0
CHARS ?= 200
TEMP ?= 0.8
generate:
	@if [ -z "$(TEXT)" ]; then echo 'Usage: make generate TEXT="your prompt here"'; exit 1; fi
	python3 benchmarks/generate.py --engine numpy \
		--load weights-numpy-d128.json \
		--d-model 128 --n-layer 2 --block-size 64 \
		--corpus data/corpus-combined.txt \
		--prompt "$(TEXT)" --chars $(CHARS) --temperature $(TEMP)
