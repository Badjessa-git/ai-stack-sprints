.PHONY: test bench bench-synthetic

test:
	python3 -W ignore::RuntimeWarning -m unittest discover -s tests -v

# recall vs latency on GPT-2's token embeddings (Sprint 4's download)
bench:
	python3 benchmarks/recall_latency.py --n 10000

bench-synthetic:
	python3 benchmarks/recall_latency.py --synthetic --n 10000
