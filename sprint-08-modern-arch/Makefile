.PHONY: test test-real download compare serve

# Needs Sprints 4 and 5 finished (safetensors, softmax, the BPE, the KVCache).
test:
	python3 -W ignore::RuntimeWarning -m unittest tests.test_llama tests.test_hf_tokenizer -v

test-real:
	python3 -W ignore::RuntimeWarning -m unittest tests.test_real_smollm -v

download:
	python3 data/download_smollm.py

# GPT-2 vs SmolLM on the same text (needs both downloads)
compare:
	python3 benchmarks/compare.py

# SmolLM behind your Sprint 6 server
serve:
	python3 serve.py
