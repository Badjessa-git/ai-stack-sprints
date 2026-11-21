.PHONY: test test-real download generate bpb

# Everything that runs without the download (tiny random GPT-2)
test:
	python3 -W ignore::RuntimeWarning -m unittest tests.test_safetensors tests.test_tokenizer tests.test_gpt2 -v

# Acceptance: OpenAI's real weights (after make download)
test-real:
	python3 -W ignore::RuntimeWarning -m unittest tests.test_real_gpt2 -v

download:
	python3 data/download_gpt2.py

# make generate TEXT="Albert Einstein was a"
TEXT ?= Albert Einstein was a
generate:
	python3 benchmarks/generate.py "$(TEXT)" --tokens 40

bpb:
	python3 benchmarks/bits_per_byte.py
