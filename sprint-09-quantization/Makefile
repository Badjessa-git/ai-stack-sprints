.PHONY: test quality profile profile-int8

# Unit tests use a tiny random GPT-2; test_real_gpt2 needs Sprint 4's download.
test:
	python3 -W ignore::RuntimeWarning -m unittest discover -s tests -v

# float32 vs int8 vs int4 on real GPT-2: size, bits/byte, tokens/sec
quality:
	python3 benchmarks/quality.py

profile:
	python3 benchmarks/profile_decode.py

profile-int8:
	python3 benchmarks/profile_decode.py --bits 8
