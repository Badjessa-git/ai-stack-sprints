.PHONY: test equivalence test-all benchmark compare

# Part 1 correctness suite (15 tests)
test:
	python3 -m unittest discover -s tests -v

# Part 2 equivalence suite: current implementation vs frozen v1 —
# merges, encodings, and decodings must stay byte-for-byte identical
equivalence:
	python3 -m unittest discover -s part2 -p "test_*.py" -v

# Everything that must stay green
test-all: test equivalence

# Part 1 compression benchmark (small corpus)
benchmark:
	python3 benchmarks/benchmark.py

# Part 2 before/after timing: frozen v1 vs current implementation
compare:
	python3 part2/compare.py
