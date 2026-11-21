.PHONY: test speed chat

# Needs Sprint 4 finished: these import your sprint-04-gpt2/src.
test:
	python3 -W ignore::RuntimeWarning -m unittest discover -s tests -v

speed:
	python3 benchmarks/speed.py --new 64
	python3 benchmarks/speed.py --new 64 --prompt-len 400

# make chat TEXT="Once upon a time" TEMP=0.8 TOP_P=0.9
TEXT ?= Once upon a time
TEMP ?= 0.8
TOP_P ?= 0.9
chat:
	python3 benchmarks/stream.py "$(TEXT)" --temperature $(TEMP) --top-p $(TOP_P)
