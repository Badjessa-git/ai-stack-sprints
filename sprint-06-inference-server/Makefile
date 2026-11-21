.PHONY: test serve serve-fake load

# Needs Sprints 4 and 5 finished for the GPT-2 engine; the rest uses FakeEngine.
test:
	python3 -W ignore::RuntimeWarning -m unittest discover -s tests -v

serve:
	python3 serve.py

serve-fake:
	python3 serve.py --fake

# with a server running in another terminal
load:
	python3 benchmarks/load_test.py --clients 1 2 4 8
