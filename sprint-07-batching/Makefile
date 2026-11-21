.PHONY: test test-sim sim batch-speed serve load

# Part A needs nothing; Parts B and C need Sprints 4-6 finished.
test:
	python3 -W ignore::RuntimeWarning -m unittest discover -s tests -v

test-sim:
	python3 -m unittest tests.test_sim -v

sim:
	python3 benchmarks/sim_report.py

batch-speed:
	python3 benchmarks/batch_speed.py

serve:
	python3 serve_batched.py --max-batch 8

# with make serve running in another terminal
load:
	python3 ../sprint-06-inference-server/benchmarks/load_test.py --clients 1 4 8
