.PHONY: test finetune finetune-gpt2 dataset

test:
	python3 -m unittest discover -s tests -v

dataset:
	python3 data/build_dataset.py

finetune:
	python3 benchmarks/finetune.py --steps 500 --lr 0.001 $(ARGS)

finetune-gpt2:
	python3 benchmarks/finetune_gpt2.py $(ARGS)
