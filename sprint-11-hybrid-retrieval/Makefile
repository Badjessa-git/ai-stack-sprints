.PHONY: test eval eval-hard rag

test:
	python3 -m unittest discover -s tests -v

# BM25 vs dense vs hybrid (needs Sprint 3's WikiText file and Sprint 4's GPT-2)
eval:
	python3 benchmarks/evaluate.py

eval-hard:
	python3 benchmarks/evaluate.py --headings-only

# with a server running (Sprint 8: cd ../sprint-08-modern-arch && make serve)
Q ?= What kind of game is Valkyria Chronicles III?
rag:
	python3 rag.py "$(Q)"
