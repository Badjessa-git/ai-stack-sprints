#!/usr/bin/env python3
"""Retrieval-augmented generation, end to end on your own stack.

    # terminal 1: your Sprint 8 server (SmolLM-135M), or Sprint 6's (GPT-2)
    cd ../sprint-08-modern-arch && python3 serve.py --port 8000
    # terminal 2:
    python3 rag.py "What kind of game is Valkyria Chronicles III?"

Retrieves the top paragraphs with your BM25 index (over the first
--articles WikiText articles), puts them in the prompt, and asks your
server's /v1/completions. SmolLM-135M is a small base model, not a chat
model: it continues text, so the prompt is shaped as a document to
continue. Expect it to quote the context more than reason over it.
"""

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import data  # noqa: E402
from src.deps import REPO  # noqa: E402
from src.retrieval import InvertedIndex  # noqa: E402

CORPUS = os.path.join(REPO, "sprint-03-transformer", "data", "wiki103-100m.txt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--articles", type=int, default=500)
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args()

    docs, _ = data.load(CORPUS, args.articles)
    index = InvertedIndex()
    for d, _, text in docs:
        index.add(d, text)
    hits = index.search(args.question, args.top)
    context = "\n\n".join(" ".join(docs[d][2].split()[:120]) for d in hits)
    prompt = f"{context}\n\nQuestion: {args.question}\nAnswer:"

    body = json.dumps({"prompt": prompt, "max_tokens": args.max_tokens, "temperature": 0,
                       "stop": ["\n\n", "Question:"]}).encode()
    req = urllib.request.Request(args.server + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=300).read())
    print("Sources:")
    for d in hits:
        print(f"  [{d}] {docs[d][1]}: {docs[d][2][:90]}...")
    print(f"\nAnswer ({out['model']}):{out['choices'][0]['text']}")
