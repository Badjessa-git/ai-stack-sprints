"""Track 2: fine-tune a pretrained GPT-2 for play classification.

This is the library path — the other half of the Sprint 4
comparison. Track 1 fine-tunes the hand-built NumPy transformer;
this track fine-tunes OpenAI's GPT-2 (124M parameters), pretrained on
40GB of text. Same task, same play-quotes dataset, different starting
point. The comparison shows how much that pretraining is worth.

How this differs from Track 1:
- Track 1 starts from YOUR weights (d256/n2 BPE, trained from scratch
  on 8M characters). This track starts from OpenAI's weights, which
  already know English. That is the whole point of the comparison.
- Track 1 uses your hand-rolled NumPy engine. This track uses
  PyTorch + Hugging Face, because GPT-2's weights only exist in that
  format — your NumPy engine cannot load them (different architecture,
  different dimensions). Using the library here does not diminish the
  hand-built model; it is the only way to get a pretrained model.
- The fine-tuning itself is the same idea in both tracks: add a
  classification head (here it is built in — AutoModelForSequence-
  Classification adds one for you), train on (quote, label) pairs,
  measure accuracy on held-out quotes.

Install (once, in the environment where you will run this):
    pip install torch transformers datasets

    torch is the tensor library (the grown-up version of your NumPy
    engine — same idea, but with GPU support and a much bigger op set).
    transformers is Hugging Face's library: it knows how to download
    GPT-2, build the model, and tokenize text. datasets loads the
    play-quotes files. None of this is installed here yet — this
    script is a skeleton to run after you install them.

Usage (after installing):
    python3 benchmarks/finetune_gpt2.py
    python3 benchmarks/finetune_gpt2.py --model distilgpt2 --epochs 3

Notes:
- GPT-2's tokenizer is BPE (the same idea as your Sprint 1 tokenizer,
  but a 50,257-token vocabulary trained on web text). It handles the
  quotes directly — no need to train merges.
- GPT-2 has no padding token by default; this script sets the pad
  token to the end-of-text token, which is the standard fix.
- On a CPU this is slow (GPT-2 is 124M parameters). A few epochs on
  ~4,000 short quotes is still feasible — tens of minutes, not hours.
  On a GPU it is minutes. The Mac's GPU is not usable by PyTorch
  here (no MPS support in this environment), so expect CPU speed.
- The accuracy here is directly comparable to Track 1's accuracy on
  the same test split (data/play_quotes_test.jsonl).
"""

import argparse
import json
import os

# TODO: install torch, transformers, datasets (see the module docstring),
# then uncomment the imports below. They are commented out so this file
# can be read and structure-checked without the libraries installed.
# import torch
# from datasets import Dataset
# from transformers import (
#     AutoModelForSequenceClassification,
#     AutoTokenizer,
#     Trainer,
#     TrainingArguments,
# )

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# Label index -> play name. Must match data/labels.json (built by
# data/build_dataset.py): hamlet 0, macbeth 1, midsummer 2, romeo 3.
LABELS = json.load(open(os.path.join(DATA, "labels.json")))
N_PLAYS = len(LABELS)


def load_quotes(split):
    """Load one split of the play-quotes dataset.

    split is "train", "val", or "test", loading
    data/play_quotes_train.jsonl, data/play_quotes_val.jsonl, or
    data/play_quotes_test.jsonl. Returns a list of {"text": ...,
    "label": int, "play": ...}. Same files Track 1 uses, so the
    comparison is apples-to-apples.
    """
    path = os.path.join(DATA, f"play_quotes_{split}.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="gpt2",
        help="pretrained model to fine-tune (gpt2 = 124M; "
        "distilgpt2 = 82M, smaller and faster on CPU)",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output",
        default="gpt2-play-classifier",
        help="directory to save the fine-tuned model",
    )
    args = parser.parse_args()

    # TODO (step 1): load the tokenizer and the model.
    #   tokenizer = AutoTokenizer.from_pretrained(args.model)
    #   tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token
    #   model = AutoModelForSequenceClassification.from_pretrained(
    #       args.model, num_labels=N_PLAYS
    #   )
    # AutoModelForSequenceClassification adds a classification head on
    # top of GPT-2 (a Linear layer, d_model -> num_labels) — the same
    # head you build by hand in Track 1, but here it comes with the
    # library. num_labels=4 tells it how many plays there are.

    # TODO (step 2): load and tokenize the dataset.
    #   train_rows = load_quotes("train")
    #   val_rows = load_quotes("val")
    #   test_rows = load_quotes("test")
    #   train_ds = Dataset.from_list(train_rows)
    #   val_ds = Dataset.from_list(val_rows)
    #   def tokenize(batch):
    #       return tokenizer(batch["text"], truncation=True, max_length=128)
    #   train_ds = train_ds.map(tokenize, batched=True)
    #   val_ds = val_ds.map(tokenize, batched=True)
    # Truncation at 128 tokens is plenty — the quotes are 1-3 lines.

    # TODO (step 3): define the training arguments and the Trainer.
    #   training_args = TrainingArguments(
    #       output_dir=args.output,
    #       num_train_epochs=args.epochs,
    #       learning_rate=args.lr,
    #       per_device_train_batch_size=args.batch_size,
    #       eval_strategy="epoch",
    #       save_strategy="no",
    #   )
    #   def compute_metrics(eval_pred):
    #       logits, labels = eval_pred
    #       preds = logits.argmax(axis=1)
    #       accuracy = (preds == labels).mean()
    #       return {"accuracy": accuracy}
    #   trainer = Trainer(
    #       model=model,
    #       args=training_args,
    #       train_dataset=train_ds,
    #       eval_dataset=val_ds,
    #       compute_metrics=compute_metrics,
    #   )
    # The Trainer is the library's training loop — it does the
    # predict -> loss -> backward -> update cycle for you, plus
    # evaluation each epoch. The compute_metrics function is where
    # accuracy is measured (same definition as Track 1: argmax == label).

    # TODO (step 4): train, evaluate on the test set, and save.
    #   trainer.train()
    #   test_ds = Dataset.from_list(test_rows).map(tokenize, batched=True)
    #   result = trainer.evaluate(test_ds)
    #   print(f"test accuracy: {result['eval_accuracy']:.3f}")
    #   trainer.save_model(args.output)
    #   tokenizer.save_pretrained(args.output)
    # The test accuracy is the number to compare against Track 1's
    # accuracy on the same test split. Chance is 0.25 (4 plays).

    raise NotImplementedError(
        "Track 2 skeleton: install torch, transformers, datasets, "
        "then fill in steps 1-4 above (uncomment the code blocks)."
    )


if __name__ == "__main__":
    main()
