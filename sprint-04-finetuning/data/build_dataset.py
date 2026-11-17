"""Build the Sprint 4 play-classification dataset.

Parses the Complete Works of Shakespeare (Gutenberg #100, already in
sprint-03-transformer/data/shakespeare-full.txt) into short labeled
quotes: 1-3 consecutive lines spoken by one character, labeled by the
play they come from.

Why these four plays: they have distinct styles and vocabularies, so
the classes are separable but not trivially so. Hamlet is philosophical
and introspective; Macbeth is dark, violent, and full of prophecy;
Romeo and Juliet is romantic and lyrical; A Midsummer Night's Dream is
whimsical, with fairies and mechanicals. A classifier that can tell
them apart has learned something real about style, not just memorized
a few keywords.

How the parse works:
- Each play is a line range in shakespeare-full.txt. We skip everything
  before the first "ACT I" line (that drops the Contents and the
  Dramatis Personae, whose all-caps names would look like speakers).
- A speaker turn starts at a line that is all capital letters ending
  in a period ("HAMLET.", "FIRST WITCH.", "LADY MACBETH.") and runs
  until a blank line. Stage directions (lines starting with "[") and
  inline bracketed asides (e.g. "[_Aside._]") are stripped, not kept.
- Each turn is split into non-overlapping chunks of up to 3 lines.
  A chunk is one quote. Quotes shorter than 20 characters or 3 words
  are dropped (they carry no signal); quotes over 400 characters are
  dropped (they are usually prose blocks, not lines).
- The split is stratified: each play contributes 80% of its quotes to
  train, 10% to val, 10% to test, so every split is balanced across
  the four plays.

Output (in this data/ directory):
- play_quotes_train.jsonl, play_quotes_val.jsonl, play_quotes_test.jsonl
  One JSON object per line: {"text": ..., "label": int, "play": ...}.
- labels.json: the label index -> play name mapping.

Usage:
    python3 data/build_dataset.py
    python3 data/build_dataset.py --max-per-play 1500 --seed 1
"""

import argparse
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SHAKESPEARE = os.path.join(
    HERE, "..", "..", "sprint-03-transformer", "data", "shakespeare-full.txt"
)

# Play name -> (first line, last line) in shakespeare-full.txt, 1-based,
# inclusive. The end of each play is the line before the next play
# begins (verified against the Contents order in the file).
PLAYS = {
    "hamlet": (34409, 41106),
    "macbeth": (95993, 100142),
    "midsummer": (114021, 117505),
    "romeo": (143343, 148609),
}

# Label index, alphabetical by play name. Saved to labels.json so the
# training scripts and the reader never have to guess.
LABELS = {name: i for i, name in enumerate(sorted(PLAYS))}

# A bracketed stage direction, inline or whole-line: "[_Aside._]", "[Exeunt.]"
BRACKET_RE = re.compile(r"\[[^\]]*\]")
# Unbracketed stage directions: short lines that describe action, not
# speech. They appear as their own line in the dialogue ("Alarums.
# Enter Macbeth.", "Flourish.", "Thunder."). They are not spoken, so
# they must not become quotes.
STAGE_RE = re.compile(
    r"^(Enter|Exit|Exeunt|Alarums|Flourish|Thunder|Sennet|Hautboys"
    r"|Music|Drums?|Trumpets?|Re-?enter|Dies|Sleeps|Sings?|Dances?)\b"
)


def is_speaker(line):
    """True if the line is a speaker label.

    A speaker label is short, ends with a period, and is all capitals:
    "HAMLET.", "FIRST WITCH.", "LADY MACBETH.". Joint labels ("HORATIO
    and MARCELLUS.") are also speakers — the word "and" is allowed in
    lowercase. Ordinary dialogue is mixed case, so it never matches.
    """
    line = line.strip()
    if len(line) > 30 or len(line) < 2:
        return False
    if not line.endswith("."):
        return False
    words = line[:-1].replace(",", "").split()
    if not words:
        return False
    return all(w == w.upper() or w.lower() == "and" for w in words)


def clean_dialogue(line):
    """Strip stage directions and extra whitespace from a dialogue line."""
    line = BRACKET_RE.sub("", line)
    return " ".join(line.split())


def extract_turns(lines):
    """Split a play's lines into speaker turns.

    Returns a list of turns; each turn is a list of cleaned dialogue
    lines (speaker labels and stage directions removed). Turns are
    split on blank lines. Only lines after the Dramatis Personae and
    the first "ACT I" are used (the Contents list also contains
    "ACT I", so the Personae gate is what keeps the character list
    out). Act/scene headers inside the play are skipped.
    """
    started = False
    seen_personae = False
    turns = []
    current = []
    for raw in lines:
        line = raw.strip()
        if not started:
            # The Contents list also contains "ACT I", before the
            # Dramatis Personae — so only start after the Personae has
            # passed AND then an "ACT I" appears. Otherwise the
            # character list (all-caps names) leaks in as dialogue.
            if line == "Dramatis Personae":
                seen_personae = True
            elif seen_personae and line.startswith("ACT I"):
                started = True
            continue
        if line.startswith("ACT ") or line.startswith("SCENE") or line.startswith("Scene"):
            # Act / scene headers inside the play, not dialogue.
            if current:
                turns.append(current)
            current = []
            continue
        if is_speaker(line):
            # A new speaker ends the previous turn.
            if current:
                turns.append(current)
            current = []
            continue
        if not line:
            if current:
                turns.append(current)
            current = []
            continue
        if line.startswith("["):
            continue
        if STAGE_RE.match(line):
            continue
        cleaned = clean_dialogue(line)
        if cleaned:
            current.append(cleaned)
    if current:
        turns.append(current)
    return turns


def quotes_from_turns(turns, max_lines=3, min_chars=20, max_chars=400):
    """Split turns into non-overlapping quotes of up to max_lines lines."""
    quotes = []
    for turn in turns:
        for i in range(0, len(turn), max_lines):
            chunk = turn[i : i + max_lines]
            text = " ".join(chunk)
            words = text.split()
            if len(text) < min_chars or len(text) > max_chars:
                continue
            if len(words) < 3:
                continue
            if not any(c.islower() for c in text):
                continue
            quotes.append(text)
    return quotes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-per-play",
        type=int,
        default=1500,
        help="cap quotes per play (keeps the classes balanced)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--shakespeare",
        type=str,
        default=SHAKESPEARE,
        help="path to shakespeare-full.txt",
    )
    args = parser.parse_args()

    all_lines = open(args.shakespeare).read().splitlines()
    rng = random.Random(args.seed)

    per_play = {}
    for play, (start, end) in PLAYS.items():
        chunk = all_lines[start - 1 : end]
        turns = extract_turns(chunk)
        quotes = quotes_from_turns(turns)
        rng.shuffle(quotes)
        quotes = quotes[: args.max_per_play]
        per_play[play] = quotes
        print(f"{play}: {len(turns)} turns -> {len(quotes)} quotes")

    # Stratified split: 80/10/10 within each play, then shuffle.
    splits = {"train": [], "val": [], "test": []}
    for play, quotes in per_play.items():
        n = len(quotes)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        for i, text in enumerate(quotes):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            splits[split].append(
                {"text": text, "label": LABELS[play], "play": play}
            )
    for split, rows in splits.items():
        rng.shuffle(rows)
        path = os.path.join(HERE, f"play_quotes_{split}.jsonl")
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        counts = {}
        for row in rows:
            counts[row["play"]] = counts.get(row["play"], 0) + 1
        print(f"{split}: {len(rows)} rows {counts} -> {path}")

    with open(os.path.join(HERE, "labels.json"), "w") as f:
        json.dump(LABELS, f, indent=2)
    print(f"labels: {LABELS}")

    print("\nsamples:")
    for row in splits["train"][:4]:
        print(f"  [{row['play']}] {row['text'][:110]}")


if __name__ == "__main__":
    main()
