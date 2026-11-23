"""Scaffold: a retrieval benchmark cut out of WikiText-103.

WikiText marks article titles as " = Title = " and sections as
" = = Section = = ". Documents are paragraphs. Each query is a section,
written as "<article title> <section heading>" (e.g. "Valkyria Chronicles
III Gameplay"), and its relevant documents are that section's paragraphs.

It's a lexical-friendly benchmark (the article title usually appears in
the paragraphs), so expect BM25 to start strong. The `--headings-only`
variant drops the article title from the query, which is much harder for
keyword search.
"""

import re

TITLE = re.compile(r"^ = ([^=].*?) = $")
SECTION = re.compile(r"^ = = ([^=].*?) = = $")


def clean(text):
    """Undo WikiText's tokenization quirks for display and embedding."""
    for a, b in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" ,", ","), (" .", "."),
                 (" 's", "'s"), (" ;", ";"), (" :", ":"), ("( ", "("), (" )", ")")):
        text = text.replace(a, b)
    return text.strip()


def load(path, n_articles=500, min_words=30, headings_only=False, skip_articles=0):
    """Return (docs, queries).

    docs:    list of (doc_id, article_title, paragraph_text)
    queries: list of (query_text, [relevant doc_ids]) for sections with at
             least one paragraph of min_words words"""
    docs, queries = [], []
    article, section, current, seen = None, None, [], 0

    def close_section():
        if article and section and current:
            q = section if headings_only else f"{article} {section}"
            queries.append((q, list(current)))
            current.clear()

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = TITLE.match(line)
            if m:
                close_section()
                seen += 1
                if seen > skip_articles + n_articles:
                    break
                article = m.group(1).strip() if seen > skip_articles else None
                section, current = None, []
                continue
            if article is None:
                continue
            m = SECTION.match(line)
            if m:
                close_section()
                section, current = m.group(1).strip(), []
                continue
            text = clean(line)
            if len(text.split()) >= min_words:
                docs.append((len(docs), article, text))
                if section:
                    current.append(len(docs) - 1)
    close_section()
    return docs, queries
