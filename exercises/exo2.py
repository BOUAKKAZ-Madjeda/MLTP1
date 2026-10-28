from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_20newsgroups


# exo2.py is in the exercises folder.
# The data folder is its sibling under the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
txt_path = PROJECT_ROOT / "data" / "txt_file.txt"

# 1–2. Read the text file as UTF-8 and print each line,
# removing only its newline character.
with open(txt_path, "r", encoding="utf-8") as f:
    for line in f:
        print(line.rstrip("\n"))


# 3–6. Load the 20 Newsgroups training data.
# The first run may need an internet connection to download it.
try:
    news = fetch_20newsgroups(subset="train")

    print("\nNumber of documents:", len(news.data))
    print("Number of labels (target values):", len(news.target))
    print("Number of classes:", len(news.target_names))

    if news.data:
        print("\nFirst document (first 300 characters):\n", news.data[0][:300])
        target = news.target[0]
        print(
            "\nIts target value:",
            target,
            "->",
            news.target_names[target],
        )

    print("\nClass names:")
    for code, name in enumerate(news.target_names):
        print(f"  {code:>2} -> {name}")

    # 7. Optional: document lengths and frequent words.
    lengths = pd.Series([len(document.split()) for document in news.data])
    print("\nDocument length (words):\n", lengths.describe().round(1))

    words = pd.Series(" ".join(news.data[:500]).lower().split())
    print("\nCounts of a few words (first 500 documents):")
    for word in ["the", "of", "and", "computer", "car"]:
        print(f"  {word:<10}{(words == word).sum()}")

except Exception as e:
    print("\n[fetch_20newsgroups failed - no internet or no local copy]")
    print("Reason:", type(e).__name__, "-", e)


print(
    """
EXPLANATION (Ex.2): news.data is a list of raw texts (n_documents).
news.target is an integer array with one label per document (same length).
target_names[k] gives the class name of integer k, e.g. target 7 -> 'rec.autos'.
"""
)