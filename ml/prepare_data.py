"""Download the 20 newsgroups subset and write v1 / v2 raw text files.

v1: original text (headers/footers/quotes kept)
v2: cleaned text (headers/footers/quotes removed)
"""
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups

CATEGORIES = ["rec.sport.hockey", "sci.space"]
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"


def dump(version: str, remove: tuple[str, ...]) -> None:
    data = fetch_20newsgroups(
        subset="all",
        categories=CATEGORIES,
        remove=remove,
        shuffle=True,
        random_state=42,
    )
    out = DATA / version
    out.mkdir(parents=True, exist_ok=True)

    with (out / "texts.txt").open("w", encoding="utf-8") as f:
        for text in data.data:
            f.write(text.replace("\n", " ").strip() + "\n")

    with (out / "labels.txt").open("w", encoding="utf-8") as f:
        for target in data.target:
            f.write(str(int(target)) + "\n")

    print(f"wrote {len(data.data)} samples to {out}")


if __name__ == "__main__":
    dump("v1", ())
    dump("v2", ("headers", "footers", "quotes"))
