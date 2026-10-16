"""Download the 20 newsgroups subset and write v1 / v2 as JSON files.

v1: original text (headers/footers/quotes kept)
v2: cleaned text (headers/footers/quotes removed)
"""
import json
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

    records = [
        {"text": str(text).replace("\n", " ").strip(), "label": int(target)}
        for text, target in zip(data.data, data.target)
    ]

    with (out / "data.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"wrote {len(records)} samples to {out / 'data.json'}")


if __name__ == "__main__":
    dump("v1", ())
    dump("v2", ("headers", "footers", "quotes"))