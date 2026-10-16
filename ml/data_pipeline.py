import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"

DataVersion = Literal["v1", "v2"]


@dataclass
class Dataset:
    texts: list[str]
    labels: list[int]

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.labels):
            raise ValueError("texts and labels length mismatch")
        if not self.texts:
            raise ValueError("dataset is empty")


def load_dataset(version: DataVersion) -> Dataset:
    base = DATA / version
    json_path = base / "data.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"Dataset {version} not found at {json_path}. Run `python -m ml.prepare_data` first."
        )

    records = json.loads(json_path.read_text(encoding="utf-8"))
    texts = [str(r["text"]) for r in records]
    labels = [int(r["label"]) for r in records]

    return Dataset(texts=texts, labels=labels)
