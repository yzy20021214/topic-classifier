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
    texts_path = base / "texts.txt"
    labels_path = base / "labels.txt"

    if not texts_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Dataset {version} not found at {base}. Run `python -m ml.prepare_data` first."
        )

    texts = [line.strip() for line in texts_path.read_text(encoding="utf-8").splitlines()]
    labels = [int(line.strip()) for line in labels_path.read_text(encoding="utf-8").splitlines()]

    return Dataset(texts=texts, labels=labels)
