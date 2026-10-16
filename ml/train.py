import argparse
import json
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.data_pipeline import load_dataset

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "ml" / "registry"
CONFIGS = ROOT / "ml" / "configs"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def build_pipeline(cfg: dict) -> Pipeline:
    tfidf_cfg = cfg["tfidf"].copy()
    if "ngram_range" in tfidf_cfg:
        tfidf_cfg["ngram_range"] = tuple(tfidf_cfg["ngram_range"])
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**tfidf_cfg)),
            ("clf", LogisticRegression(**cfg["clf"])),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-version", default="v1", choices=["v1", "v2"])
    parser.add_argument("--experiment", default="topic-classifier")
    args = parser.parse_args()

    cfg = yaml.safe_load((CONFIGS / args.config).read_text(encoding="utf-8"))
    ds = load_dataset(args.data_version)  # type: ignore[arg-type]

    x_train, x_test, y_train, y_test = train_test_split(
        ds.texts, ds.labels, test_size=0.2, random_state=42, stratify=ds.labels
    )

    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=cfg["name"]):
        mlflow.log_param("git_sha", git_sha())
        mlflow.log_param("data_version", args.data_version)
        mlflow.log_param("tfidf", json.dumps(cfg["tfidf"], sort_keys=True))
        mlflow.log_param("clf", json.dumps(cfg["clf"], sort_keys=True))

        pipe = build_pipeline(cfg)
        pipe.fit(x_train, y_train)

        preds = pipe.predict(x_test)
        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average="macro"))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        REGISTRY.mkdir(parents=True, exist_ok=True)
        model_path = REGISTRY / "model.joblib"
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(str(model_path))
        mlflow.sklearn.log_model(pipe, "model")

        print(f"accuracy={acc:.4f} f1_macro={f1:.4f}")


if __name__ == "__main__":
    main()