# Experiments

Two MLflow runs are tracked under experiment `topic-classifier`:

1. **baseline** — `ml/configs/baseline.yaml`, data v1
   - TF-IDF (1-gram, 5k features) + LogisticRegression(C=1.0)
2. **improved** — `ml/configs/improved.yaml`, data v2
   - TF-IDF (1-2 gram, 20k features, min_df=2, sublinear_tf) + LogisticRegression(C=5.0)

## Production-worthy run

`improved` is selected as production because it uses the cleaned v2 dataset
and a richer feature space, which raises macro-F1 on the held-out test set.

## Optimized metric

**Macro-F1**. The two classes are roughly balanced, but macro-F1 punishes
per-class failures equally, which matters for an API that must serve both
topics reliably.

## How to reproduce

```bash
python -m ml.train --config baseline.yaml --data-version v1
python -m ml.train --config improved.yaml --data-version v2
mlflow ui