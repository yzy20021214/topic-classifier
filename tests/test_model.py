import pytest

from ml.data_pipeline import Dataset


def test_dataset_rejects_length_mismatch():
    with pytest.raises(ValueError):
        Dataset(texts=["a", "b"], labels=[0])


def test_dataset_rejects_empty():
    with pytest.raises(ValueError):
        Dataset(texts=[], labels=[])


def test_pipeline_predicts_probabilities(toy_model):
    proba = toy_model.predict_proba(["hockey goal", "mars rocket"])
    assert proba.shape == (2, 2)
    assert all(abs(row.sum() - 1.0) < 1e-6 for row in proba)


def test_pipeline_handles_unknown_words(toy_model):
    pred = toy_model.predict(["zzzzz qqqqq wwww"])
    assert pred.shape == (1,)
    assert pred[0] in (0, 1)
