import pytest

from ml.data_pipeline import Dataset, load_dataset


def test_load_dataset_missing_version_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset("v99")


def test_dataset_dataclass_ok():
    ds = Dataset(texts=["a", "b"], labels=[0, 1])
    assert len(ds.texts) == 2
    assert ds.labels == [0, 1]


def test_dataset_labels_must_be_int_like():
    ds = Dataset(texts=["x"], labels=[1])
    assert isinstance(ds.labels[0], int)
