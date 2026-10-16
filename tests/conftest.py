import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.main import app, get_model


def _train_toy_model() -> Pipeline:
    texts = [
        "the team scored a goal in the hockey match",
        "nhl playoffs are exciting this year",
        "rocket launch to mars was successful",
        "nasa discovered a new planet",
        "hockey player injured during game",
        "space telescope captured new images",
        "the puck went into the net",
        "astronauts completed a spacewalk",
        "goalie made an incredible save",
        "satellite entered orbit today",
    ]
    labels = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipe.fit(texts, labels)
    return pipe


@pytest.fixture
def toy_model() -> Pipeline:
    return _train_toy_model()


@pytest.fixture
def client(toy_model: Pipeline):
    app.dependency_overrides[get_model] = lambda: toy_model
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
