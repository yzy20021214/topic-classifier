def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_hockey_text(client):
    r = client.post("/predict", json={"text": "the hockey team scored a goal"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"hockey", "space"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["label"] == "hockey"


def test_predict_space_text(client):
    r = client.post("/predict", json={"text": "nasa launched a rocket to mars"})
    assert r.status_code == 200
    assert r.json()["label"] == "space"


def test_predict_empty_string_returns_422(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_whitespace_only_returns_422(client):
    r = client.post("/predict", json={"text": "    "})
    assert r.status_code == 422


def test_predict_missing_field_returns_422(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_too_long_text_returns_422(client):
    r = client.post("/predict", json={"text": "a" * 6000})
    assert r.status_code == 422
