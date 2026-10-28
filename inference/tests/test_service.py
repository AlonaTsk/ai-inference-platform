from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.data import LABELS, generate
from app.train import build_pipeline


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory) -> Path:
    import joblib

    texts, labels = generate(n=400, seed=1)
    model = build_pipeline().fit(texts, labels)
    d = tmp_path_factory.mktemp("model")
    joblib.dump(model, d / "model.joblib")
    return d


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setattr(main, "MODEL_DIR", model_dir)
    with TestClient(main.app) as c:
        yield c


def test_dataset_is_deterministic():
    assert generate(n=50, seed=7) == generate(n=50, seed=7)


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_ready_when_model_loaded(client):
    assert client.get("/readyz").status_code == 200


def test_not_ready_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "MODEL_DIR", tmp_path)
    main.state["model"] = None
    with TestClient(main.app) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/readyz").status_code == 503
        assert c.post("/predict", json={"texts": ["hi"]}).status_code == 503


def test_predict_returns_one_prediction_per_text(client):
    resp = client.post("/predict", json={"texts": [
        "I was charged twice this month",
        "the app crashes when I open settings",
    ]})
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert [p["label"] for p in preds] == ["billing", "technical"]
    for p in preds:
        assert set(p["scores"]) == set(LABELS)
        assert abs(sum(p["scores"].values()) - 1) < 1e-3


@pytest.mark.parametrize("body", [
    {},
    {"texts": []},
    {"texts": ["   "]},
    {"texts": ["x" * 2001]},
    {"texts": ["ok"] * 65},
])
def test_predict_rejects_invalid_input(client, body):
    assert client.post("/predict", json=body).status_code == 422
