from collections.abc import Iterator
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from app.data import LABELS, generate
from app.main import app
from app.train import build_pipeline


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    texts, labels = generate(n=400, seed=1)
    model = build_pipeline().fit(texts, labels)
    d = tmp_path_factory.mktemp("model")
    joblib.dump(model, d / "model.joblib")
    return d


@pytest.fixture
def client(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    with TestClient(app) as c:
        yield c


def test_dataset_is_deterministic() -> None:
    assert generate(n=50, seed=7) == generate(n=50, seed=7)


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_ready_when_model_loaded(client: TestClient) -> None:
    assert client.get("/readyz").status_code == 200


def test_metadata_404_without_metrics_file(client: TestClient) -> None:
    assert client.get("/metadata").status_code == 404


def test_not_ready_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/readyz").status_code == 503
        assert c.post("/predict", json={"texts": ["hi"]}).status_code == 503


def test_predict_returns_one_prediction_per_text(client: TestClient) -> None:
    resp = client.post(
        "/predict",
        json={"texts": ["I was charged twice this month", "the app crashes when I open settings"]},
    )
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert [p["label"] for p in preds] == ["billing", "technical"]
    for p in preds:
        assert set(p["scores"]) == set(LABELS)
        assert sum(p["scores"].values()) == pytest.approx(1, abs=1e-3)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"texts": []},
        {"texts": ["   "]},
        {"texts": ["x" * 2001]},
        {"texts": ["ok"] * 65},
    ],
    ids=["missing", "empty-list", "blank", "too-long", "batch-too-big"],
)
def test_predict_rejects_invalid_input(client: TestClient, body: dict) -> None:
    assert client.post("/predict", json=body).status_code == 422
