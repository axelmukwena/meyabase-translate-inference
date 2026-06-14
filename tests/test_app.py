from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "en-ng" in body["directions"]


def test_translate_ok():
    resp = client.post("/translate", json={"text": "Hello", "direction": "en-ng"})
    assert resp.status_code == 200
    assert resp.json() == {"translation": "<meyabase/en-ng-translation>::Hello"}


def test_translate_bad_direction_returns_400():
    resp = client.post("/translate", json={"text": "Hello", "direction": "zz-yy"})
    assert resp.status_code == 400
    assert "Unknown direction" in resp.json()["detail"]


def test_translate_missing_text_returns_422():
    resp = client.post("/translate", json={"direction": "en-ng"})
    assert resp.status_code == 422  # pydantic validation
