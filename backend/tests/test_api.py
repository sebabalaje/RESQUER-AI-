from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_requires_consent():
    response = client.post("/analyze", json={"text": "I feel overwhelmed", "consent_confirmed": False})
    assert response.status_code == 400

def test_analysis_returns_limits_and_features():
    response = client.post("/analyze", json={"text": "I feel overwhelmed and need someone to talk to", "consent_confirmed": True})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "experimental_demo"
    assert "self_reference_count" in data["features"]
    assert 0 <= data["demo_model_score"] <= 1
    assert data["limitations"]
