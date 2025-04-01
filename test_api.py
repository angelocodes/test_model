import pytest
from app import app 

@pytest.fixture
def client():
    app.testing = True  # Enable testing mode
    client = app.test_client()
    return client

def test_predict_positive(client):
    response = client.post("/predict", json={"text": "I love this movie!"})
    assert response.status_code == 200
    data = response.get_json()
    assert "sentiment" in data
    assert data["sentiment"] == "positive"

def test_predict_negative(client):
    response = client.post("/predict", json={"text": "This is the worst movie ever!"})
    assert response.status_code == 200
    data = response.get_json()
    assert "sentiment" in data
    assert data["sentiment"] == "negative"