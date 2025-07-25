import requests

def test_api_predict():
    url = "http://127.0.0.1:8000/predict"
    sample = {"features": [0.5, 1.2, -0.3, 0.8, 1.5]}  # Example scaled features
    response = requests.post(url, json=sample)
    assert response.status_code == 200
    assert "prediction" in response.json()
