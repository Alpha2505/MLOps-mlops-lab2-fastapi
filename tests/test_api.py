def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_type"] == "RandomForestClassifier"
    assert len(body["classes"]) == 3
    assert len(body["features"]) == 13
    assert 0 <= body["test_accuracy"] <= 1


def test_predict_valid_sample(client, sample):
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    body = r.json()
    assert body["class_id"] == 0  # first row of the dataset is class_0
    assert 0 <= body["confidence"] <= 1


def test_predict_missing_field_returns_422(client, sample):
    sample.pop("proline")
    assert client.post("/predict", json=sample).status_code == 422


def test_predict_negative_value_returns_422(client, sample):
    sample["alcohol"] = -1.0
    assert client.post("/predict", json=sample).status_code == 422


def test_predict_string_number_is_converted(client, sample):
    sample["alcohol"] = "14.23"  # Pydantic converts this to a float
    assert client.post("/predict", json=sample).status_code == 200


def test_batch_returns_one_prediction_per_sample(client, sample):
    r = client.post("/predict/batch", json={"samples": [sample, sample, sample]})
    assert r.status_code == 200
    assert r.json()["count"] == 3


def test_batch_empty_returns_400(client):
    r = client.post("/predict/batch", json={"samples": []})
    assert r.status_code == 400


def test_batch_too_large_returns_400(client, sample):
    r = client.post("/predict/batch", json={"samples": [sample] * 51})
    assert r.status_code == 400
