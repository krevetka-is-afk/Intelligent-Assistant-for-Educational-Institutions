def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metrics_endpoint_requires_api_key(client):
    r = client.get("/metrics")
    assert r.status_code == 401


def test_metrics_endpoint_available_with_api_key(client, auth_headers):
    r = client.get("/metrics", headers=auth_headers)
    assert r.status_code == 200
    assert "rag_requests_total" in r.text
