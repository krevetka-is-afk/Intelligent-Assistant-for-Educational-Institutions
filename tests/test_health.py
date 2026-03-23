def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metrics_endpoint_available(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "rag_requests_total" in r.text
