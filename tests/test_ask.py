def test_ask(client):
    response = client.post("/ask", json={"question": "Hello world"})
    assert response.status_code == 200
