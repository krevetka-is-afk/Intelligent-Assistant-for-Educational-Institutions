def test_ask(client, auth_headers):
    response = client.post("/ask", json={"question": "Hello world"}, headers=auth_headers)
    assert response.status_code == 200
