from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import api
from backend.request_limits import RequestLimiter


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, 'request_limiter', RequestLimiter(max_concurrent=1, per_minute=1, per_day=10))
    monkeypatch.setattr(api.app.state, 'search_service', SimpleNamespace(search=lambda **kwargs: ('shirt', {'provider': 'gemini'}, [])), raising=False)
    monkeypatch.setattr(api, 'extract_conservative_metadata', lambda **kwargs: SimpleNamespace(attributes=SimpleNamespace(to_dict=lambda: {})))
    return TestClient(api.app)


def test_search_and_analyze_share_quota_and_return_retry_after(client):
    assert client.post('/search', data={'query': 'shirt'}).status_code == 200
    response = client.post('/analyze', data={'text_input': 'shirt'})
    assert response.status_code == 429
    assert int(response.headers['Retry-After']) >= 1


def test_analyze_consumes_shared_quota_before_search(client):
    assert client.post('/analyze', data={'text_input': 'shirt'}).status_code == 200
    response = client.post('/search', data={'query': 'shirt'})
    assert response.status_code == 429


def test_invalid_provider_rejected_before_model_and_without_quota(client):
    assert client.post('/search', data={'query': 'shirt', 'provider': 'unknown'}).status_code == 400
    assert client.post('/search', data={'query': 'shirt', 'provider': 'gemini'}).status_code == 200


def test_status_does_not_invoke_embedding_or_consume_quota(client, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('Configuration status must not call the model')
    monkeypatch.setattr(api.app.state.search_service, 'retriever', SimpleNamespace(encode_text=fail), raising=False)
    response = client.get('/status')
    assert response.status_code == 200
    assert response.json()['embedding'] == 'not_checked'
    assert response.json()['check_mode'] == 'configuration_only'
    assert client.post('/search', data={'query': 'shirt'}).status_code == 200


def test_search_preserves_http_exceptions_and_releases_slot(client, monkeypatch):
    from fastapi import HTTPException
    def fail(**kwargs):
        raise HTTPException(status_code=429, detail='upstream limit', headers={'Retry-After': '7'})
    monkeypatch.setattr(api, 'request_limiter', RequestLimiter(max_concurrent=1, per_minute=10, per_day=10))
    monkeypatch.setattr(api.app.state.search_service, 'search', fail)
    response = client.post('/search', data={'query': 'shirt'})
    assert response.status_code == 429
    assert response.headers['Retry-After'] == '7'
    assert client.post('/analyze', data={'text_input': 'shirt'}).status_code == 200


def test_search_returns_safe_fashion_input_guidance(client, monkeypatch):
    from pipeline.target_description.target_description_generation import InvalidSearchInput

    def fail(**kwargs):
        raise InvalidSearchInput('옷이 정확히 보이는 사진을 첨부하세요.')

    monkeypatch.setattr(api.app.state.search_service, 'search', fail)
    response = client.post('/search', data={'query': 'shirt'})
    assert response.status_code == 400
    assert response.json()['detail'] == '옷이 정확히 보이는 사진을 첨부하세요.'
