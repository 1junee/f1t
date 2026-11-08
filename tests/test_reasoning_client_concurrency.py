"""외부 모델 없이 캐시·모델 초기화의 동시 접근을 검증한다."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

from pipeline import recommendation_service as service
from pipeline.target_description import target_description_generation as generation


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(service, 'PIPELINE_MODE', 'supabase_vector')
    monkeypatch.setattr(service, 'RetrieveModule', Mock())
    monkeypatch.setattr(service, 'SupabaseVectorSearchClient', Mock())
    return service.FashionRecommendationPipeline()


@pytest.mark.parametrize('provider,constructor', [
    ('gemini', 'GeminiReasoningClient'), ('openai', 'OpenAIReasoningClient'),
    ('qwen', 'Qwen3VLReasoningClient'),
])
def test_concurrent_first_requests_share_one_client(monkeypatch, pipeline, provider, constructor):
    start = Barrier(8)
    constructing = Event()
    release = Event()
    client = object()

    def construct():
        constructing.set()
        assert release.wait(5)
        return client

    factory = Mock(side_effect=construct)
    monkeypatch.setattr(service, constructor, factory)

    def get_client():
        start.wait(timeout=5)
        return pipeline._get_reasoning_client(provider)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(get_client) for _ in range(8)]
        try:
            assert constructing.wait(5)
        finally:
            release.set()
        assert all(future.result(timeout=5) is client for future in futures)
    factory.assert_called_once()
    assert not pipeline._client_cache_lock.locked()


def test_initialization_failure_releases_lock_and_can_retry(monkeypatch, pipeline):
    client = object()
    factory = Mock(side_effect=[RuntimeError('init failed'), client])
    monkeypatch.setattr(service, 'GeminiReasoningClient', factory)
    with pytest.raises(RuntimeError, match='init failed'):
        pipeline._get_reasoning_client('gemini')
    assert 'gemini' not in pipeline._client_cache
    assert not pipeline._client_cache_lock.locked()
    assert pipeline._get_reasoning_client('gemini') is client
    assert factory.call_count == 2


def test_inference_runs_outside_cache_lock(monkeypatch, pipeline):
    both_running = Barrier(2)

    def run(**kwargs):
        assert not pipeline._client_cache_lock.locked()
        both_running.wait(timeout=5)
        return {'Target Image Description': 'shirt'}

    monkeypatch.setattr(service, 'GeminiReasoningClient', lambda: SimpleNamespace(run=run))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(
            pipeline._run_target_description_reasoning,
            query='shirt', raw_fallback='shirt', image_bytes=None,
            image_mime_type=None, provider='gemini',
        ) for _ in range(2)]
        assert all(f.result(timeout=5)[2] == 'ok' for f in futures)


@pytest.fixture
def fake_qwen_dependencies(monkeypatch):
    processor = object()
    model = SimpleNamespace(eval=Mock())
    processor_factory = Mock(return_value=processor)
    model_factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False), float32='float32', bfloat16='bfloat16',
    ))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=Mock(return_value='/fake/model')))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=processor_factory),
    ))
    client = generation.Qwen3VLReasoningClient(device='cpu')
    monkeypatch.setattr(client, '_resolve_model_class', lambda path: SimpleNamespace(from_pretrained=model_factory))
    return client, processor_factory, model_factory, model


def test_concurrent_qwen_loads_model_once(fake_qwen_dependencies):
    client, processor_factory, model_factory, model = fake_qwen_dependencies
    start = Barrier(8)
    loading = Event()
    release = Event()

    def load_model(*args, **kwargs):
        loading.set()
        assert release.wait(5)
        return model

    model_factory.side_effect = load_model

    def load():
        start.wait(timeout=5)
        client._load()
        return client.model

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(load) for _ in range(8)]
        try:
            assert loading.wait(5)
        finally:
            release.set()
        assert all(f.result(timeout=5) is model for f in futures)
    processor_factory.assert_called_once()
    model_factory.assert_called_once()
    model.eval.assert_called_once()
    assert not client._load_lock.locked()


def test_qwen_failed_load_clears_partial_state_and_allows_retry(fake_qwen_dependencies):
    client, _, model_factory, model = fake_qwen_dependencies
    model_factory.side_effect = [RuntimeError('load failed'), model]
    with pytest.raises(generation.ReasoningProviderError):
        client._load()
    assert client.model is None and client.processor is None
    assert not client._load_lock.locked()
    client._load()
    assert client.model is model
