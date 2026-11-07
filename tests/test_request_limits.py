from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi import HTTPException

from backend.request_limits import RequestLimiter


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


def test_minute_window_expires_and_denials_do_not_consume_quota():
    clock = Clock()
    limiter = RequestLimiter(max_concurrent=1, per_minute=1, per_day=2, clock=clock)
    with limiter.acquire():
        pass
    clock.value = 59.2
    with pytest.raises(HTTPException) as error:
        with limiter.acquire():
            pytest.fail('A rejected request must not execute')
    assert error.value.status_code == 429
    assert error.value.headers == {'Retry-After': '1'}
    clock.value = 60
    with limiter.acquire():
        pass


def test_daily_window_expires_independently_of_minute_window():
    clock = Clock()
    limiter = RequestLimiter(max_concurrent=1, per_minute=10, per_day=1, clock=clock)
    with limiter.acquire():
        pass
    clock.value = 60
    with pytest.raises(HTTPException) as error:
        with limiter.acquire():
            pass
    assert error.value.headers == {'Retry-After': '86340'}
    clock.value = 86400
    with limiter.acquire():
        pass


def test_failure_releases_slot_but_still_consumes_request_quota():
    limiter = RequestLimiter(max_concurrent=1, per_minute=2, per_day=10)
    with pytest.raises(RuntimeError):
        with limiter.acquire():
            raise RuntimeError('provider failed after request started')
    with limiter.acquire():
        pass
    with pytest.raises(HTTPException):
        with limiter.acquire():
            pass


def test_concurrent_admission_is_atomic_and_releases_slot():
    limiter = RequestLimiter(max_concurrent=2, per_minute=100, per_day=100)
    start = Barrier(8)
    finish = Barrier(8)

    def enter():
        start.wait(timeout=5)
        try:
            with limiter.acquire():
                finish.wait(timeout=5)
                return True
        except HTTPException as error:
            assert error.status_code == 429
            assert int(error.headers['Retry-After']) >= 1
            finish.wait(timeout=5)
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(lambda _: enter(), range(8))) == 2
    with limiter.acquire():
        pass


@pytest.mark.parametrize('raw', ['0', '-1', 'bad', ''])
def test_invalid_environment_limits_use_bounded_defaults(monkeypatch, raw):
    for key in ['MODEL_MAX_CONCURRENT_REQUESTS', 'MODEL_REQUESTS_PER_MINUTE', 'MODEL_REQUESTS_PER_DAY']:
        monkeypatch.setenv(key, raw)
    limiter = RequestLimiter.from_env()
    assert (limiter.max_concurrent, limiter.per_minute, limiter.per_day) == (2, 10, 100)


def test_valid_environment_limits(monkeypatch):
    monkeypatch.setenv('MODEL_MAX_CONCURRENT_REQUESTS', '3')
    monkeypatch.setenv('MODEL_REQUESTS_PER_MINUTE', '20')
    monkeypatch.setenv('MODEL_REQUESTS_PER_DAY', '200')
    limiter = RequestLimiter.from_env()
    assert (limiter.max_concurrent, limiter.per_minute, limiter.per_day) == (3, 20, 200)
