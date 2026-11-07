"""One-process request admission limits for the public demo model endpoints."""
from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from contextlib import contextmanager
from threading import Lock
from typing import Callable, Iterator

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def _positive_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        if value > 0:
            return value
    except ValueError:
        pass
    logger.warning("Invalid %s; using default %s", name, default)
    return default


class RequestLimiter:
    # 사용자/IP별 한도가 아니라, 이 프로세스의 모델 요청 전체가 공유하는 한도다.
    # 단조 증가 시계로 최근 60초/24시간을 계산하므로 날짜 변경·시계 보정에 영향받지 않는다.
    def __init__(
        self,
        *,
        max_concurrent: int = 2,
        per_minute: int = 10,
        per_day: int = 100,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if min(max_concurrent, per_minute, per_day) < 1:
            raise ValueError("Request limits must be positive")
        self.max_concurrent = max_concurrent
        self.per_minute = per_minute
        self.per_day = per_day
        self._clock = clock
        self._minute: deque[float] = deque()
        self._day: deque[float] = deque()
        self._active = 0
        self._lock = Lock()

    @classmethod
    def from_env(cls) -> RequestLimiter:
        return cls(
            max_concurrent=_positive_env("MODEL_MAX_CONCURRENT_REQUESTS", 2),
            per_minute=_positive_env("MODEL_REQUESTS_PER_MINUTE", 10),
            per_day=_positive_env("MODEL_REQUESTS_PER_DAY", 100),
        )

    @contextmanager
    def acquire(self) -> Iterator[None]:
        # 검사와 사용량 기록을 같은 lock 안에서 처리해 동시 요청의 한도 초과를 막는다.
        with self._lock:
            now = self._clock()
            for events, window in ((self._minute, 60), (self._day, 86400)):
                while events and events[0] <= now - window:
                    events.popleft()
            retries = []
            if len(self._minute) >= self.per_minute:
                retries.append(self._minute[0] + 60 - now)
            if len(self._day) >= self.per_day:
                retries.append(self._day[0] + 86400 - now)
            if self._active >= self.max_concurrent:
                retries.append(1)  # 실제 종료 시각은 알 수 없어 최소 재시도 간격을 안내한다.
            if retries:
                raise HTTPException(
                    status_code=429,
                    detail="Model request limit reached. Please try again later.",
                    headers={"Retry-After": str(max(1, math.ceil(max(retries))))},
                )
            self._minute.append(now)
            self._day.append(now)
            self._active += 1
        try:
            yield
        finally:
            # 실패해도 이미 외부 호출이 과금됐을 수 있어 요청 수는 차감하지 않는다.
            # 실행 슬롯만 반환하며, 모델 호출 중에는 lock을 잡고 있지 않는다.
            with self._lock:
                self._active -= 1
