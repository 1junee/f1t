"""검색 요청 하나의 실제 경과 시간을 기록한다."""
from contextlib import contextmanager
from threading import Lock
from time import perf_counter


class SearchTimings:
    def __init__(self, clock=perf_counter):
        self._clock = clock
        self._started = clock()
        self._stages = []
        self._lock = Lock()

    @contextmanager
    def stage(self, stage_id, label, parallel_group=None):
        started = self._clock()
        record = {"id": stage_id, "label": label}
        if parallel_group:
            record["parallel_group"] = parallel_group
        with self._lock:
            self._stages.append(record)
        try:
            yield
        finally:
            # 작업 스레드 안에서 재므로 Future.result()의 대기 시간이 섞이지 않는다.
            with self._lock:
                record["duration_ms"] = round((self._clock() - started) * 1000, 2)

    def run(self, stage_id, label, parallel_group, function, *args, **kwargs):
        with self.stage(stage_id, label, parallel_group):
            return function(*args, **kwargs)

    def snapshot(self):
        # 병렬 작업의 시간을 더하지 않고 요청 시작부터 실제로 흐른 시간을 기록한다.
        with self._lock:
            return {
                "total_ms": round((self._clock() - self._started) * 1000, 2),
                "stages": [dict(stage) for stage in self._stages],
            }
