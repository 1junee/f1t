from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from pipeline.timing import SearchTimings


def test_overlapping_stages_use_wall_time_and_do_not_leak_between_requests():
    now = [0.0]
    timing = SearchTimings(clock=lambda: now[0])
    with timing.stage("a", "첫 작업", "analysis"):
        now[0] = 1.0
        with timing.stage("b", "겹치는 작업", "analysis"):
            now[0] = 4.0
        now[0] = 5.0
    result = timing.snapshot()
    assert result["total_ms"] == 5000
    assert [s["duration_ms"] for s in result["stages"]] == [5000, 3000]
    assert sum(s["duration_ms"] for s in result["stages"]) > result["total_ms"]
    assert SearchTimings(clock=lambda: now[0]).snapshot() == {"total_ms": 0, "stages": []}
    result["stages"][0]["duration_ms"] = -1
    assert timing.snapshot()["stages"][0]["duration_ms"] == 5000


def test_parallel_wrapper_preserves_results_and_records_every_task():
    timing = SearchTimings()
    barrier = Barrier(2)

    def work(value):
        barrier.wait(timeout=2)
        return value

    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(timing.run, str(i), "작업", "encoding", work, i) for i in range(2)]
        assert [job.result(timeout=3) for job in jobs] == [0, 1]
    stages = timing.snapshot()["stages"]
    assert {stage["id"] for stage in stages} == {"0", "1"}
    assert all(stage["duration_ms"] >= 0 for stage in stages)


def test_failed_task_keeps_elapsed_time_and_original_exception():
    now = [0.0]
    timing = SearchTimings(clock=lambda: now[0])
    with pytest.raises(ValueError, match="failure"):
        with timing.stage("failed", "실패한 작업"):
            now[0] = 2.0
            raise ValueError("failure")
    assert timing.snapshot()["stages"][0]["duration_ms"] == 2000
