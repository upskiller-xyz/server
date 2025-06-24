import pytest
from ..processing.processor import MultiThreader

# from concurrent.futures import


def dummy_func(x):
    return x * 2


def test_estimate_workers(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    n = MultiThreader.estimate_workers(3)
    assert n == 3
    n = MultiThreader.estimate_workers(100)
    assert n == 40  # 8*5


def test_run_and__run(monkeypatch):
    # Patch ThreadPoolExecutor to run synchronously for test
    class DummyFuture:
        def __init__(self, val):
            self._val = val

        def result(self):
            return self._val

    class DummyExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def submit(self, func, i):
            return DummyFuture(func(i))

    monkeypatch.setattr("server.processing.processor.ThreadPoolExecutor", DummyExecutor)
    monkeypatch.setattr("os.cpu_count", lambda: 2)
    inp = [1, 2, 3]
    results = MultiThreader._run(dummy_func, inp)
    assert [f.result() for f in results] == [2, 4, 6]
    # Patch as_completed to just yield the dummy futures in order
    monkeypatch.setattr("concurrent.futures.as_completed", lambda fs: iter(fs))
    results_iter = MultiThreader.run(dummy_func, inp)
    assert list(results_iter) == [2, 4, 6]
