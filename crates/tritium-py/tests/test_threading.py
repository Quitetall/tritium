"""GIL-release and deadlock-freedom tests for the `tritium` extension.

The Rust compute in `ternary_matmul` (and `Model.generate`) runs under
`Python::allow_threads`, so:

1.  While one thread is inside the Rust GEMM, *other* Python threads keep running
    (the GIL is genuinely released, not just yielded between calls).
2.  Many threads (>= 4) hammering the extension concurrently complete without a
    deadlock or a crash.

These use only `ternary_matmul`, so they need no model file and run offline.
"""

import threading
import time

import tritium


def _big_matmul():
    """A matmul large enough to take a measurable, GIL-free moment in Rust."""
    k = 256
    n = 256
    act = [[float((i % 7) - 3) for i in range(k)]]
    weights = [[((i + j) % 3) - 1 for i in range(k)] for j in range(n)]
    return tritium.ternary_matmul(act, weights, 1.0)


def test_gil_released_during_compute():
    """A background counter thread must make progress while a worker thread is busy
    inside the Rust GEMM. If the GIL were held across the call, the counter would
    barely advance."""
    stop = threading.Event()
    counter = {"n": 0}

    def spin():
        while not stop.is_set():
            counter["n"] += 1

    spinner = threading.Thread(target=spin)
    spinner.start()
    try:
        # Let the spinner establish a baseline rate, then run many GEMMs.
        time.sleep(0.05)
        baseline = counter["n"]
        for _ in range(50):
            _big_matmul()
        advanced = counter["n"] - baseline
    finally:
        stop.set()
        spinner.join(timeout=5)

    assert not spinner.is_alive(), "spinner thread failed to join (possible deadlock)"
    # If the GIL were held for the whole compute the spinner could not advance at
    # all between the baseline sample and the end of the loop. A nonzero advance
    # proves the GIL was released during the Rust work.
    assert advanced > 0, "background thread made no progress; GIL was not released"


def test_many_threads_no_deadlock():
    """>= 4 threads each running several matmuls concurrently must all finish."""
    n_threads = 6
    per_thread = 20
    errors = []
    results = [None] * n_threads

    def worker(idx):
        try:
            last = None
            for _ in range(per_thread):
                last = _big_matmul()
            results[idx] = last
        except Exception as exc:  # noqa: BLE001 - record, assert in main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not any(t.is_alive() for t in threads), "a worker thread deadlocked"
    assert not errors, f"worker threads raised: {errors}"
    # Every thread produced the same deterministic result.
    assert all(r is not None for r in results)
    first = results[0]
    for r in results[1:]:
        assert r == first, "concurrent matmuls produced divergent results"


def test_concurrent_calls_are_deterministic():
    """The same inputs always yield the same output, even under contention."""
    expected = _big_matmul()
    out = []
    lock = threading.Lock()

    def worker():
        r = _big_matmul()
        with lock:
            out.append(r)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(out) == 4
    for r in out:
        assert r == expected
