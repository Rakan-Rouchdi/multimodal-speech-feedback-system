from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator


class Timer:
    """
    Simple timer collector for pipeline stage durations in milliseconds.
    """
    def __init__(self):
        self.ms: Dict[str, int] = {}

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            ms = round((end - start) * 1000, 1)
            self.ms[name] = ms if ms > 0 else 0.1

