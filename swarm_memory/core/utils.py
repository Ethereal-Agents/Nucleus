"""
swarm_memory/core/utils.py

Shared utilities across the SwarmMemory package.
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def timed(label: str) -> Generator[None, None, None]:
    """
    Context manager that logs the wall-clock duration of a code block.

    Example:
        with timed("dense_search"):
            results = db.execute(...)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug("⏱  %s: %.1f ms", label, elapsed_ms)
