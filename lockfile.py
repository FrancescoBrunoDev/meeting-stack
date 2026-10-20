"""An exclusive lock shared across processes, released automatically on death.

A tiny wrapper over `fcntl.flock`: the kernel drops the lock when the holding
process exits or crashes, so a killed job can never leave it stuck. Used to keep
image generation and transcription jobs from running at the same time -- the two
together put the GPU and the NPU under load and can wedge halogen.
"""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path


class Lock:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self, wait: float, poll: float = 0.25) -> bool:
        """Take the lock, waiting up to `wait` seconds (0 tries once)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._fd is None:
            self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        deadline = time.monotonic() + wait
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError:
                if time.monotonic() >= deadline:
                    self.release()
                    return False
                time.sleep(poll)

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> "Lock":
        self.acquire(0)
        return self

    def __exit__(self, *exc) -> None:
        self.release()
