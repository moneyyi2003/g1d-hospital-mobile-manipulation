"""Non-blocking RGB recording and deadline-based real-time pacing."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from queue import Full, Queue
import threading
import time
from typing import Callable


@dataclass(frozen=True)
class _FrameJob:
    pixels: object
    path: Path
    on_saved: Callable[[Path, bytes], None] | None


class AsyncPngWriter:
    """Encode lossless RGB frames off the simulation/control thread.

    The queue is deliberately bounded: if storage ever becomes slower than
    capture for a sustained period, memory use remains bounded and the
    producer applies backpressure instead of eventually exhausting RAM.
    """

    _STOP = object()

    def __init__(self, image_module, *, max_pending: int = 4) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._image = image_module
        self._queue: Queue[object] = Queue(maxsize=max_pending)
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="habitat-png-writer",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        pixels: object,
        path: Path,
        on_saved: Callable[[Path, bytes], None] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("RGB writer is already closed")
        job = _FrameJob(pixels, path, on_saved)
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(job, timeout=0.1)
                return
            except Full:
                # Recheck the worker error rather than blocking forever if an
                # encoder failure filled the bounded queue.
                continue

    def close(self) -> None:
        if self._closed:
            self._raise_if_failed()
            return
        self._closed = True
        while True:
            try:
                self._queue.put(self._STOP, timeout=0.1)
                break
            except Full:
                # The worker keeps draining after an encoding error, allowing
                # us to deliver the sentinel and join it before surfacing the
                # failure below.
                continue
        self._thread.join()
        self._raise_if_failed()

    def __enter__(self) -> "AsyncPngWriter":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._STOP:
                return
            if self._error is not None:
                continue
            job = item
            assert isinstance(job, _FrameJob)
            temporary = job.path.with_name(f".{job.path.name}.tmp")
            try:
                image = self._image.fromarray(job.pixels)
                preview = BytesIO()
                if job.on_saved is not None:
                    # The browser preview is intentionally lossy and small;
                    # the PNG written below remains the lossless run artifact.
                    image.save(preview, format="JPEG", quality=80)
                # compress_level=1 remains lossless but is several times
                # faster than Pillow's default level for 640x480 Habitat RGB.
                image.save(
                    temporary,
                    format="PNG",
                    compress_level=1,
                )
                temporary.replace(job.path)
                if job.on_saved is not None:
                    job.on_saved(job.path, preview.getvalue())
            except BaseException as exc:  # surfaced on the producer thread
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                with self._error_lock:
                    if self._error is None:
                        self._error = exc

    def _raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("Failed to encode a Habitat RGB frame") from error


class RealtimePacer:
    """Keep a loop on absolute deadlines so work time is not added to sleep."""

    def __init__(
        self,
        period_sec: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if period_sec <= 0:
            raise ValueError("period_sec must be positive")
        self._period = period_sec
        self._clock = clock
        self._sleep = sleeper
        self._deadline = clock()

    def wait(self) -> None:
        self._deadline += self._period
        remaining = self._deadline - self._clock()
        if remaining > 0:
            self._sleep(remaining)
        elif remaining < -self._period:
            # Do not attempt a burst of catch-up iterations after a long
            # external stall (for example, Nav2 replanning).
            self._deadline = self._clock()


__all__ = ["AsyncPngWriter", "RealtimePacer"]
