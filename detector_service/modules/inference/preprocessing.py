"""Open video sources and yield frames at a configured interval."""

from typing import Generator
from urllib.parse import urlsplit

import cv2
import numpy as np


LIVE_STREAM_SCHEMES = frozenset({"rtmp", "rtsp", "tcp", "udp"})


class Preprocessing:
    """Adapt an OpenCV video source into a sampled frame generator."""

    def __init__(
        self,
        filename: str,
        drop_rate: int = 10,
        *,
        network_read_retries: int = 3,
    ) -> None:
        self.filename = filename
        self.drop_rate = drop_rate
        self.network_read_retries = network_read_retries

    def _is_live_stream(self) -> bool:
        """Return whether the source uses a supported live-stream scheme."""
        return urlsplit(str(self.filename)).scheme.lower() in LIVE_STREAM_SCHEMES

    def capture_video(self) -> Generator[np.ndarray, None, None]:
        """Yield the first decoded frame and every Nth frame thereafter."""
        if self.drop_rate <= 0:
            raise ValueError("drop_rate must be greater than 0.")
        if (
            not isinstance(self.network_read_retries, int)
            or isinstance(self.network_read_retries, bool)
            or self.network_read_retries < 0
        ):
            raise ValueError("network_read_retries must be a non-negative integer.")

        capture = cv2.VideoCapture(self.filename)
        if not capture.isOpened():
            raise ValueError(
                f"Unable to open video source '{self.filename}'."
            )

        live_stream = self._is_live_stream()
        decoded_index = 0
        consecutive_read_failures = 0
        try:
            while True:
                decoded, frame = capture.read()
                if not decoded:
                    if not live_stream:
                        return

                    consecutive_read_failures += 1
                    if consecutive_read_failures > self.network_read_retries:
                        attempts = consecutive_read_failures
                        raise RuntimeError(
                            "Unable to read from network video source "
                            f"'{self.filename}' after {attempts} consecutive "
                            f"attempts ({self.network_read_retries} retries)."
                        )
                    continue

                consecutive_read_failures = 0
                emit_frame = decoded_index % self.drop_rate == 0
                decoded_index += 1
                if emit_frame:
                    yield frame
        finally:
            capture.release()
