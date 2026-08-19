"""Open video sources and yield frames at a configured interval."""

from typing import Generator

import cv2
import numpy as np


class Preprocessing:
    """Adapt an OpenCV video source into a sampled frame generator."""

    def __init__(self, filename: str, drop_rate: int = 10) -> None:
        self.filename = filename
        self.drop_rate = drop_rate

    def capture_video(self) -> Generator[np.ndarray, None, None]:
        """Yield decoded frames numbered 0, n, 2n, and release the capture."""
        if self.drop_rate <= 0:
            raise ValueError("drop_rate must be greater than 0.")

        capture = cv2.VideoCapture(self.filename)
        if not capture.isOpened():
            raise ValueError(
                f"Error: Unable to open video file '{self.filename}'."
            )

        decoded_index = 0
        try:
            while capture.isOpened():
                decoded, frame = capture.read()
                if not decoded:
                    return

                emit_frame = decoded_index % self.drop_rate == 0
                decoded_index += 1
                if emit_frame:
                    yield frame
        finally:
            capture.release()
