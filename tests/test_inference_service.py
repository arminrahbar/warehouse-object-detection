import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.VideoCapture = Mock(name="VideoCapture")
    sys.modules["cv2"] = cv2_stub

from detector_service.modules.inference.preprocessing import Preprocessing


class _CaptureDouble:
    def __init__(self, frames, *, initially_open=True):
        self._frames = iter(frames)
        self._open = initially_open
        self.read_count = 0
        self.released = False

    def isOpened(self):
        return self._open

    def read(self):
        self.read_count += 1
        try:
            return True, next(self._frames)
        except StopIteration:
            return False, None

    def release(self):
        self.released = True
        self._open = False


class PreprocessingTests(unittest.TestCase):
    @staticmethod
    def _frames(count):
        return [np.full((2, 3, 1), index, dtype=np.uint8) for index in range(count)]

    def test_constructor_preserves_source_and_default_interval(self):
        stream = Preprocessing("udp://127.0.0.1:23000")

        self.assertEqual(stream.filename, "udp://127.0.0.1:23000")
        self.assertEqual(stream.drop_rate, 10)

    def test_first_and_every_nth_decoded_frame_are_yielded(self):
        capture = _CaptureDouble(self._frames(8))

        with patch(
            "detector_service.modules.inference.preprocessing.cv2.VideoCapture",
            return_value=capture,
        ) as constructor:
            selected = list(Preprocessing("clip.mp4", drop_rate=3).capture_video())

        constructor.assert_called_once_with("clip.mp4")
        self.assertEqual([int(frame[0, 0, 0]) for frame in selected], [0, 3, 6])
        self.assertEqual(capture.read_count, 9)
        self.assertTrue(capture.released)

    def test_decode_failure_ends_iteration_and_releases_source(self):
        capture = _CaptureDouble(self._frames(0))

        with patch(
            "detector_service.modules.inference.preprocessing.cv2.VideoCapture",
            return_value=capture,
        ):
            self.assertEqual(
                list(Preprocessing("empty.mp4", drop_rate=1).capture_video()),
                [],
            )

        self.assertEqual(capture.read_count, 1)
        self.assertTrue(capture.released)

    def test_closing_partially_consumed_generator_releases_source(self):
        capture = _CaptureDouble(self._frames(5))

        with patch(
            "detector_service.modules.inference.preprocessing.cv2.VideoCapture",
            return_value=capture,
        ):
            frames = Preprocessing("live-stream", drop_rate=2).capture_video()
            first = next(frames)
            self.assertEqual(int(first[0, 0, 0]), 0)
            self.assertFalse(capture.released)
            frames.close()

        self.assertTrue(capture.released)

    def test_decoder_exception_still_releases_source(self):
        capture = _CaptureDouble(self._frames(1))
        capture.read = Mock(side_effect=RuntimeError("decoder failed"))

        with patch(
            "detector_service.modules.inference.preprocessing.cv2.VideoCapture",
            return_value=capture,
        ):
            with self.assertRaisesRegex(RuntimeError, "decoder failed"):
                list(Preprocessing("unstable-stream", drop_rate=1).capture_video())

        self.assertTrue(capture.released)

    def test_invalid_interval_is_rejected_before_source_is_opened(self):
        with patch(
            "detector_service.modules.inference.preprocessing.cv2.VideoCapture"
        ) as constructor:
            with self.assertRaisesRegex(ValueError, "greater than 0"):
                list(Preprocessing("clip.mp4", drop_rate=0).capture_video())

        constructor.assert_not_called()

    def test_unopenable_source_reports_its_name(self):
        capture = _CaptureDouble([], initially_open=False)

        with patch(
            "detector_service.modules.inference.preprocessing.cv2.VideoCapture",
            return_value=capture,
        ):
            with self.assertRaisesRegex(ValueError, "offline-camera"):
                list(
                    Preprocessing(
                        "offline-camera",
                        drop_rate=1,
                    ).capture_video()
                )

        self.assertEqual(capture.read_count, 0)

    def test_capture_that_closes_before_reading_is_released(self):
        capture = _CaptureDouble(self._frames(2))
        capture.isOpened = Mock(side_effect=[True, False])

        with patch(
            "detector_service.modules.inference.preprocessing.cv2.VideoCapture",
            return_value=capture,
        ):
            selected = list(Preprocessing("camera", drop_rate=1).capture_video())

        self.assertEqual(selected, [])
        self.assertEqual(capture.read_count, 0)
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
