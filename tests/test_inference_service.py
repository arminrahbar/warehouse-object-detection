import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.VideoCapture = Mock(name="VideoCapture")
    cv2_stub.dnn = types.SimpleNamespace()
    sys.modules["cv2"] = cv2_stub

from detector_service.modules.inference import model as model_module
from detector_service.modules.inference.model import Detector
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


class _NetworkDouble:
    def __init__(self):
        self.layer_names = ["stem", "head_small", "neck", "head_large"]
        self.unconnected_layers = np.asarray([[2], [4]])
        self.outputs = [np.asarray([[0.5, 0.5, 0.2, 0.2, 0.9, 0.8]])]
        self.input_blob = None
        self.forward_calls = []
        self.layer_discovery_error = None

    def setInput(self, blob):
        self.input_blob = blob

    def getLayerNames(self):
        if self.layer_discovery_error is not None:
            raise self.layer_discovery_error
        return self.layer_names

    def getUnconnectedOutLayers(self):
        return self.unconnected_layers

    def forward(self, *args):
        self.forward_calls.append(args)
        return self.outputs


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


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.class_path = Path(self.temp_directory.name) / "objects.names"
        self.class_path.write_text("pallet\nforklift\nworker\n", encoding="utf-8")

        self.network = _NetworkDouble()
        self.blob = object()
        self.dnn = types.SimpleNamespace(
            readNet=Mock(return_value=self.network),
            blobFromImage=Mock(return_value=self.blob),
        )
        self.dnn_patch = patch.object(model_module.cv2, "dnn", self.dnn)
        self.dnn_patch.start()
        self.addCleanup(self.dnn_patch.stop)

        self.detector = Detector(
            "detector.weights",
            "detector.cfg",
            str(self.class_path),
            score_threshold=0.5,
        )

    def test_constructor_loads_network_and_class_names(self):
        self.dnn.readNet.assert_called_once_with("detector.weights", "detector.cfg")
        self.assertEqual(self.detector.classes, ["pallet", "forklift", "worker"])
        self.assertEqual(self.detector.img_height, 0)
        self.assertEqual(self.detector.img_width, 0)
        self.assertEqual(self.detector.score_threshold, 0.5)

    def test_predict_builds_expected_blob_and_runs_named_output_layers(self):
        frame = np.zeros((120, 200, 3), dtype=np.uint8)

        outputs = self.detector.predict(frame)

        self.dnn.blobFromImage.assert_called_once_with(
            frame,
            scalefactor=1 / 255.0,
            size=(416, 416),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.assertIs(self.network.input_blob, self.blob)
        self.assertEqual(self.network.forward_calls, [(["head_small", "head_large"],)])
        self.assertIs(outputs, self.network.outputs)
        self.assertEqual((self.detector.img_height, self.detector.img_width), (120, 200))

    def test_predict_falls_back_to_default_forward_when_layers_cannot_be_resolved(self):
        self.network.layer_discovery_error = RuntimeError("unsupported backend")

        outputs = self.detector.predict(np.ones((8, 10, 3), dtype=np.uint8))

        self.assertEqual(self.network.forward_calls, [()])
        self.assertIs(outputs, self.network.outputs)

    def test_predict_ignores_invalid_output_layer_indices(self):
        self.network.unconnected_layers = np.asarray([[0], [8]])

        self.detector.predict(np.ones((8, 10, 3), dtype=np.uint8))

        self.assertEqual(self.network.forward_calls, [()])

    def test_predict_rejects_none_and_zero_sized_frames(self):
        with self.assertRaisesRegex(ValueError, "Empty frame"):
            self.detector.predict(None)
        with self.assertRaisesRegex(ValueError, "Empty frame"):
            self.detector.predict(np.empty((0, 4, 3), dtype=np.uint8))

        self.dnn.blobFromImage.assert_not_called()

    def test_post_process_filters_objectness_and_decodes_pixel_boxes(self):
        self.detector.img_width = 200
        self.detector.img_height = 100
        raw_outputs = [
            [
                np.asarray(
                    [0.50, 0.25, 0.20, 0.40, 0.80, 0.10, 0.70, 0.20]
                ),
                np.asarray(
                    [0.20, 0.30, 0.10, 0.20, 0.50, 0.10, 0.90, 0.00]
                ),
                np.asarray([0.10, 0.10, 0.10, 0.10]),
            ],
            [
                np.asarray(
                    [0.10, 0.20, 0.30, 0.20, 0.90, 0.40, 0.40, 0.20]
                ),
            ],
        ]

        boxes, classes, objectness, class_scores = self.detector.post_process(
            raw_outputs
        )

        self.assertEqual(boxes, [[80, 5, 40, 40], [-10, 10, 60, 20]])
        self.assertEqual(classes, [1, 0])
        self.assertEqual(objectness, [0.8, 0.9])
        np.testing.assert_allclose(class_scores[0], [0.10, 0.70, 0.20])
        np.testing.assert_allclose(class_scores[1], [0.40, 0.40, 0.20])

    def test_post_process_skips_rows_without_class_probabilities(self):
        self.detector.img_width = 20
        self.detector.img_height = 20

        result = self.detector.post_process(
            [np.asarray([[0.5, 0.5, 0.5, 0.5, 0.9]])]
        )

        self.assertEqual(result, ([], [], [], []))


if __name__ == "__main__":
    unittest.main()
