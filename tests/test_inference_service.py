import contextlib
import io
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
    cv2_stub.FONT_HERSHEY_SIMPLEX = 0
    cv2_stub.VideoCapture = Mock(name="VideoCapture")
    cv2_stub.dnn = types.SimpleNamespace()
    cv2_stub.rectangle = Mock(side_effect=lambda frame, *args, **kwargs: frame)
    cv2_stub.putText = Mock(side_effect=lambda frame, *args, **kwargs: frame)
    cv2_stub.imwrite = Mock(return_value=True)
    sys.modules["cv2"] = cv2_stub

from detector_service import app as app_module
from detector_service.app import (
    DEFAULT_MODEL_DIR,
    InferenceService,
    build_parser,
    positive_int,
)
from detector_service.modules.inference import model as model_module
from detector_service.modules.inference.model import Detector
from detector_service.modules.inference.nms import NMS
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


class _ClosableFrameIterator:
    def __init__(self, frames):
        self._frames = iter(frames)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._frames)

    def close(self):
        self.closed = True


class _StreamDouble:
    def __init__(self, frame_count):
        frames = [
            np.full((20, 20, 3), index, dtype=np.uint8)
            for index in range(frame_count)
        ]
        self.iterator = _ClosableFrameIterator(frames)

    def capture_video(self):
        return self.iterator


class _ServiceDetectorDouble:
    classes = ["pallet", "forklift"]

    def __init__(self, *, interrupt=False, error=None):
        self.interrupt = interrupt
        self.error = error
        self.predict_calls = []
        self.post_process_calls = []

    def predict(self, frame):
        self.predict_calls.append(frame)
        if self.interrupt:
            raise KeyboardInterrupt
        if self.error is not None:
            raise self.error
        return ["raw-output"]

    def post_process(self, predictions):
        self.post_process_calls.append(predictions)
        return (
            [[2, 3, 10, 8]],
            [0],
            [0.8],
            [[0.75, 0.25]],
        )


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


class InferenceServiceTests(unittest.TestCase):
    def setUp(self):
        self.rectangle = patch.object(app_module.cv2, "rectangle", Mock())
        self.put_text = patch.object(app_module.cv2, "putText", Mock())
        self.imwrite = patch.object(app_module.cv2, "imwrite", Mock(return_value=True))
        self.rectangle_mock = self.rectangle.start()
        self.put_text_mock = self.put_text.start()
        self.imwrite_mock = self.imwrite.start()
        self.addCleanup(self.rectangle.stop)
        self.addCleanup(self.put_text.stop)
        self.addCleanup(self.imwrite.stop)

    @staticmethod
    def _service(frame_count=1, *, save_dir=None, max_frames=None, detector=None):
        stream = _StreamDouble(frame_count)
        service = InferenceService(
            stream=stream,
            detector=detector or _ServiceDetectorDouble(),
            nms=NMS(score_threshold=0.5, nms_iou_threshold=0.3),
            save_dir=save_dir,
            max_frames=max_frames,
        )
        return service, stream

    def test_run_reports_combined_confidence_and_closes_stream(self):
        service, stream = self._service()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            processed = service.run()

        self.assertEqual(processed, 1)
        self.assertTrue(stream.iterator.closed)
        self.assertIn("[FRAME] index=0 detections=1", output.getvalue())
        self.assertIn("combined_confidence=0.6000", output.getvalue())
        self.assertIn("Inference completed. Processed 1 frames", output.getvalue())
        self.assertEqual(self.put_text_mock.call_args.args[1], "pallet: 0.60")
        self.imwrite_mock.assert_not_called()

    def test_run_annotates_a_copy_of_the_stream_frame(self):
        detector = _ServiceDetectorDouble()
        service, _ = self._service(detector=detector)

        with patch.object(
            service,
            "draw_boxes",
            side_effect=lambda frame, *args: frame,
        ) as draw_boxes, contextlib.redirect_stdout(io.StringIO()):
            service.run()

        source_frame = detector.predict_calls[0]
        annotation_frame = draw_boxes.call_args.args[0]
        self.assertIsNot(annotation_frame, source_frame)
        np.testing.assert_array_equal(annotation_frame, source_frame)

    def test_draw_boxes_uses_fallback_name_for_unknown_class(self):
        service, _ = self._service()
        frame = np.zeros((30, 30, 3), dtype=np.uint8)

        returned = service.draw_boxes(
            frame,
            bboxes=[[4.8, 5.2, 10.9, 6.1]],
            class_ids=[9],
            confidences=[0.725],
        )

        self.assertIs(returned, frame)
        self.rectangle_mock.assert_called_once_with(
            frame,
            (4, 5),
            (14, 11),
            (0, 255, 0),
            2,
        )
        self.put_text_mock.assert_called_once_with(
            frame,
            "class_9: 0.72",
            (4, 20),
            app_module.cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    def test_save_frame_writes_zero_padded_jpeg_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service, _ = self._service(save_dir=temp_dir)
            frame = np.zeros((4, 4, 3), dtype=np.uint8)

            path = service.save_frame(frame, frame_number=27)

        self.assertEqual(path.name, "frame_000027.jpg")
        self.imwrite_mock.assert_called_once_with(str(path), frame)

    def test_constructor_creates_nested_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "nested" / "detections"

            service, _ = self._service(save_dir=output_dir)

            self.assertEqual(service.save_dir, output_dir)
            self.assertTrue(output_dir.is_dir())

    def test_save_frame_raises_when_encoder_reports_failure(self):
        self.imwrite_mock.return_value = False
        with tempfile.TemporaryDirectory() as temp_dir:
            service, _ = self._service(save_dir=temp_dir)

            with self.assertRaisesRegex(OSError, "frame_000004.jpg"):
                service.save_frame(
                    np.zeros((4, 4, 3), dtype=np.uint8),
                    frame_number=4,
                )

    def test_frame_limit_stops_and_closes_longer_stream(self):
        service, stream = self._service(frame_count=5, max_frames=2)
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            processed = service.run()

        self.assertEqual(processed, 2)
        self.assertTrue(stream.iterator.closed)
        self.assertIn("Reached configured frame limit: 2", output.getvalue())
        self.assertNotIn("[FRAME] index=2", output.getvalue())

    def test_keyboard_interrupt_is_reported_and_stream_is_closed(self):
        detector = _ServiceDetectorDouble(interrupt=True)
        service, stream = self._service(detector=detector)
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            processed = service.run()

        self.assertEqual(processed, 0)
        self.assertTrue(stream.iterator.closed)
        self.assertIn("Inference interrupted by user", output.getvalue())
        self.assertIn("Processed 0 frames", output.getvalue())

    def test_non_interrupt_exception_propagates_after_stream_cleanup(self):
        detector = _ServiceDetectorDouble(error=RuntimeError("inference failed"))
        service, stream = self._service(detector=detector)

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            service.run()

        self.assertTrue(stream.iterator.closed)

    def test_constructor_rejects_nonpositive_frame_limit(self):
        for invalid in (0, -1):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError,
                "greater than 0",
            ):
                self._service(max_frames=invalid)

    def test_positive_int_accepts_only_values_above_zero(self):
        self.assertEqual(positive_int("7"), 7)
        for invalid in ("0", "-2", "not-a-number"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                Exception,
                "positive integer",
            ):
                positive_int(invalid)

    def test_cli_defaults_and_overrides_match_runtime_contract(self):
        defaults = build_parser().parse_args([])
        self.assertEqual(defaults.source, "udp://127.0.0.1:23000")
        self.assertEqual(defaults.weights.parent, DEFAULT_MODEL_DIR)
        self.assertTrue(defaults.weights.name.endswith("_2.weights"))
        self.assertEqual(defaults.frame_interval, 60)
        self.assertIsNone(defaults.max_frames)
        self.assertEqual(defaults.candidate_threshold, 0.5)
        self.assertEqual(defaults.confidence_threshold, 0.5)
        self.assertEqual(defaults.nms_iou_threshold, 0.3)
        self.assertFalse(defaults.no_save)
        self.assertIsInstance(defaults.save_dir, Path)

        configured = build_parser().parse_args(
            [
                "--source",
                "clip.mp4",
                "--frame-interval",
                "4",
                "--max-frames",
                "3",
                "--no-save",
            ]
        )
        self.assertEqual(configured.source, "clip.mp4")
        self.assertEqual(configured.frame_interval, 4)
        self.assertEqual(configured.max_frames, 3)
        self.assertTrue(configured.no_save)

    def test_main_composes_cli_dependencies_and_returns_service_result(self):
        stream = object()
        detector = types.SimpleNamespace(classes=[])
        nms = object()
        service = Mock()
        service.run.return_value = 12

        with patch.object(app_module, "Preprocessing", return_value=stream) as prep, \
            patch.object(app_module, "Detector", return_value=detector) as model, \
            patch.object(app_module, "NMS", return_value=nms) as nms_constructor, \
            patch.object(
                app_module,
                "InferenceService",
                return_value=service,
            ) as service_constructor:
            result = app_module.main(
                [
                    "--source",
                    "clip.mp4",
                    "--weights",
                    "custom.weights",
                    "--config",
                    "custom.cfg",
                    "--classes",
                    "custom.names",
                    "--frame-interval",
                    "5",
                    "--max-frames",
                    "2",
                    "--candidate-threshold",
                    "0.4",
                    "--confidence-threshold",
                    "0.6",
                    "--nms-iou-threshold",
                    "0.25",
                    "--no-save",
                ]
            )

        self.assertEqual(result, 12)
        prep.assert_called_once_with("clip.mp4", drop_rate=5)
        model.assert_called_once_with(
            "custom.weights",
            "custom.cfg",
            "custom.names",
            0.4,
        )
        nms_constructor.assert_called_once_with(0.6, 0.25)
        service_constructor.assert_called_once_with(
            stream,
            detector,
            nms,
            save_dir=None,
            max_frames=2,
        )
        service.run.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
