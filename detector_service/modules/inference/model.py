"""OpenCV-DNN adapter for YOLO model execution and candidate decoding."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


OPENCV_ERRORS = (cv2.error,) if hasattr(cv2, "error") else ()


class Detector:
    """Load a YOLO network, run frames, and decode raw output candidates."""

    def __init__(
        self,
        weights_path: str,
        config_path: str,
        class_path: str,
        score_threshold: float = 0.5,
    ) -> None:
        try:
            threshold = float(score_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "score_threshold must be a number between 0 and 1"
            ) from exc
        if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError("score_threshold must be between 0 and 1")

        with Path(class_path).open("r", encoding="utf-8") as class_file:
            self.classes = [line.strip() for line in class_file]
        if not self.classes:
            raise ValueError("Class vocabulary must contain at least one class name.")

        empty_lines = [
            index
            for index, class_name in enumerate(self.classes, start=1)
            if not class_name
        ]
        if empty_lines:
            raise ValueError(
                "Class vocabulary contains empty class names on line(s): "
                + ", ".join(str(index) for index in empty_lines)
            )

        duplicate_names = []
        seen_names = set()
        for class_name in self.classes:
            if class_name in seen_names and class_name not in duplicate_names:
                duplicate_names.append(class_name)
            seen_names.add(class_name)
        if duplicate_names:
            raise ValueError(
                "Class vocabulary contains duplicate class names: "
                + ", ".join(repr(name) for name in duplicate_names)
            )

        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.img_height: int = 0
        self.img_width: int = 0
        self.score_threshold = threshold

    def _output_layer_names(self) -> List[str]:
        """Resolve one-based OpenCV output indices, falling back on failure."""
        try:
            layer_names = self.net.getLayerNames()
            disconnected = self.net.getUnconnectedOutLayers()
            if disconnected is None:
                return []

            resolved = []
            for raw_index in np.asarray(disconnected).reshape(-1):
                zero_based = int(raw_index) - 1
                if 0 <= zero_based < len(layer_names):
                    resolved.append(layer_names[zero_based])
            return resolved
        except OPENCV_ERRORS:
            return []

    def predict(self, preprocessed_frame: np.ndarray) -> List[np.ndarray]:
        """Run a non-empty BGR frame through the network's output layers."""
        if preprocessed_frame is None or preprocessed_frame.size == 0:
            raise ValueError("Empty frame provided to Detector.predict().")

        self.img_height, self.img_width = preprocessed_frame.shape[:2]
        network_input = cv2.dnn.blobFromImage(
            preprocessed_frame,
            scalefactor=1 / 255.0,
            size=(416, 416),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(network_input)

        output_layers = self._output_layer_names()
        if output_layers:
            return self.net.forward(output_layers)
        return self.net.forward()

    def post_process(
        self,
        predict_output: List[np.ndarray],
    ) -> Tuple[List[List[int]], List[int], List[float], List[np.ndarray]]:
        """Decode objectness-qualified YOLO rows into pixel-space xywh boxes."""
        boxes = []
        predicted_classes = []
        objectness_values = []
        probability_vectors = []

        for output_layer in predict_output:
            for raw_detection in output_layer:
                if len(raw_detection) < 5:
                    continue

                objectness = float(raw_detection[4])
                if objectness <= self.score_threshold:
                    continue

                probabilities = raw_detection[5:]
                if len(probabilities) == 0:
                    continue

                predicted_class = int(np.argmax(probabilities))
                center_x = int(raw_detection[0] * self.img_width)
                center_y = int(raw_detection[1] * self.img_height)
                box_width = int(raw_detection[2] * self.img_width)
                box_height = int(raw_detection[3] * self.img_height)
                left = int(center_x - box_width / 2)
                top = int(center_y - box_height / 2)

                boxes.append([left, top, box_width, box_height])
                predicted_classes.append(predicted_class)
                objectness_values.append(objectness)
                probability_vectors.append(probabilities)

        return (
            boxes,
            predicted_classes,
            objectness_values,
            probability_vectors,
        )
