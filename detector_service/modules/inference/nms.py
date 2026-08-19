"""Class-aware non-maximum suppression for decoded detections."""

from collections.abc import Sequence
from numbers import Integral
from typing import List, Tuple

import numpy as np


class NMS:
    """Filter and de-duplicate detections using their predicted classes."""

    def __init__(self, score_threshold: float, nms_iou_threshold: float) -> None:
        self.score_threshold = self._validate_threshold(
            score_threshold,
            "score_threshold",
        )
        self.nms_iou_threshold = self._validate_threshold(
            nms_iou_threshold,
            "nms_iou_threshold",
        )

    @staticmethod
    def _validate_threshold(value: float, name: str) -> float:
        """Normalize a probability-like threshold and reject invalid values."""
        try:
            threshold = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number between 0 and 1") from exc

        if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
        return threshold

    @staticmethod
    def _calculate_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Measure IoU between one xywh box and zero or more xywh boxes."""
        candidates = np.asarray(boxes, dtype=float)
        if candidates.size == 0:
            return np.empty(0, dtype=float)

        reference = np.asarray(box, dtype=float)
        reference_end = reference[:2] + reference[2:]
        candidate_ends = candidates[:, :2] + candidates[:, 2:]

        intersection_start = np.maximum(reference[:2], candidates[:, :2])
        intersection_end = np.minimum(reference_end, candidate_ends)
        intersection_size = np.clip(
            intersection_end - intersection_start,
            a_min=0.0,
            a_max=None,
        )
        intersection_area = intersection_size[:, 0] * intersection_size[:, 1]

        reference_area = reference[2] * reference[3]
        candidate_areas = candidates[:, 2] * candidates[:, 3]
        union_area = reference_area + candidate_areas - intersection_area

        result = np.zeros(candidates.shape[0], dtype=float)
        np.divide(
            intersection_area,
            union_area,
            out=result,
            where=union_area > 0.0,
        )
        return result

    @staticmethod
    def _validate_inputs(
        bboxes: Sequence[Sequence[float]],
        class_ids: Sequence[int],
        scores: Sequence[float],
        class_scores: Sequence[Sequence[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate collection lengths and numeric box/objectness values."""
        lengths = {
            "bboxes": len(bboxes),
            "class_ids": len(class_ids),
            "scores": len(scores),
            "class_scores": len(class_scores),
        }
        if len(set(lengths.values())) != 1:
            received = ", ".join(
                f"{field}={length}" for field, length in lengths.items()
            )
            raise ValueError(
                f"Detection inputs must have equal lengths; received {received}"
            )

        count = lengths["bboxes"]
        if count == 0:
            return np.empty((0, 4), dtype=float), np.empty(0, dtype=float)

        try:
            boxes = np.asarray(bboxes, dtype=float)
            objectness = np.asarray(scores, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Bounding boxes and objectness scores must be numeric"
            ) from exc

        if boxes.shape != (count, 4):
            raise ValueError("Each bounding box must contain [x, y, width, height]")
        if not np.all(np.isfinite(boxes)):
            raise ValueError("Bounding boxes must contain finite values")
        if np.any(boxes[:, 2:] < 0.0):
            raise ValueError("Bounding-box width and height cannot be negative")

        if objectness.shape != (count,):
            raise ValueError("Each detection must have one objectness score")
        if (
            not np.all(np.isfinite(objectness))
            or np.any(objectness < 0.0)
            or np.any(objectness > 1.0)
        ):
            raise ValueError(
                "Objectness scores must be finite values between 0 and 1"
            )

        return boxes, objectness

    @staticmethod
    def confidence_scores(
        class_ids: Sequence[int],
        scores: Sequence[float],
        class_scores: Sequence[Sequence[float]],
    ) -> List[float]:
        """Compute objectness times predicted-class probability per detection."""
        lengths = {
            "class_ids": len(class_ids),
            "scores": len(scores),
            "class_scores": len(class_scores),
        }
        if len(set(lengths.values())) != 1:
            received = ", ".join(
                f"{field}={length}" for field, length in lengths.items()
            )
            raise ValueError(
                f"Confidence inputs must have equal lengths; received {received}"
            )

        combined = []
        for position in range(lengths["class_ids"]):
            predicted_class = class_ids[position]
            if isinstance(predicted_class, (bool, np.bool_)) or not isinstance(
                predicted_class,
                Integral,
            ):
                raise ValueError(
                    f"class_ids[{position}] must be a non-negative integer"
                )
            predicted_class = int(predicted_class)
            if predicted_class < 0:
                raise ValueError(
                    f"class_ids[{position}] must be a non-negative integer"
                )

            try:
                objectness = float(scores[position])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"scores[{position}] must be numeric") from exc
            if not np.isfinite(objectness) or not 0.0 <= objectness <= 1.0:
                raise ValueError(f"scores[{position}] must be between 0 and 1")

            try:
                probabilities = np.asarray(
                    class_scores[position],
                    dtype=float,
                ).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"class_scores[{position}] must be numeric"
                ) from exc

            if predicted_class >= probabilities.size:
                raise ValueError(
                    f"class_scores[{position}] has no entry for class "
                    f"{predicted_class}"
                )
            if (
                not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0.0)
                or np.any(probabilities > 1.0)
            ):
                raise ValueError(
                    f"class_scores[{position}] must contain values between 0 and 1"
                )

            combined.append(objectness * probabilities[predicted_class])

        return [float(value) for value in combined]

    def filter(
        self,
        bboxes: List[List[int]],
        class_ids: List[int],
        scores: List[float],
        class_scores: List[Sequence[float]],
    ) -> Tuple[List[List[int]], List[int], List[float], List[Sequence[float]]]:
        """Apply confidence filtering followed by same-class IoU suppression."""
        boxes, _ = self._validate_inputs(
            bboxes,
            class_ids,
            scores,
            class_scores,
        )
        if len(bboxes) == 0:
            return [], [], [], []

        combined = np.asarray(
            self.confidence_scores(class_ids, scores, class_scores),
            dtype=float,
        )
        eligible = np.flatnonzero(combined >= self.score_threshold)

        retained = []
        eligible_classes = {int(class_ids[index]) for index in eligible}
        for predicted_class in sorted(eligible_classes):
            class_members = np.asarray(
                [
                    index
                    for index in eligible
                    if int(class_ids[index]) == predicted_class
                ],
                dtype=int,
            )
            ranking = np.lexsort((class_members, -combined[class_members]))
            pending = class_members[ranking]

            while pending.size:
                winner = int(pending[0])
                retained.append(winner)
                pending = pending[1:]
                if not pending.size:
                    break

                overlaps = self._calculate_iou(boxes[winner], boxes[pending])
                pending = pending[overlaps <= self.nms_iou_threshold]

        retained.sort(key=lambda index: (-combined[index], index))
        return (
            [bboxes[index] for index in retained],
            [class_ids[index] for index in retained],
            [scores[index] for index in retained],
            [class_scores[index] for index in retained],
        )
