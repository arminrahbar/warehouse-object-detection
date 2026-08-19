"""Detection matching, precision-recall, and interpolated AP metrics."""

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np


DetectionRecord = Tuple[float, bool]
DetectionMatches = Dict[int, List[DetectionRecord]]
GroundTruthCounts = Dict[int, int]


def calculate_iou(box_a, box_b):
    """Return IoU for two boxes expressed as `[x, y, width, height]`."""
    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("Bounding boxes must contain exactly four values.")

    first = np.asarray(box_a, dtype=float)
    second = np.asarray(box_b, dtype=float)
    first_end = first[:2] + first[2:]
    second_end = second[:2] + second[2:]

    overlap_start = np.maximum(first[:2], second[:2])
    overlap_end = np.minimum(first_end, second_end)
    overlap_size = np.maximum(0.0, overlap_end - overlap_start)
    intersection = float(overlap_size[0] * overlap_size[1])

    first_area = max(0.0, float(first[2])) * max(0.0, float(first[3]))
    second_area = max(0.0, float(second[2])) * max(0.0, float(second[3]))
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _validate_image_inputs(
    boxes,
    classes,
    scores,
    class_scores,
    ground_truth_boxes,
    ground_truth_classes,
):
    prediction_lengths = {
        len(boxes),
        len(classes),
        len(scores),
        len(class_scores),
    }
    if len(prediction_lengths) != 1:
        raise ValueError(
            "Each image must have the same number of boxes, class IDs, "
            "objectness scores, and class-score vectors."
        )
    if len(ground_truth_boxes) != len(ground_truth_classes):
        raise ValueError(
            "Each image must have the same number of ground-truth boxes and "
            "class IDs."
        )


def _detection_score(objectness, class_score, predicted_class, eval_type):
    objectness_value = float(objectness)
    class_id = int(predicted_class)
    probabilities = np.asarray(class_score, dtype=float).reshape(-1)

    if class_id < 0:
        raise ValueError("Predicted class IDs must be non-negative.")
    if probabilities.size == 0:
        predicted_probability = 0.0
    elif probabilities.size == 1:
        predicted_probability = float(probabilities[0])
    elif class_id < probabilities.size:
        predicted_probability = float(probabilities[class_id])
    else:
        raise ValueError(
            f"Predicted class {class_id} is outside a class-score vector "
            f"of length {probabilities.size}."
        )

    if eval_type == "class_scores":
        return predicted_probability
    if eval_type == "combined":
        return objectness_value * predicted_probability
    if eval_type == "objectness":
        return objectness_value
    raise ValueError(f"Unsupported eval_type: {eval_type}")


def match_detections(
    boxes: Sequence[Sequence[Sequence[float]]],
    classes: Sequence[Sequence[int]],
    scores: Sequence[Sequence[float]],
    cls_scores: Sequence[Sequence[Sequence[float]]],
    gt_boxes: Sequence[Sequence[Sequence[float]]],
    gt_classes: Sequence[Sequence[int]],
    map_iou_threshold: float,
    eval_type: str = "combined",
) -> Tuple[DetectionMatches, GroundTruthCounts]:
    """Match predictions to same-class labels once per image and object."""
    threshold = float(map_iou_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("map_iou_threshold must be between 0 and 1.")
    supported_score_types = {"class_scores", "combined", "objectness"}
    if eval_type not in supported_score_types:
        raise ValueError(f"Unsupported eval_type: {eval_type}")

    image_counts = {
        len(boxes),
        len(classes),
        len(scores),
        len(cls_scores),
        len(gt_boxes),
        len(gt_classes),
    }
    if len(image_counts) != 1:
        raise ValueError(
            "Prediction and ground-truth inputs must describe the same images."
        )

    matches = defaultdict(list)
    label_counts = defaultdict(int)

    for image_index in range(len(boxes)):
        image_predictions = (
            boxes[image_index],
            classes[image_index],
            scores[image_index],
            cls_scores[image_index],
        )
        image_labels = (gt_boxes[image_index], gt_classes[image_index])
        _validate_image_inputs(*image_predictions, *image_labels)

        labels_by_class = defaultdict(list)
        for label_box, label_class in zip(*image_labels):
            class_id = int(label_class)
            if class_id < 0:
                raise ValueError("Ground-truth class IDs must be non-negative.")
            if len(label_box) != 4:
                raise ValueError("Bounding boxes must contain exactly four values.")
            labels_by_class[class_id].append(label_box)
            label_counts[class_id] += 1

        predictions_by_class = defaultdict(list)
        for original_index, (
            prediction_box,
            prediction_class,
            objectness,
            probability_vector,
        ) in enumerate(zip(*image_predictions)):
            class_id = int(prediction_class)
            if len(prediction_box) != 4:
                raise ValueError("Bounding boxes must contain exactly four values.")
            evaluation_score = _detection_score(
                objectness,
                probability_vector,
                class_id,
                eval_type,
            )
            predictions_by_class[class_id].append(
                (float(evaluation_score), original_index, prediction_box)
            )

        for class_id, class_predictions in predictions_by_class.items():
            available_labels = labels_by_class.get(class_id, [])
            consumed_labels = set()
            ranked_predictions = sorted(
                class_predictions,
                key=lambda record: (-record[0], record[1]),
            )

            for evaluation_score, _, prediction_box in ranked_predictions:
                best_label_index = -1
                best_overlap = 0.0
                for label_index, label_box in enumerate(available_labels):
                    if label_index in consumed_labels:
                        continue
                    overlap = calculate_iou(prediction_box, label_box)
                    if best_label_index < 0 or overlap > best_overlap:
                        best_label_index = label_index
                        best_overlap = overlap

                is_match = (
                    best_label_index >= 0 and best_overlap >= threshold
                )
                if is_match:
                    consumed_labels.add(best_label_index)
                matches[class_id].append((evaluation_score, bool(is_match)))

    return dict(matches), dict(label_counts)


def calculate_precision_recall_curve(
    matches: DetectionMatches,
    ground_truth_counts: GroundTruthCounts,
    num_classes: int = 20,
):
    """Create score-ordered precision, recall, and threshold arrays by class."""
    if num_classes < 0:
        raise ValueError("num_classes must be non-negative.")

    precision = {}
    recall = {}
    thresholds = {}

    for class_id in range(num_classes):
        ranked = sorted(
            matches.get(class_id, []),
            key=lambda record: record[0],
            reverse=True,
        )
        if not ranked:
            precision[class_id] = np.empty(0, dtype=float)
            recall[class_id] = np.empty(0, dtype=float)
            thresholds[class_id] = np.empty(0, dtype=float)
            continue

        thresholds[class_id] = np.asarray(
            [score for score, _ in ranked],
            dtype=float,
        )
        true_positives = np.asarray(
            [is_true_positive for _, is_true_positive in ranked],
            dtype=int,
        )
        cumulative_true_positives = np.cumsum(true_positives)
        cumulative_predictions = np.arange(1, len(ranked) + 1)
        precision[class_id] = (
            cumulative_true_positives / cumulative_predictions
        )

        label_count = int(ground_truth_counts.get(class_id, 0))
        if label_count > 0:
            recall[class_id] = cumulative_true_positives / label_count
        else:
            recall[class_id] = np.zeros(len(ranked), dtype=float)

    return precision, recall, thresholds


def calculate_map_x_point_interpolated(
    precision_recall_points,
    num_classes,
    num_interpolated_points=11,
):
    """Average maximum precision at evenly spaced recall thresholds."""
    if num_classes < 0:
        raise ValueError("num_classes must be non-negative.")
    if num_interpolated_points <= 0:
        raise ValueError("num_interpolated_points must be positive.")
    if num_classes == 0:
        return 0.0

    recall_levels = np.linspace(0.0, 1.0, num_interpolated_points)
    class_averages = []
    for class_id in range(num_classes):
        points = precision_recall_points.get(class_id, [])
        interpolated = []
        for required_recall in recall_levels:
            eligible_precisions = [
                float(precision_value)
                for recall_value, precision_value in points
                if float(recall_value) >= required_recall
            ]
            interpolated.append(
                max(eligible_precisions) if eligible_precisions else 0.0
            )
        class_averages.append(float(np.mean(interpolated)))

    return float(np.mean(class_averages))
