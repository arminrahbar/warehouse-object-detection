"""Rank labeled images by interpretable detector-error signals."""

from collections.abc import Mapping

import numpy as np
import pandas as pd


ERROR_COMPONENT_COLUMNS = (
    "localization_error",
    "confidence_error",
    "false_positive_rate",
    "false_negative_rate",
)


def _coerce_boxes(values, name):
    boxes = np.asarray(values, dtype=float)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name} must have shape [N, 4]")
    if not np.isfinite(boxes).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(boxes[:, 2:] < 0):
        raise ValueError(f"{name} widths and heights must be non-negative")
    return boxes


def _coerce_vector(values, name, expected_length, *, integer=False):
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error

    if vector.size == 0:
        vector = np.empty(0, dtype=float)
    if vector.ndim != 1 or len(vector) != expected_length:
        raise ValueError(
            f"{name} must be a one-dimensional vector of length "
            f"{expected_length}"
        )
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    if integer and not np.equal(vector, np.floor(vector)).all():
        raise ValueError(f"{name} must contain integer values")
    return vector.astype(int if integer else float)


def _iou_against_candidates(box, candidates):
    if len(candidates) == 0:
        return np.empty(0, dtype=float)

    left = np.maximum(box[0], candidates[:, 0])
    top = np.maximum(box[1], candidates[:, 1])
    right = np.minimum(
        box[0] + box[2],
        candidates[:, 0] + candidates[:, 2],
    )
    bottom = np.minimum(
        box[1] + box[3],
        candidates[:, 1] + candidates[:, 3],
    )

    intersection = np.maximum(0.0, right - left) * np.maximum(
        0.0,
        bottom - top,
    )
    box_area = max(0.0, box[2]) * max(0.0, box[3])
    candidate_areas = np.maximum(0.0, candidates[:, 2]) * np.maximum(
        0.0,
        candidates[:, 3],
    )
    union = box_area + candidate_areas - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def compute_image_error_components(
    pred_boxes,
    pred_classes,
    pred_confidences,
    gt_boxes,
    gt_classes,
    *,
    iou_threshold=0.5,
    confidence_floor=0.5,
):
    """Measure four bounded detector-error components for one image."""

    if not 0 < iou_threshold < 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    if not 0 <= confidence_floor < 1:
        raise ValueError("confidence_floor must be in [0, 1)")

    predictions = _coerce_boxes(pred_boxes, "pred_boxes")
    labels = _coerce_boxes(gt_boxes, "gt_boxes")
    prediction_classes = _coerce_vector(
        pred_classes,
        "pred_classes",
        len(predictions),
        integer=True,
    )
    confidences = _coerce_vector(
        pred_confidences,
        "pred_confidences",
        len(predictions),
    )
    label_classes = _coerce_vector(
        gt_classes,
        "gt_classes",
        len(labels),
        integer=True,
    )

    if np.any((confidences < 0) | (confidences > 1)):
        raise ValueError("pred_confidences must be within [0, 1]")

    label_available = np.ones(len(labels), dtype=bool)
    matched_ious = []
    matched_confidences = []

    ranked_predictions = np.argsort(-confidences, kind="stable")
    for prediction_index in ranked_predictions:
        eligible_labels = np.flatnonzero(
            label_available
            & (label_classes == prediction_classes[prediction_index])
        )
        if len(eligible_labels) == 0:
            continue

        overlaps = _iou_against_candidates(
            predictions[prediction_index],
            labels[eligible_labels],
        )
        best_position = int(np.argmax(overlaps))
        best_overlap = float(overlaps[best_position])
        if best_overlap < iou_threshold:
            continue

        matched_label = eligible_labels[best_position]
        label_available[matched_label] = False
        matched_ious.append(best_overlap)
        matched_confidences.append(float(confidences[prediction_index]))

    prediction_count = len(predictions)
    ground_truth_count = len(labels)
    matched_count = len(matched_ious)
    false_positive_count = prediction_count - matched_count
    missed_count = ground_truth_count - matched_count

    if matched_count:
        mean_iou = float(np.mean(matched_ious))
        mean_confidence = float(np.mean(matched_confidences))
        localization_error = (1.0 - mean_iou) / (1.0 - iou_threshold)
        confidence_error = (1.0 - mean_confidence) / (
            1.0 - confidence_floor
        )
    else:
        mean_iou = 0.0
        mean_confidence = 0.0
        localization_error = 0.0
        confidence_error = 0.0

    false_positive_rate = (
        false_positive_count / prediction_count if prediction_count else 0.0
    )
    false_negative_rate = (
        missed_count / ground_truth_count if ground_truth_count else 0.0
    )

    return {
        "localization_error": float(np.clip(localization_error, 0.0, 1.0)),
        "confidence_error": float(np.clip(confidence_error, 0.0, 1.0)),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "prediction_count": int(prediction_count),
        "ground_truth_count": int(ground_truth_count),
        "matched_prediction_count": int(matched_count),
        "false_positive_prediction_count": int(false_positive_count),
        "matched_gt_count": int(matched_count),
        "missed_gt_count": int(missed_count),
        "mean_matched_iou": mean_iou,
        "mean_matched_confidence": mean_confidence,
    }


def score_error_components(component_table, weights):
    """Return a scored copy of an image-level error-component table."""

    if not isinstance(component_table, pd.DataFrame):
        raise TypeError("component_table must be a pandas DataFrame")
    if not isinstance(weights, Mapping):
        raise TypeError("weights must be a mapping")

    missing = [
        component
        for component in ERROR_COMPONENT_COLUMNS
        if component not in component_table.columns
    ]
    if missing:
        raise KeyError(f"Missing error component columns: {missing}")

    component_weights = {}
    for component in ERROR_COMPONENT_COLUMNS:
        weight = float(weights.get(component, 0.0))
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(
                "Error-component weights must be finite and non-negative"
            )
        component_weights[component] = weight

    weight_sum = sum(component_weights.values())
    if weight_sum <= 0:
        raise ValueError("At least one error-component weight must be positive")

    scored = component_table.copy()
    contribution_columns = []
    for component in ERROR_COMPONENT_COLUMNS:
        values = pd.to_numeric(scored[component], errors="raise").astype(float)
        invalid = (~np.isfinite(values)) | (values < 0) | (values > 1)
        if invalid.any():
            raise ValueError(
                f"{component} values must be finite and within [0, 1]"
            )

        contribution_name = f"contribution_{component}"
        scored[contribution_name] = (
            values * component_weights[component] / weight_sum
        )
        contribution_columns.append(contribution_name)

    contributions = scored[contribution_columns]
    scored["error_score"] = contributions.sum(axis=1)
    scored["dominant_component"] = (
        contributions.idxmax(axis=1)
        .str.removeprefix("contribution_")
    )
    scored.loc[
        contributions.max(axis=1) == 0,
        "dominant_component",
    ] = "none"
    return scored
