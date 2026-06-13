import numpy as np
from sklearn.preprocessing import label_binarize


def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    """
    ax, ay, aw, ah = [float(v) for v in boxA]
    bx, by, bw, bh = [float(v) for v in boxB]

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


def _make_score_vector(score, cls_score, pred_class, num_classes, eval_type):
    cls_score = np.asarray(cls_score, dtype=float)

    if cls_score.ndim == 0:
        score_vector = np.zeros(num_classes, dtype=float)
        if 0 <= int(pred_class) < num_classes:
            score_vector[int(pred_class)] = float(cls_score)
    else:
        score_vector = cls_score.astype(float).copy()

        if len(score_vector) < num_classes:
            padded = np.zeros(num_classes, dtype=float)
            padded[:len(score_vector)] = score_vector
            score_vector = padded
        elif len(score_vector) > num_classes:
            score_vector = score_vector[:num_classes]

    if eval_type == "class_scores":
        return score_vector

    if eval_type == "combined":
        return score_vector * float(score)

    if eval_type == "objectness":
        objectness_vector = np.zeros(num_classes, dtype=float)
        if 0 <= int(pred_class) < num_classes:
            objectness_vector[int(pred_class)] = float(score)
        return objectness_vector

    raise ValueError(f"Unsupported eval_type: {eval_type}")


def match_detections(boxes, classes, scores, cls_scores, gt_boxes, gt_classes, map_iou_threshold, eval_type="class_scores"):
    """
    Evaluate detections by matching predicted bounding boxes with ground truth boxes and generate
    corresponding true labels and prediction scores for further evaluation.
    """
    y_true = []
    pred_scores = []

    num_classes_candidates = []

    for image_cls_scores in cls_scores:
        arr = np.asarray(image_cls_scores)
        if arr.ndim == 2 and arr.shape[1] > 0:
            num_classes_candidates.append(arr.shape[1])
        elif arr.ndim == 1 and arr.size > 0:
            num_classes_candidates.append(arr.size)

    for image_classes in classes:
        for class_id in image_classes:
            num_classes_candidates.append(int(class_id) + 1)

    for image_gt_classes in gt_classes:
        for class_id in image_gt_classes:
            num_classes_candidates.append(int(class_id) + 1)

    num_classes = max(num_classes_candidates) if num_classes_candidates else 0
    num_images = max(len(boxes), len(gt_boxes))

    for image_idx in range(num_images):
        image_boxes = boxes[image_idx] if image_idx < len(boxes) else []
        image_classes = classes[image_idx] if image_idx < len(classes) else []
        image_scores = scores[image_idx] if image_idx < len(scores) else []
        image_cls_scores = cls_scores[image_idx] if image_idx < len(cls_scores) else []

        image_gt_boxes = gt_boxes[image_idx] if image_idx < len(gt_boxes) else []
        image_gt_classes = gt_classes[image_idx] if image_idx < len(gt_classes) else []

        matched_gt_indices = set()

        detection_indices = list(range(len(image_boxes)))
        detection_indices.sort(
            key=lambda i: float(image_scores[i]) if i < len(image_scores) else 0.0,
            reverse=True
        )

        for detection_idx in detection_indices:
            detection_box = image_boxes[detection_idx]
            detection_class = int(image_classes[detection_idx]) if detection_idx < len(image_classes) else 0
            detection_score = float(image_scores[detection_idx]) if detection_idx < len(image_scores) else 0.0

            if detection_idx < len(image_cls_scores):
                detection_cls_score = image_cls_scores[detection_idx]
            else:
                detection_cls_score = np.zeros(num_classes, dtype=float)

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(image_gt_boxes):
                if gt_idx in matched_gt_indices:
                    continue

                iou = calculate_iou(detection_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            score_vector = _make_score_vector(
                detection_score,
                detection_cls_score,
                detection_class,
                num_classes,
                eval_type
            )

            if best_gt_idx >= 0 and best_iou >= map_iou_threshold:
                y_true.append(int(image_gt_classes[best_gt_idx]))
                pred_scores.append(score_vector)
                matched_gt_indices.add(best_gt_idx)
            else:
                y_true.append(-1)
                pred_scores.append(score_vector)

        for gt_idx, gt_class in enumerate(image_gt_classes):
            if gt_idx not in matched_gt_indices:
                y_true.append(int(gt_class))
                pred_scores.append(np.zeros(num_classes, dtype=float))

    return y_true, pred_scores


def calculate_precision_recall_curve(y_true, pred_scores, num_classes=20):
    """
    Compute the precision-recall curve for each class in a multi-class classification task.
    """
    if len(y_true) == 0:
        precision = {class_idx: [] for class_idx in range(num_classes)}
        recall = {class_idx: [] for class_idx in range(num_classes)}
        thresholds = {class_idx: np.array([]) for class_idx in range(num_classes)}
        return precision, recall, thresholds

    y_true = np.asarray(y_true)
    pred_scores = np.asarray(pred_scores, dtype=float)

    if pred_scores.ndim == 1:
        pred_scores = pred_scores.reshape(-1, 1)

    if pred_scores.shape[1] < num_classes:
        padded_scores = np.zeros((pred_scores.shape[0], num_classes), dtype=float)
        padded_scores[:, :pred_scores.shape[1]] = pred_scores
        pred_scores = padded_scores
    elif pred_scores.shape[1] > num_classes:
        pred_scores = pred_scores[:, :num_classes]

    y_true_bin = np.zeros((len(y_true), num_classes), dtype=int)

    for row_idx, label in enumerate(y_true):
        label = int(label)
        if 0 <= label < num_classes:
            y_true_bin[row_idx, label] = 1

    precision = {}
    recall = {}
    thresholds = {}

    for class_idx in range(num_classes):
        class_scores = pred_scores[:, class_idx]

        sorted_indices = np.argsort(class_scores)[::-1]
        sorted_scores = class_scores[sorted_indices]
        sorted_true = y_true_bin[sorted_indices, class_idx]

        total_positives = int(np.sum(y_true_bin[:, class_idx]))

        class_precision = []
        class_recall = []
        class_thresholds = []

        true_positives = 0
        false_positives = 0

        for score, true_value in zip(sorted_scores, sorted_true):
            if int(true_value) == 1:
                true_positives += 1
            else:
                false_positives += 1

            denominator = true_positives + false_positives

            if denominator > 0:
                class_precision.append(true_positives / denominator)
            else:
                class_precision.append(0.0)

            if total_positives > 0:
                class_recall.append(true_positives / total_positives)
            else:
                class_recall.append(0.0)

            class_thresholds.append(float(score))

        precision[class_idx] = class_precision
        recall[class_idx] = class_recall
        thresholds[class_idx] = np.asarray(class_thresholds, dtype=float)

    return precision, recall, thresholds


def calculate_map_x_point_interpolated(precision_recall_points, num_classes, num_interpolated_points=11):
    """
    Calculate the Mean Average Precision (mAP) using x-point interpolation for multi-class object detection tasks.
    """
    if num_classes == 0:
        return 0.0

    mean_average_precisions = []
    recall_thresholds = np.linspace(0.0, 1.0, num_interpolated_points)

    for class_idx in range(num_classes):
        points = precision_recall_points.get(class_idx, [])

        interpolated_precisions = []

        for recall_threshold in recall_thresholds:
            possible_precisions = [
                precision
                for recall_value, precision in points
                if recall_value >= recall_threshold
            ]

            if possible_precisions:
                interpolated_precisions.append(max(possible_precisions))
            else:
                interpolated_precisions.append(0.0)

        mean_average_precisions.append(float(np.mean(interpolated_precisions)))

    return float(np.mean(mean_average_precisions))