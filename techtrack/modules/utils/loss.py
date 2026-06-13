
import itertools
import numpy as np

class Loss:
    """
    *Modified* YOLO Loss for Hard Negative Mining.

    Attributes:
        num_classes (int): Number of classes.
        iou_threshold (float): Intersection over Union (IoU) threshold.
        lambda_coord (float): Weighting factor for localization loss.
        lambda_noobj (float): Weighting factor for no object confidence loss.
    """

    def __init__(self, iou_threshold=0.5, lambda_coord=0.5, lambda_obj=0.5, lambda_noobj=0.5, lambda_cls=0.5, num_classes=20):
        """
        Initialize the Loss object with the given parameters.

        Internal Process:
        1. Stores the provided hyperparameters as instance attributes.
        2. Defines the column names for loss components to track them in results.

        Args:
            num_classes (int): Number of classes.
            lambda_coord (float): Weighting factor for localization loss.
            lambda_obj (float): Weighting factor for objectness loss.
            lambda_noobj (float): Weighting factor for no object confidence loss.
            lambda_cls (float): Weighting factor for classification loss.
        """
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.columns = [
            'total_loss',
            'loc_loss',
            'conf_loss_obj',
            'conf_loss_noobj',
            'class_loss'
        ]
        self.iou_threshold = iou_threshold

    def _calculate_iou_xyxy(self, box_a, box_b):
        """
        Calculates IoU for boxes in [x1, y1, x2, y2] format.
        """
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return float(intersection / union)

    def get_predictions(self, predictions):
        """
        Extracts bounding box coordinates, objectness scores, and class scores from predictions.

        Internal Process:
        1. Iterates over predictions to extract bounding box coordinates.
        2. Extracts objectness scores.
        3. Extracts class scores.

        Args:
            predictions (list): List of predicted bounding boxes and associated scores.

        Returns:
            tuple: (bounding boxes, objectness scores, class scores)
        """
        pred_boxes = []
        objectness_scores = []
        class_scores = []

        for image_predictions in predictions:
            for prediction in image_predictions:
                pred_boxes.append(prediction[:4])
                objectness_scores.append(prediction[4])

                scores = list(prediction[5:])

                if len(scores) < self.num_classes:
                    scores = scores + [0.0] * (self.num_classes - len(scores))
                elif len(scores) > self.num_classes:
                    scores = scores[:self.num_classes]

                class_scores.append(scores)

        if len(pred_boxes) == 0:
            return (
                np.empty((0, 4), dtype=float),
                np.empty((0,), dtype=float),
                np.empty((0, self.num_classes), dtype=float)
            )

        return (
            np.asarray(pred_boxes, dtype=float),
            np.asarray(objectness_scores, dtype=float),
            np.asarray(class_scores, dtype=float)
        )

    def get_annotations(self, annotations):
        """
        Extract ground truth bounding boxes and class IDs from annotations.

        Internal Process:
        1. Iterates over annotations to extract bounding box coordinates.
        2. Extracts the corresponding class labels.

        Args:
            annotations (list): List of ground truth annotations.

        Returns:
            tuple: (ground truth bounding boxes, class labels)
        """
        gt_boxes = []
        gt_class_ids = []

        for annotation in annotations:
            gt_class_ids.append(annotation[0])
            gt_boxes.append(annotation[1:5])

        if len(gt_boxes) == 0:
            return (
                np.empty((0, 4), dtype=float),
                np.empty((0,), dtype=int)
            )

        return (
            np.asarray(gt_boxes, dtype=float),
            np.asarray(gt_class_ids, dtype=int)
        )

    def compute(self, predictions, annotations):
        """
        Compute the YOLO loss components.

        Internal Process:
        1. Extracts predictions and annotations of a single image/frame.
        2. Iterates through annotations to compute localization, confidence, and class loss.
        3. Computes total loss using predefined weighting factors.

        Args:
            predictions (list): List of predictions of a single image.
            annotations (list): List of ground truth annotations of a single image.

        Returns:
            dict: Dictionary containing the computed loss components.
        """
        loc_loss = 0.0 # localization loss
        class_loss = 0.0 # classification loss
        conf_loss_obj = 0.0 # with object (or confidence) loss
        conf_loss_noobj = 0.0 # no object (or confidence) loss
        total_loss = 0.0 # aggregate loss including loc_loss, class_loss, conf_loss_obj, etc.

        # TASK: Complete this method to compute the Loss function.
        #         This method calculates the localization, objectness
        #         (or confidence) and classification loss.
        #         This method will be called in the HardNegativeMiner class.
        #         ----------------------------------------------------------
        #         HINT: For simplicity complete use get_predictions(), get_annotations().
        #         You may add class methods to improve the readability of this code.

        pred_box, objectness_score, class_scores = self.get_predictions(predictions)
        gt_box, gt_class_id = self.get_annotations(annotations)

        if pred_box.shape[0] == 0:
            total_loss = 0.0
            return {
                "total_loss": total_loss,
                "loc_loss": loc_loss,
                "conf_loss_obj": conf_loss_obj,
                "conf_loss_noobj": conf_loss_noobj,
                "class_loss": class_loss
            }

        if gt_box.shape[0] == 0:
            conf_loss_noobj = float(np.sum(objectness_score ** 2))
            total_loss = self.lambda_noobj * conf_loss_noobj
            return {
                "total_loss": total_loss,
                "loc_loss": loc_loss,
                "conf_loss_obj": conf_loss_obj,
                "conf_loss_noobj": conf_loss_noobj,
                "class_loss": class_loss
            }

        for pred_idx in range(pred_box.shape[0]):
            current_pred_box = pred_box[pred_idx]
            current_objectness = float(objectness_score[pred_idx])
            current_class_scores = class_scores[pred_idx]

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx in range(gt_box.shape[0]):
                iou = self._calculate_iou_xyxy(current_pred_box, gt_box[gt_idx])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0 and best_iou >= self.iou_threshold:
                target_box = gt_box[best_gt_idx]
                target_class = int(gt_class_id[best_gt_idx])

                loc_loss += float(np.sum((current_pred_box - target_box) ** 2))
                conf_loss_obj += float((1.0 - current_objectness) ** 2)

                if 0 <= target_class < len(current_class_scores):
                    correct_class_score = float(current_class_scores[target_class])
                else:
                    correct_class_score = 0.0

                class_loss += float((1.0 - correct_class_score) ** 2)
            else:
                conf_loss_noobj += float(current_objectness ** 2)

        total_loss = (
            self.lambda_coord * loc_loss
            + self.lambda_obj * conf_loss_obj
            + self.lambda_noobj * conf_loss_noobj
            + self.lambda_cls * class_loss
        )

        return {
            "total_loss": total_loss,
            "loc_loss": loc_loss,
            "conf_loss_obj": conf_loss_obj,
            "conf_loss_noobj": conf_loss_noobj,
            "class_loss": class_loss
        }
