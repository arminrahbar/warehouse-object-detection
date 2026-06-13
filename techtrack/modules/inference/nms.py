import numpy as np
from typing import List, Tuple


class NMS:
    """
    Implements Non-Maximum Suppression (NMS) to filter redundant bounding boxes 
    in object detection.

    This class takes bounding boxes, confidence scores, and class IDs and applies 
    NMS to retain only the most relevant bounding boxes based on confidence scores 
    and Intersection over Union (IoU) thresholding.
    """

    def __init__(self, score_threshold: float, nms_iou_threshold: float) -> None:
        """
        Initializes the NMS filter with confidence and IoU thresholds.

        :param score_threshold: The minimum confidence score required to retain a bounding box.
        :param nms_iou_threshold: The Intersection over Union (IoU) threshold for non-maximum suppression.

        :ivar self.score_threshold: The threshold below which detections are discarded.
        :ivar self.nms_iou_threshold: The IoU threshold that determines whether two boxes 
                                      are considered redundant.
        """
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def _calculate_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Calculates IoU between one bounding box and an array of bounding boxes.

        :param box: A single box in [x, y, width, height] format.
        :param boxes: Multiple boxes in [x, y, width, height] format.

        :return: NumPy array of IoU values.
        """
        if boxes.size == 0:
            return np.array([])

        box_x1 = box[0]
        box_y1 = box[1]
        box_x2 = box[0] + box[2]
        box_y2 = box[1] + box[3]

        boxes_x1 = boxes[:, 0]
        boxes_y1 = boxes[:, 1]
        boxes_x2 = boxes[:, 0] + boxes[:, 2]
        boxes_y2 = boxes[:, 1] + boxes[:, 3]

        inter_x1 = np.maximum(box_x1, boxes_x1)
        inter_y1 = np.maximum(box_y1, boxes_y1)
        inter_x2 = np.minimum(box_x2, boxes_x2)
        inter_y2 = np.minimum(box_y2, boxes_y2)

        inter_width = np.maximum(0, inter_x2 - inter_x1)
        inter_height = np.maximum(0, inter_y2 - inter_y1)
        intersection_area = inter_width * inter_height

        box_area = max(0, box[2]) * max(0, box[3])
        boxes_area = np.maximum(0, boxes[:, 2]) * np.maximum(0, boxes[:, 3])

        union_area = box_area + boxes_area - intersection_area

        return np.where(union_area > 0, intersection_area / union_area, 0.0)

    def filter(
        self,
        bboxes: List[List[int]],
        class_ids: List[int],
        scores: List[float],
        class_scores: List[float],
    ) -> Tuple[List[List[int]], List[int], List[float], List[float]]:
        """
        Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

        :param bboxes: A list of bounding boxes, where each box is represented as 
                       [x, y, width, height]. (x, y) is the top-left corner.
        :param class_ids: A list of class IDs corresponding to each bounding box.
        :param scores: A list of confidence scores for each bounding box.
        :param class_scores: A list of class-specific scores for each detection.

        :return: A tuple containing:
            - **filtered_bboxes (List[List[int]])**: The final bounding boxes after NMS.
            - **filtered_class_ids (List[int])**: The class IDs of retained bounding boxes.
            - **filtered_scores (List[float])**: The confidence scores of retained bounding boxes.
            - **filtered_class_scores (List[float])**: The class-specific scores of retained boxes.

        **How NMS Works:**
        - The function selects the bounding box with the highest confidence.
        - It suppresses any boxes that have a high IoU (overlapping area) with this selected box.
        - This process is repeated until all valid boxes are retained.

        **Example Usage:**
        ```python
        nms_processor = NMS(score_threshold=0.5, nms_iou_threshold=0.4)
        final_bboxes, final_class_ids, final_scores, final_class_scores = nms_processor.filter(
            bboxes, class_ids, scores, class_scores
        )
        ```
        """

        # TASK: Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.
        #         DO NOT USE **cv2.dnn.NMSBoxes()** for this Assignment. For Assignment 2, you will be
        #         permitted to use this function.
        if not bboxes or not class_ids or not scores or not class_scores:
            return [], [], [], []

        boxes_array = np.array(bboxes, dtype=float)
        scores_array = np.array(scores, dtype=float)

        valid_indices = np.where(scores_array > self.score_threshold)[0]

        if len(valid_indices) == 0:
            return [], [], [], []

        sorted_indices = valid_indices[np.argsort(scores_array[valid_indices])[::-1]]
        selected_indices = []

        while len(sorted_indices) > 0:
            current_index = int(sorted_indices[0])
            selected_indices.append(current_index)

            if len(sorted_indices) == 1:
                break

            remaining_indices = sorted_indices[1:]
            ious = self._calculate_iou(
                boxes_array[current_index],
                boxes_array[remaining_indices]
            )

            keep_mask = ious <= self.nms_iou_threshold
            sorted_indices = remaining_indices[keep_mask]

        filtered_bboxes = [bboxes[i] for i in selected_indices]
        filtered_class_ids = [class_ids[i] for i in selected_indices]
        filtered_scores = [scores[i] for i in selected_indices]
        filtered_class_scores = [class_scores[i] for i in selected_indices]

        # Return these variables in order as described in Line 46-50:
        return filtered_bboxes, filtered_class_ids, filtered_scores, filtered_class_scores