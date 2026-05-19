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
        #
        # Return these variables in order as described in Line 46-50:
        # return filtered_bboxes, filtered_class_ids, filtered_scores, filtered_class_scores
