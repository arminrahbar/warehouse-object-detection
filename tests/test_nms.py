import unittest

import numpy as np

from detector_service.modules.inference.nms import NMS


class ClassAwareNMSTests(unittest.TestCase):
    def setUp(self):
        self.nms = NMS(score_threshold=0.45, nms_iou_threshold=0.35)

    def test_combined_confidence_is_objectness_times_predicted_probability(self):
        combined = NMS.confidence_scores(
            class_ids=[1, 0],
            scores=[0.80, 0.50],
            class_scores=[[0.10, 0.75], [0.90, 0.10]],
        )

        np.testing.assert_allclose(combined, [0.60, 0.45])

    def test_threshold_is_inclusive(self):
        result = self.nms.filter(
            bboxes=[[2, 4, 8, 6]],
            class_ids=[0],
            scores=[0.50],
            class_scores=[[0.90]],
        )

        self.assertEqual(result[0], [[2, 4, 8, 6]])

    def test_same_class_overlap_keeps_higher_combined_confidence(self):
        boxes, classes, objectness, probabilities = self.nms.filter(
            bboxes=[[10, 10, 20, 20], [12, 12, 20, 20]],
            class_ids=[2, 2],
            scores=[0.95, 0.80],
            class_scores=[
                [0.05, 0.05, 0.50],
                [0.05, 0.05, 0.90],
            ],
        )

        self.assertEqual(boxes, [[12, 12, 20, 20]])
        self.assertEqual(classes, [2])
        self.assertEqual(objectness, [0.80])
        self.assertEqual(probabilities, [[0.05, 0.05, 0.90]])

    def test_identical_boxes_with_different_classes_do_not_compete(self):
        boxes, classes, _, _ = self.nms.filter(
            bboxes=[[4, 4, 12, 12], [4, 4, 12, 12]],
            class_ids=[0, 1],
            scores=[0.90, 0.90],
            class_scores=[[0.80, 0.20], [0.20, 0.70]],
        )

        self.assertEqual(boxes, [[4, 4, 12, 12], [4, 4, 12, 12]])
        self.assertEqual(classes, [0, 1])

    def test_retained_detections_are_globally_ranked(self):
        boxes, classes, _, _ = self.nms.filter(
            bboxes=[[0, 0, 5, 5], [20, 20, 5, 5], [40, 40, 5, 5]],
            class_ids=[1, 0, 1],
            scores=[0.70, 0.95, 0.80],
            class_scores=[[0.10, 0.80], [0.70, 0.10], [0.10, 0.90]],
        )

        self.assertEqual(boxes, [[40, 40, 5, 5], [20, 20, 5, 5], [0, 0, 5, 5]])
        self.assertEqual(classes, [1, 0, 1])

    def test_candidate_below_combined_threshold_is_removed(self):
        boxes, classes, scores, class_scores = self.nms.filter(
            bboxes=[[0, 0, 5, 5]],
            class_ids=[0],
            scores=[0.90],
            class_scores=[[0.49]],
        )

        self.assertEqual((boxes, classes, scores, class_scores), ([], [], [], []))

    def test_empty_input_produces_four_empty_collections(self):
        self.assertEqual(self.nms.filter([], [], [], []), ([], [], [], []))

    def test_numpy_backed_inputs_are_supported(self):
        boxes, classes, objectness, probabilities = self.nms.filter(
            bboxes=np.asarray([[3, 5, 7, 9]]),
            class_ids=np.asarray([0]),
            scores=np.asarray([0.75]),
            class_scores=np.asarray([[0.80]]),
        )

        np.testing.assert_array_equal(boxes[0], [3, 5, 7, 9])
        self.assertEqual(classes, [0])
        self.assertEqual(objectness, [0.75])
        np.testing.assert_array_equal(probabilities[0], [0.80])

    def test_collection_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            self.nms.filter(
                bboxes=[[0, 0, 5, 5]],
                class_ids=[0],
                scores=[0.8],
                class_scores=[],
            )

    def test_negative_box_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            self.nms.filter(
                bboxes=[[0, 0, -1, 5]],
                class_ids=[0],
                scores=[0.8],
                class_scores=[[0.9]],
            )

    def test_missing_predicted_class_probability_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no entry for class 2"):
            self.nms.filter(
                bboxes=[[0, 0, 5, 5]],
                class_ids=[2],
                scores=[0.8],
                class_scores=[[0.6, 0.4]],
            )

    def test_invalid_probability_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            self.nms.filter(
                bboxes=[[0, 0, 5, 5]],
                class_ids=[0],
                scores=[0.8],
                class_scores=[[1.2]],
            )

    def test_invalid_constructor_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "score_threshold"):
            NMS(score_threshold=np.inf, nms_iou_threshold=0.3)


if __name__ == "__main__":
    unittest.main()
