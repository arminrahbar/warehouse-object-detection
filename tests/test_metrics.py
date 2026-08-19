import unittest

import numpy as np

from detector_service.modules.utils.metrics import (
    calculate_iou,
    calculate_map_x_point_interpolated,
    calculate_precision_recall_curve,
    match_detections,
)


def _evaluate(
    boxes,
    classes,
    objectness,
    class_scores,
    label_boxes,
    label_classes,
    *,
    eval_type="combined",
    num_classes=1,
):
    matches, counts = match_detections(
        boxes=[boxes],
        classes=[classes],
        scores=[objectness],
        cls_scores=[class_scores],
        gt_boxes=[label_boxes],
        gt_classes=[label_classes],
        map_iou_threshold=0.5,
        eval_type=eval_type,
    )
    precision, recall, thresholds = calculate_precision_recall_curve(
        matches,
        counts,
        num_classes=num_classes,
    )
    points = {
        class_id: list(zip(recall[class_id], precision[class_id]))
        for class_id in range(num_classes)
    }
    mean_ap = calculate_map_x_point_interpolated(
        points,
        num_classes=num_classes,
    )
    return matches, counts, precision, recall, thresholds, mean_ap


class IoUTests(unittest.TestCase):
    def test_identical_boxes_have_unit_overlap(self):
        self.assertEqual(calculate_iou([2, 3, 8, 6], [2, 3, 8, 6]), 1.0)

    def test_partial_overlap_uses_intersection_over_union(self):
        self.assertAlmostEqual(
            calculate_iou([0, 0, 10, 10], [5, 0, 10, 10]),
            50 / 150,
        )

    def test_disjoint_or_zero_area_boxes_have_zero_overlap(self):
        self.assertEqual(calculate_iou([0, 0, 4, 4], [10, 10, 2, 2]), 0.0)
        self.assertEqual(calculate_iou([0, 0, 0, 4], [0, 0, 0, 4]), 0.0)
        self.assertEqual(calculate_iou([0, 0, -2, 4], [0, 0, 2, 4]), 0.0)

    def test_box_shape_is_validated(self):
        with self.assertRaisesRegex(ValueError, "exactly four"):
            calculate_iou([0, 0, 4], [0, 0, 4, 4])


class DetectionMatchingTests(unittest.TestCase):
    def test_complete_miss_remains_a_false_negative(self):
        matches, counts, _, _, _, mean_ap = _evaluate(
            [],
            [],
            [],
            [],
            [[0, 0, 10, 10]],
            [0],
        )

        self.assertEqual(matches, {})
        self.assertEqual(counts, {0: 1})
        self.assertEqual(mean_ap, 0.0)

    def test_correct_detection_has_combined_score_and_perfect_ap(self):
        matches, counts, _, _, _, mean_ap = _evaluate(
            [[0, 0, 10, 10]],
            [0],
            [0.8],
            [[0.9]],
            [[0, 0, 10, 10]],
            [0],
        )

        self.assertEqual(counts, {0: 1})
        self.assertEqual(len(matches[0]), 1)
        self.assertAlmostEqual(matches[0][0][0], 0.72)
        self.assertTrue(matches[0][0][1])
        self.assertEqual(mean_ap, 1.0)

    def test_wrong_class_overlap_is_not_a_match(self):
        matches, counts, _, _, _, mean_ap = _evaluate(
            [[0, 0, 10, 10]],
            [1],
            [0.9],
            [[0.1, 0.8]],
            [[0, 0, 10, 10]],
            [0],
            num_classes=2,
        )

        self.assertEqual(counts, {0: 1})
        self.assertEqual(len(matches[1]), 1)
        self.assertAlmostEqual(matches[1][0][0], 0.72)
        self.assertFalse(matches[1][0][1])
        self.assertEqual(mean_ap, 0.0)

    def test_duplicate_detection_cannot_reuse_a_label(self):
        matches, _, precision, recall, _, _ = _evaluate(
            [[0, 0, 10, 10], [1, 1, 10, 10]],
            [0, 0],
            [0.9, 0.8],
            [[1.0], [1.0]],
            [[0, 0, 10, 10]],
            [0],
        )

        self.assertEqual(matches[0], [(0.9, True), (0.8, False)])
        np.testing.assert_allclose(precision[0], [1.0, 0.5])
        np.testing.assert_allclose(recall[0], [1.0, 1.0])

    def test_two_predictions_can_match_two_distinct_labels(self):
        matches, *_ = _evaluate(
            [[0, 0, 10, 10], [20, 0, 10, 10]],
            [0, 0],
            [0.9, 0.8],
            [[1.0], [1.0]],
            [[0, 0, 10, 10], [20, 0, 10, 10]],
            [0, 0],
        )

        self.assertEqual(matches[0], [(0.9, True), (0.8, True)])

    def test_iou_equal_to_threshold_is_a_match(self):
        matches, counts = match_detections(
            boxes=[[[0, 0, 10, 10]]],
            classes=[[0]],
            scores=[[0.9]],
            cls_scores=[[[1.0]]],
            gt_boxes=[[[0, 0, 20, 10]]],
            gt_classes=[[0]],
            map_iou_threshold=0.5,
        )

        self.assertEqual(counts, {0: 1})
        self.assertEqual(matches[0], [(0.9, True)])

    def test_matching_state_is_isolated_between_images(self):
        matches, counts = match_detections(
            boxes=[[[0, 0, 4, 4]], [[10, 10, 4, 4]]],
            classes=[[0], [0]],
            scores=[[0.9], [0.8]],
            cls_scores=[[[1.0]], [[1.0]]],
            gt_boxes=[[[0, 0, 4, 4]], [[10, 10, 4, 4]]],
            gt_classes=[[0], [0]],
            map_iou_threshold=0.5,
        )

        self.assertEqual(counts, {0: 2})
        self.assertEqual(matches[0], [(0.9, True), (0.8, True)])

    def test_equal_scores_preserve_input_order(self):
        matches, *_ = _evaluate(
            [[20, 20, 5, 5], [0, 0, 10, 10]],
            [0, 0],
            [0.8, 0.8],
            [[0.5], [0.5]],
            [[0, 0, 10, 10]],
            [0],
        )

        self.assertEqual(matches[0], [(0.4, False), (0.4, True)])

    def test_evaluation_score_modes_are_distinct(self):
        common = dict(
            boxes=[[0, 0, 10, 10]],
            classes=[1],
            objectness=[0.6],
            class_scores=[[0.2, 0.75]],
            label_boxes=[[0, 0, 10, 10]],
            label_classes=[1],
            num_classes=2,
        )

        combined = _evaluate(**common, eval_type="combined")[0][1][0][0]
        probability = _evaluate(**common, eval_type="class_scores")[0][1][0][0]
        objectness = _evaluate(**common, eval_type="objectness")[0][1][0][0]

        self.assertAlmostEqual(combined, 0.45)
        self.assertAlmostEqual(probability, 0.75)
        self.assertAlmostEqual(objectness, 0.6)

    def test_single_class_score_can_represent_predicted_class_score(self):
        matches, *_ = _evaluate(
            [[0, 0, 10, 10]],
            [5],
            [0.5],
            [[0.8]],
            [[0, 0, 10, 10]],
            [5],
            num_classes=6,
        )

        self.assertEqual(matches[5], [(0.4, True)])

    def test_mismatched_image_and_detection_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same images"):
            match_detections(
                boxes=[[]],
                classes=[],
                scores=[[]],
                cls_scores=[[]],
                gt_boxes=[[]],
                gt_classes=[[]],
                map_iou_threshold=0.5,
            )

        with self.assertRaisesRegex(ValueError, "same number of boxes"):
            match_detections(
                boxes=[[[0, 0, 2, 2]]],
                classes=[[]],
                scores=[[0.8]],
                cls_scores=[[[1.0]]],
                gt_boxes=[[]],
                gt_classes=[[]],
                map_iou_threshold=0.5,
            )

    def test_invalid_threshold_score_mode_and_classes_are_rejected(self):
        base = dict(
            boxes=[[]],
            classes=[[]],
            scores=[[]],
            cls_scores=[[]],
            gt_boxes=[[]],
            gt_classes=[[]],
        )
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            match_detections(**base, map_iou_threshold=1.1)
        with self.assertRaisesRegex(ValueError, "Unsupported eval_type"):
            match_detections(
                **base,
                map_iou_threshold=0.5,
                eval_type="confidence",
            )
        with self.assertRaisesRegex(ValueError, "Ground-truth class"):
            match_detections(
                boxes=[[]],
                classes=[[]],
                scores=[[]],
                cls_scores=[[]],
                gt_boxes=[[[0, 0, 2, 2]]],
                gt_classes=[[-1]],
                map_iou_threshold=0.5,
            )
        with self.assertRaisesRegex(ValueError, "Predicted class IDs"):
            match_detections(
                boxes=[[[0, 0, 2, 2]]],
                classes=[[-1]],
                scores=[[0.8]],
                cls_scores=[[[1.0]]],
                gt_boxes=[[]],
                gt_classes=[[]],
                map_iou_threshold=0.5,
            )
        with self.assertRaisesRegex(ValueError, "outside a class-score vector"):
            match_detections(
                boxes=[[[0, 0, 2, 2]]],
                classes=[[2]],
                scores=[[0.8]],
                cls_scores=[[[0.4, 0.6]]],
                gt_boxes=[[]],
                gt_classes=[[]],
                map_iou_threshold=0.5,
            )


class PrecisionRecallTests(unittest.TestCase):
    def test_curve_is_ranked_by_score(self):
        precision, recall, thresholds = calculate_precision_recall_curve(
            {0: [(0.2, False), (0.9, True), (0.5, True)]},
            {0: 2},
            num_classes=1,
        )

        np.testing.assert_allclose(thresholds[0], [0.9, 0.5, 0.2])
        np.testing.assert_allclose(precision[0], [1.0, 1.0, 2 / 3])
        np.testing.assert_allclose(recall[0], [0.5, 1.0, 1.0])

    def test_empty_classes_receive_empty_float_arrays(self):
        precision, recall, thresholds = calculate_precision_recall_curve(
            {},
            {0: 2},
            num_classes=2,
        )

        for class_id in range(2):
            self.assertEqual(precision[class_id].dtype, float)
            self.assertEqual(recall[class_id].dtype, float)
            self.assertEqual(thresholds[class_id].dtype, float)
            self.assertEqual(precision[class_id].size, 0)

    def test_predictions_without_labels_have_zero_recall(self):
        precision, recall, _ = calculate_precision_recall_curve(
            {0: [(0.7, False)]},
            {},
            num_classes=1,
        )

        np.testing.assert_allclose(precision[0], [0.0])
        np.testing.assert_allclose(recall[0], [0.0])

    def test_negative_class_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            calculate_precision_recall_curve({}, {}, num_classes=-1)


class InterpolatedAveragePrecisionTests(unittest.TestCase):
    def test_eleven_point_interpolation_uses_max_precision_to_the_right(self):
        score = calculate_map_x_point_interpolated(
            {0: [(0.5, 1.0), (1.0, 0.5)]},
            num_classes=1,
            num_interpolated_points=11,
        )

        self.assertAlmostEqual(score, 8.5 / 11)

    def test_empty_class_contributes_zero_to_mean(self):
        score = calculate_map_x_point_interpolated(
            {0: [(1.0, 1.0)]},
            num_classes=2,
        )

        self.assertEqual(score, 0.5)

    def test_zero_classes_returns_zero(self):
        self.assertEqual(calculate_map_x_point_interpolated({}, 0), 0.0)

    def test_invalid_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            calculate_map_x_point_interpolated({}, -1)
        with self.assertRaisesRegex(ValueError, "positive"):
            calculate_map_x_point_interpolated({}, 1, 0)


if __name__ == "__main__":
    unittest.main()
