import random
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


try:
    import cv2
except ModuleNotFoundError:
    cv2 = types.ModuleType("cv2")
    sys.modules["cv2"] = cv2


def _flip(image, code):
    axis = 1 if code == 1 else 0
    return np.flip(image, axis=axis).copy()


def _resize(image, size, interpolation=None):
    width, height = size
    shape = (height, width) + image.shape[2:]
    return np.zeros(shape, dtype=image.dtype)


# Keep this stub sufficient for every test module when OpenCV is unavailable.
cv2.FONT_HERSHEY_SIMPLEX = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)
cv2.INTER_LINEAR = getattr(cv2, "INTER_LINEAR", 1)
cv2.VideoCapture = getattr(cv2, "VideoCapture", Mock(name="VideoCapture"))
cv2.dnn = getattr(cv2, "dnn", types.SimpleNamespace())
cv2.rectangle = getattr(
    cv2,
    "rectangle",
    Mock(side_effect=lambda frame, *args, **kwargs: frame),
)
cv2.putText = getattr(
    cv2,
    "putText",
    Mock(side_effect=lambda frame, *args, **kwargs: frame),
)
cv2.imwrite = getattr(cv2, "imwrite", Mock(return_value=True))
cv2.flip = getattr(cv2, "flip", _flip)
cv2.GaussianBlur = getattr(
    cv2,
    "GaussianBlur",
    lambda image, kernel, sigma: image.copy(),
)
cv2.resize = getattr(cv2, "resize", _resize)

from detector_service.modules.rectification import augmentation as augmentation_module
from detector_service.modules.rectification.augmentation import Augmenter


class AugmenterTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)

    def test_image_validation_rejects_missing_wrong_type_and_wrong_rank(self):
        invalid_cases = (
            (None, ValueError, "got None"),
            ([[1, 2]], TypeError, "NumPy array"),
            (np.asarray([1, 2]), ValueError, "2D grayscale or 3D color"),
            (np.zeros((1, 2, 3, 4)), ValueError, "2D grayscale or 3D color"),
        )

        for image, error_type, message in invalid_cases:
            with self.subTest(image=image), self.assertRaisesRegex(
                error_type,
                message,
            ):
                Augmenter.change_brightness(image=image)

    def test_boxes_must_be_a_two_dimensional_array_with_five_columns(self):
        for boxes in ([0, 1, 2, 3, 4], [[0, 1, 2, 3]]):
            with self.subTest(boxes=boxes), self.assertRaisesRegex(
                ValueError,
                r"shape \[N, >=5\]",
            ):
                Augmenter.horizontal_flip(image=self.image, boxes=boxes)

    def test_horizontal_flip_updates_normalized_yolo_x_center(self):
        boxes = np.asarray([[2, 0.2, 0.4, 0.3, 0.5]], dtype=np.float64)

        flipped, transformed = Augmenter.horizontal_flip(
            image=self.image,
            boxes=boxes,
        )

        np.testing.assert_array_equal(flipped, np.flip(self.image, axis=1))
        np.testing.assert_allclose(transformed, [[2, 0.8, 0.4, 0.3, 0.5]])
        np.testing.assert_allclose(boxes, [[2, 0.2, 0.4, 0.3, 0.5]])
        self.assertEqual(transformed.dtype, np.float32)

    def test_horizontal_flip_updates_absolute_xyxy_coordinates(self):
        _, transformed = Augmenter.horizontal_flip(
            image=self.image,
            boxes=[[3, 1, 2, 4, 6]],
            box_format="xyxy",
        )

        np.testing.assert_allclose(transformed, [[3, 6, 2, 9, 6]])

    def test_vertical_flip_updates_yolo_and_xyxy_coordinates(self):
        _, yolo = Augmenter.vertical_flip(
            image=self.image,
            boxes=[[1, 0.2, 0.25, 0.4, 0.3]],
        )
        _, xyxy = Augmenter.vertical_flip(
            image=self.image,
            boxes=[[1, 1, 1, 4, 5]],
            box_format="xyxy",
        )

        np.testing.assert_allclose(yolo, [[1, 0.2, 0.75, 0.4, 0.3]])
        np.testing.assert_allclose(xyxy, [[1, 1, 3, 4, 7]])

    def test_invalid_box_format_is_rejected_when_boxes_are_present(self):
        with self.assertRaisesRegex(ValueError, "either 'yolo' or 'xyxy'"):
            Augmenter.vertical_flip(
                image=self.image,
                boxes=[[1, 0, 0, 1, 1]],
                box_format="corners",
            )

    def test_image_only_calls_return_an_array_not_a_tuple(self):
        result = Augmenter.horizontal_flip(image=self.image)

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.flip(self.image, axis=1))

    def test_even_blur_kernel_is_promoted_to_next_odd_size(self):
        blurred = np.full_like(self.image, 7)
        source_boxes = np.asarray([[1, 0.5, 0.5, 0.2, 0.2]])

        with patch.object(
            augmentation_module.cv2,
            "GaussianBlur",
            return_value=blurred,
        ) as gaussian_blur:
            result_image, result_boxes = Augmenter.gaussian_blur(
                image=self.image,
                boxes=source_boxes,
                kernel_size=4,
                sigma=1.25,
            )

        gaussian_blur.assert_called_once_with(self.image, (5, 5), 1.25)
        self.assertIs(result_image, blurred)
        np.testing.assert_allclose(result_boxes, source_boxes)
        self.assertIsNot(result_boxes, source_boxes)

    def test_nonpositive_blur_kernel_is_rejected(self):
        for kernel_size in (0, -3):
            with self.subTest(kernel_size=kernel_size), self.assertRaisesRegex(
                ValueError,
                "kernel_size must be positive",
            ):
                Augmenter.gaussian_blur(
                    image=self.image,
                    kernel_size=kernel_size,
                )

    def test_resize_scales_xyxy_boxes_by_each_axis(self):
        resized = np.zeros((4, 20, 3), dtype=np.uint8)

        with patch.object(
            augmentation_module.cv2,
            "resize",
            return_value=resized,
        ) as resize:
            result_image, result_boxes = Augmenter.resize(
                image=self.image,
                boxes=[[4, 1, 2, 4, 6]],
                box_format="xyxy",
                width=20,
                height=4,
                interpolation=17,
            )

        resize.assert_called_once_with(self.image, (20, 4), interpolation=17)
        self.assertIs(result_image, resized)
        np.testing.assert_allclose(result_boxes, [[4, 2, 1, 8, 3]])

    def test_resize_leaves_normalized_yolo_boxes_unchanged(self):
        source = np.asarray([[0, 0.25, 0.75, 0.2, 0.4]], dtype=np.float32)

        _, transformed = Augmenter.resize(
            image=self.image,
            boxes=source,
            width=5,
            height=16,
        )

        np.testing.assert_array_equal(transformed, source)
        self.assertIsNot(transformed, source)

    def test_resize_scale_computes_both_target_dimensions(self):
        result = Augmenter.resize(image=self.image, width=999, scale=0.5)

        self.assertEqual(result.shape, (4, 5, 3))

    def test_resize_validates_dimension_configuration(self):
        invalid_cases = (
            ({}, "either width/height or scale"),
            ({"scale": 0}, "scale must be positive"),
            ({"width": 0, "height": 5}, "width and height must be positive"),
        )

        for arguments, message in invalid_cases:
            with self.subTest(arguments=arguments), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                Augmenter.resize(image=self.image, **arguments)

    def test_brightness_uses_linear_adjustment_clipping_and_uint8_output(self):
        image = np.asarray([[0, 100, 250]], dtype=np.uint8)

        adjusted = Augmenter.change_brightness(
            image=image,
            alpha=1.5,
            beta=-10,
        )

        np.testing.assert_array_equal(adjusted, [[0, 140, 255]])
        self.assertEqual(adjusted.dtype, np.uint8)

    def test_transform_is_seeded_deterministic_and_does_not_touch_global_rng(self):
        boxes = np.asarray([[1, 0.3, 0.2, 0.25, 0.4]], dtype=np.float32)
        random.seed(9876)
        global_state = random.getstate()

        first_image, first_boxes = Augmenter.transform(
            image=self.image,
            boxes=boxes,
            seed=42,
            max_transforms=3,
            kernel_size=3,
            alpha=0.9,
            beta=5,
        )
        second_image, second_boxes = Augmenter.transform(
            image=self.image,
            boxes=boxes,
            seed=42,
            max_transforms=3,
            kernel_size=3,
            alpha=0.9,
            beta=5,
        )

        np.testing.assert_array_equal(first_image, second_image)
        np.testing.assert_array_equal(first_boxes, second_boxes)
        np.testing.assert_allclose(boxes, [[1, 0.3, 0.2, 0.25, 0.4]])
        self.assertEqual(random.getstate(), global_state)

    def test_random_transform_excludes_horizontal_flip_and_clamps_limit_to_one(self):
        calls = []

        def operation(name):
            def apply(**kwargs):
                calls.append(name)
                if kwargs["boxes"] is None:
                    return kwargs["image"]
                return kwargs["image"], kwargs["boxes"]

            return apply

        with patch.object(
            Augmenter,
            "vertical_flip",
            side_effect=operation("vertical"),
        ), patch.object(
            Augmenter,
            "gaussian_blur",
            side_effect=operation("blur"),
        ), patch.object(
            Augmenter,
            "change_brightness",
            side_effect=operation("brightness"),
        ), patch.object(
            Augmenter,
            "horizontal_flip",
            side_effect=operation("horizontal"),
        ):
            Augmenter.transform(
                image=self.image,
                seed=10,
                max_transforms=0,
            )

        self.assertEqual(len(calls), 1)
        self.assertNotIn("horizontal", calls)

    def test_resize_joins_random_candidates_only_when_requested(self):
        def identity(**kwargs):
            if kwargs["boxes"] is None:
                return kwargs["image"]
            return kwargs["image"], kwargs["boxes"]

        with patch.object(Augmenter, "vertical_flip", side_effect=identity), \
            patch.object(Augmenter, "gaussian_blur", side_effect=identity), \
            patch.object(Augmenter, "change_brightness", side_effect=identity), \
            patch.object(Augmenter, "resize", side_effect=identity) as resize:
            for seed in range(100):
                Augmenter.transform(
                    image=self.image,
                    seed=seed,
                    max_transforms=4,
                    include_resize=False,
                )
            resize.assert_not_called()

            for seed in range(100):
                Augmenter.transform(
                    image=self.image,
                    seed=seed,
                    max_transforms=4,
                    include_resize=True,
                    width=5,
                    height=4,
                )

        self.assertGreater(resize.call_count, 0)


if __name__ == "__main__":
    unittest.main()
