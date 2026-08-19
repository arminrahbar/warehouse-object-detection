"""Image augmentations with optional bounding-box transformations."""

import random

import cv2
import numpy as np


class Augmenter:
    """Apply deterministic or randomly composed image augmentations."""

    @staticmethod
    def _validate_image(image):
        if image is None:
            raise ValueError("Expected keyword argument 'image', but got None.")
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array.")
        if image.ndim not in (2, 3):
            raise ValueError("image must be a 2D grayscale or 3D color array.")
        return image

    @staticmethod
    def _copy_boxes(boxes):
        if boxes is None:
            return None

        copied = np.asarray(boxes, dtype=np.float32).copy()
        if copied.ndim != 2 or copied.shape[1] < 5:
            raise ValueError("boxes must have shape [N, >=5].")
        return copied

    @staticmethod
    def _return(image, boxes):
        return image if boxes is None else (image, boxes)

    @staticmethod
    def _inputs(kwargs):
        image = Augmenter._validate_image(kwargs.get("image"))
        boxes = Augmenter._copy_boxes(kwargs.get("boxes"))
        return image, boxes

    @staticmethod
    def horizontal_flip(**kwargs):
        image, boxes = Augmenter._inputs(kwargs)
        flipped = cv2.flip(image, 1)

        if boxes is not None:
            box_format = kwargs.get("box_format", "yolo")
            if box_format == "yolo":
                boxes[:, 1] = 1.0 - boxes[:, 1]
            elif box_format == "xyxy":
                width = image.shape[1]
                left = boxes[:, 1].copy()
                right = boxes[:, 3].copy()
                boxes[:, 1] = width - right
                boxes[:, 3] = width - left
            else:
                raise ValueError("box_format must be either 'yolo' or 'xyxy'.")

        return Augmenter._return(flipped, boxes)

    @staticmethod
    def vertical_flip(**kwargs):
        image, boxes = Augmenter._inputs(kwargs)
        flipped = cv2.flip(image, 0)

        if boxes is not None:
            box_format = kwargs.get("box_format", "yolo")
            if box_format == "yolo":
                boxes[:, 2] = 1.0 - boxes[:, 2]
            elif box_format == "xyxy":
                height = image.shape[0]
                top = boxes[:, 2].copy()
                bottom = boxes[:, 4].copy()
                boxes[:, 2] = height - bottom
                boxes[:, 4] = height - top
            else:
                raise ValueError("box_format must be either 'yolo' or 'xyxy'.")

        return Augmenter._return(flipped, boxes)

    @staticmethod
    def gaussian_blur(**kwargs):
        image, boxes = Augmenter._inputs(kwargs)
        kernel_size = int(kwargs.get("kernel_size", 5))
        sigma = float(kwargs.get("sigma", 0))

        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred = cv2.GaussianBlur(
            image,
            (kernel_size, kernel_size),
            sigma,
        )
        return Augmenter._return(blurred, boxes)

    @staticmethod
    def resize(**kwargs):
        image, boxes = Augmenter._inputs(kwargs)
        box_format = kwargs.get("box_format", "yolo")
        original_height, original_width = image.shape[:2]
        target_width = kwargs.get("width")
        target_height = kwargs.get("height")

        if target_width is None or target_height is None:
            if "scale" not in kwargs:
                raise ValueError("resize requires either width/height or scale.")
            scale = float(kwargs["scale"])
            if scale <= 0:
                raise ValueError("scale must be positive.")
            target_width = int(round(original_width * scale))
            target_height = int(round(original_height * scale))

        target_width = int(target_width)
        target_height = int(target_height)
        if target_width <= 0 or target_height <= 0:
            raise ValueError("width and height must be positive.")

        resized = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=kwargs.get("interpolation", cv2.INTER_LINEAR),
        )

        if boxes is not None:
            if box_format == "yolo":
                pass
            elif box_format == "xyxy":
                boxes[:, [1, 3]] *= target_width / original_width
                boxes[:, [2, 4]] *= target_height / original_height
            else:
                raise ValueError("box_format must be either 'yolo' or 'xyxy'.")

        return Augmenter._return(resized, boxes)

    @staticmethod
    def change_brightness(**kwargs):
        image, boxes = Augmenter._inputs(kwargs)
        alpha = float(kwargs.get("alpha", 1.0))
        beta = float(kwargs.get("beta", 30))

        adjusted = image.astype(np.float32) * alpha + beta
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return Augmenter._return(adjusted, boxes)

    @staticmethod
    def transform(**kwargs):
        image, boxes = Augmenter._inputs(kwargs)
        rng = random.Random(kwargs.get("seed"))
        box_format = kwargs.get("box_format", "yolo")
        max_transforms = max(1, int(kwargs.get("max_transforms", 3)))

        candidates = [
            "vertical_flip",
            "gaussian_blur",
            "change_brightness",
        ]
        if kwargs.get("include_resize", False):
            candidates.append("resize")

        rng.shuffle(candidates)
        selected_count = rng.randint(1, min(max_transforms, len(candidates)))

        current_image = image.copy()
        current_boxes = None if boxes is None else boxes.copy()
        for name in candidates[:selected_count]:
            operation_kwargs = dict(kwargs)
            operation_kwargs.update(
                image=current_image,
                boxes=current_boxes,
                box_format=box_format,
            )
            result = getattr(Augmenter, name)(**operation_kwargs)
            if current_boxes is None:
                current_image = result
            else:
                current_image, current_boxes = result

        return Augmenter._return(current_image, current_boxes)
