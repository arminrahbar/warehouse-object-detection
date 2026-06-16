import cv2
import numpy as np
import random
from typing import Optional, Tuple, Union


class Augmenter:
    """
    Dataset augmentation utilities for TechTrack.

    Supported transformations:
    - horizontal_flip
    - vertical_flip
    - gaussian_blur
    - resize
    - change_brightness
    - transform

    The methods support image-only use and optional bounding-box transformation.

    Bounding-box formats supported:
    - "yolo": rows shaped [class_id, x_center, y_center, width, height]
              where coordinates are normalized to [0, 1]
    - "xyxy": rows shaped [class_id, x1, y1, x2, y2]
              where coordinates are pixel coordinates

    By default, methods return only the transformed image.
    If boxes are supplied through `boxes=...`, methods return `(image, boxes)`.
    """

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
        arr = np.asarray(boxes, dtype=np.float32).copy()
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError("boxes must have shape [N, >=5].")
        return arr

    @staticmethod
    def _return(image, boxes):
        if boxes is None:
            return image
        return image, boxes

    @staticmethod
    def horizontal_flip(**kwargs):
        """
        Horizontally flip the image.

        Parameters:
            image: NumPy image array.
            boxes: Optional bounding boxes.
            box_format: "yolo" or "xyxy".

        Returns:
            image if boxes are not supplied;
            otherwise (image, boxes).
        """
        image = Augmenter._validate_image(kwargs.get("image"))
        boxes = Augmenter._copy_boxes(kwargs.get("boxes"))
        box_format = kwargs.get("box_format", "yolo")

        flipped = cv2.flip(image, 1)

        if boxes is not None:
            h, w = image.shape[:2]

            if box_format == "yolo":
                # [class_id, x_center, y_center, width, height], normalized
                boxes[:, 1] = 1.0 - boxes[:, 1]

            elif box_format == "xyxy":
                # [class_id, x1, y1, x2, y2], pixel coordinates
                x1 = boxes[:, 1].copy()
                x2 = boxes[:, 3].copy()
                boxes[:, 1] = w - x2
                boxes[:, 3] = w - x1

            else:
                raise ValueError("box_format must be either 'yolo' or 'xyxy'.")

        return Augmenter._return(flipped, boxes)

    @staticmethod
    def vertical_flip(**kwargs):
        """
        Vertically flip the image.

        Parameters:
            image: NumPy image array.
            boxes: Optional bounding boxes.
            box_format: "yolo" or "xyxy".

        Returns:
            image if boxes are not supplied;
            otherwise (image, boxes).
        """
        image = Augmenter._validate_image(kwargs.get("image"))
        boxes = Augmenter._copy_boxes(kwargs.get("boxes"))
        box_format = kwargs.get("box_format", "yolo")

        flipped = cv2.flip(image, 0)

        if boxes is not None:
            h, w = image.shape[:2]

            if box_format == "yolo":
                # [class_id, x_center, y_center, width, height], normalized
                boxes[:, 2] = 1.0 - boxes[:, 2]

            elif box_format == "xyxy":
                # [class_id, x1, y1, x2, y2], pixel coordinates
                y1 = boxes[:, 2].copy()
                y2 = boxes[:, 4].copy()
                boxes[:, 2] = h - y2
                boxes[:, 4] = h - y1

            else:
                raise ValueError("box_format must be either 'yolo' or 'xyxy'.")

        return Augmenter._return(flipped, boxes)

    @staticmethod
    def gaussian_blur(**kwargs):
        """
        Apply Gaussian blur to the image.

        Parameters:
            image: NumPy image array.
            boxes: Optional bounding boxes. Boxes are unchanged.
            kernel_size: Odd integer kernel size. Default: 5.
            sigma: Gaussian sigma. Default: 0.

        Returns:
            image if boxes are not supplied;
            otherwise (image, boxes).
        """
        image = Augmenter._validate_image(kwargs.get("image"))
        boxes = Augmenter._copy_boxes(kwargs.get("boxes"))

        kernel_size = int(kwargs.get("kernel_size", 5))
        sigma = float(kwargs.get("sigma", 0))

        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        return Augmenter._return(blurred, boxes)

    @staticmethod
    def resize(**kwargs):
        """
        Resize the image.

        Parameters:
            image: NumPy image array.
            boxes: Optional bounding boxes.
            box_format: "yolo" or "xyxy".
            width: Target width.
            height: Target height.
            interpolation: OpenCV interpolation flag.

        Returns:
            image if boxes are not supplied;
            otherwise (image, boxes).
        """
        image = Augmenter._validate_image(kwargs.get("image"))
        boxes = Augmenter._copy_boxes(kwargs.get("boxes"))
        box_format = kwargs.get("box_format", "yolo")

        original_h, original_w = image.shape[:2]
        target_w = kwargs.get("width")
        target_h = kwargs.get("height")

        if target_w is None or target_h is None:
            scale = kwargs.get("scale")
            if scale is None:
                raise ValueError("resize requires either width/height or scale.")
            scale = float(scale)
            if scale <= 0:
                raise ValueError("scale must be positive.")
            target_w = int(round(original_w * scale))
            target_h = int(round(original_h * scale))

        target_w = int(target_w)
        target_h = int(target_h)

        if target_w <= 0 or target_h <= 0:
            raise ValueError("width and height must be positive.")

        interpolation = kwargs.get("interpolation", cv2.INTER_LINEAR)
        resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)

        if boxes is not None:
            if box_format == "yolo":
                # Normalized YOLO boxes do not change under resizing.
                pass

            elif box_format == "xyxy":
                scale_x = target_w / original_w
                scale_y = target_h / original_h
                boxes[:, [1, 3]] *= scale_x
                boxes[:, [2, 4]] *= scale_y

            else:
                raise ValueError("box_format must be either 'yolo' or 'xyxy'.")

        return Augmenter._return(resized, boxes)

    @staticmethod
    def change_brightness(**kwargs):
        """
        Adjust brightness and contrast.

        Parameters:
            image: NumPy image array.
            boxes: Optional bounding boxes. Boxes are unchanged.
            alpha: Contrast multiplier. Default: 1.0.
            beta: Brightness offset. Default: 30.

        Formula:
            output = image * alpha + beta

        Returns:
            image if boxes are not supplied;
            otherwise (image, boxes).
        """
        image = Augmenter._validate_image(kwargs.get("image"))
        boxes = Augmenter._copy_boxes(kwargs.get("boxes"))

        alpha = float(kwargs.get("alpha", 1.0))
        beta = float(kwargs.get("beta", 30))

        adjusted = image.astype(np.float32) * alpha + beta
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        return Augmenter._return(adjusted, boxes)

    @staticmethod
    def transform(**kwargs):
        """
        Apply a random sequence of augmentations.

        Parameters:
            image: NumPy image array.
            boxes: Optional bounding boxes.
            box_format: "yolo" or "xyxy".
            max_transforms: Maximum number of transforms to apply. Default: 3.
            seed: Optional random seed.

        Returns:
            image if boxes are not supplied;
            otherwise (image, boxes).
        """
        image = Augmenter._validate_image(kwargs.get("image"))
        boxes = Augmenter._copy_boxes(kwargs.get("boxes"))

        seed = kwargs.get("seed")
        rng = random.Random(seed)

        box_format = kwargs.get("box_format", "yolo")
        max_transforms = int(kwargs.get("max_transforms", 3))
        max_transforms = max(1, max_transforms)

        available = [
            "vertical_flip",
            "gaussian_blur",
            "change_brightness",
        ]

        # Resize is excluded from the default random chain because resizing can
        # complicate later evaluation unless the entire pipeline expects the new size.
        if kwargs.get("include_resize", False):
            available.append("resize")

        rng.shuffle(available)
        n_transforms = rng.randint(1, min(max_transforms, len(available)))
        selected = available[:n_transforms]

        current_image = image.copy()
        current_boxes = boxes.copy() if boxes is not None else None

        for name in selected:
            method = getattr(Augmenter, name)

            method_kwargs = dict(kwargs)
            method_kwargs["image"] = current_image
            method_kwargs["boxes"] = current_boxes
            method_kwargs["box_format"] = box_format

            result = method(**method_kwargs)

            if current_boxes is None:
                current_image = result
            else:
                current_image, current_boxes = result

        return Augmenter._return(current_image, current_boxes)
