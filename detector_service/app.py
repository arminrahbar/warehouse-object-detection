"""Command-line UDP video inference service."""

from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

import cv2

from detector_service.modules.inference.model import Detector
from detector_service.modules.inference.nms import NMS
from detector_service.modules.inference.preprocessing import Preprocessing


SERVICE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SERVICE_DIR / "storage" / "yolo_model_2"


def positive_int(value: str) -> int:
    """Convert a CLI value to an integer greater than zero."""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ArgumentTypeError("value must be a positive integer") from exc
    if number <= 0:
        raise ArgumentTypeError("value must be a positive integer")
    return number


class InferenceService:
    """Coordinate stream sampling, inference, NMS, reporting, and output."""

    def __init__(
        self,
        stream: Preprocessing,
        detector: Detector,
        nms: NMS,
        save_dir: Path | str | None,
        max_frames: int | None = None,
    ) -> None:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be greater than 0")

        self.stream = stream
        self.detector = detector
        self.nms = nms
        self.save_dir = Path(save_dir) if save_dir else None
        self.max_frames = max_frames

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.detector.classes):
            return self.detector.classes[class_id]
        return f"class_{class_id}"

    def draw_boxes(self, frame, bboxes, class_ids, confidences):
        """Mutate a frame with green boxes and red class/confidence labels."""
        for bbox, class_id, confidence in zip(
            bboxes,
            class_ids,
            confidences,
        ):
            x, y, width, height = (int(coordinate) for coordinate in bbox)
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"{self._class_name(class_id)}: {confidence:.2f}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        return frame

    def save_frame(self, frame, frame_number: int) -> Path | None:
        """Write one numbered JPEG, or return immediately when saving is off."""
        if self.save_dir is None:
            return None

        destination = self.save_dir / f"frame_{frame_number:06d}.jpg"
        if not cv2.imwrite(str(destination), frame):
            raise OSError(f"Unable to write annotated frame: {destination}")
        return destination

    @staticmethod
    def _report_frame(frame_number, bboxes, class_ids, confidences) -> None:
        print(f"[FRAME] index={frame_number} detections={len(bboxes)}")
        for bbox, class_id, confidence in zip(
            bboxes,
            class_ids,
            confidences,
        ):
            print(
                f"[DETECTION] frame={frame_number} bbox={bbox} "
                f"class_id={class_id} "
                f"combined_confidence={confidence:.4f}"
            )

    def run(self) -> int:
        """Process sampled frames until exhaustion, interruption, or a limit."""
        processed_frames = 0
        frame_stream = self.stream.capture_video()

        try:
            for frame_number, frame in enumerate(frame_stream):
                raw_outputs = self.detector.predict(frame)
                candidates = self.detector.post_process(raw_outputs)
                retained = self.nms.filter(*candidates)
                confidences = self.nms.confidence_scores(
                    retained[1],
                    retained[2],
                    retained[3],
                )

                self._report_frame(
                    frame_number,
                    retained[0],
                    retained[1],
                    confidences,
                )
                annotated = self.draw_boxes(
                    frame.copy(),
                    retained[0],
                    retained[1],
                    confidences,
                )
                self.save_frame(annotated, frame_number)
                processed_frames += 1

                if (
                    self.max_frames is not None
                    and processed_frames >= self.max_frames
                ):
                    print(
                        "[INFO] Reached configured frame limit: "
                        f"{self.max_frames}."
                    )
                    break
        except KeyboardInterrupt:
            print("[INFO] Inference interrupted by user.")
        finally:
            close = getattr(frame_stream, "close", None)
            if close is not None:
                close()

        print(f"[INFO] Inference completed. Processed {processed_frames} frames.")
        return processed_frames


def build_parser() -> ArgumentParser:
    """Build the command-line interface for the inference service."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="udp://127.0.0.1:23000",
        help="Video file or network-stream URL understood by OpenCV.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_MODEL_DIR / "yolov4-tiny-logistics_size_416_2.weights",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MODEL_DIR / "yolov4-tiny-logistics_size_416_2.cfg",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=DEFAULT_MODEL_DIR / "logistics.names",
    )
    parser.add_argument(
        "--frame-interval",
        type=positive_int,
        default=60,
        help="Process one of every N decoded frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        default=None,
        help="Stop after processing N sampled frames; unlimited by default.",
    )
    parser.add_argument(
        "--candidate-threshold",
        type=float,
        default=0.5,
        help="Minimum raw objectness retained before post-processing.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum combined confidence retained by NMS.",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.3,
        help="Same-class IoU above which the lower-ranked box is suppressed.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=SERVICE_DIR / "storage" / "detections",
        help="Directory for annotated frames.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run inference without writing annotated frames.",
    )
    return parser


def main(argv=None) -> int:
    """Construct configured runtime components and execute the service."""
    args = build_parser().parse_args(argv)
    stream = Preprocessing(args.source, drop_rate=args.frame_interval)
    detector = Detector(
        str(args.weights),
        str(args.config),
        str(args.classes),
        args.candidate_threshold,
    )
    nms = NMS(args.confidence_threshold, args.nms_iou_threshold)
    service = InferenceService(
        stream,
        detector,
        nms,
        save_dir=None if args.no_save else args.save_dir,
        max_frames=args.max_frames,
    )
    return service.run()


if __name__ == "__main__":
    main()
