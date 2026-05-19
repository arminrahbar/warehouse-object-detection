import cv2
import os
from modules.inference.nms import NMS
from modules.inference.model import Detector
from modules.inference.preprocessing import Preprocessing


class InferenceService:
    """
    Handles real-time video inference by integrating preprocessing, object detection, 
    and Non-Maximum Suppression (NMS) filtering.

    This service continuously captures video frames, applies object detection, 
    filters results using NMS, and outputs predictions.
    """

    def __init__(self, stream: Preprocessing, detector: Detector, nms: NMS, save_dir: str, drop_rate: int = 10) -> None:
        """
        Initializes the inference service.

        :param stream: An instance of the Preprocessing class for video capture.
        :param detector: An instance of the Model class for object detection.
        :param nms: An instance of the NMS class for filtering overlapping bounding boxes.
        :param save_dir: Directory string where processed frames will be saved.
        :param drop_rate: The rate at which frames are dropped (default: 10).

        :ivar self.stream: Video stream processor.
        :ivar self.detector: Object detection model.
        :ivar self.nms: Non-Maximum Suppression processor.
        """
        self.stream = stream
        self.detector = detector
        self.nms = nms
        self.drop_rate = drop_rate
        self.save_dir = save_dir

        print("[INFO] Inference Service initialized.")

    def draw_boxes(self, frame, bboxes, class_ids, confidence_scores):
        """
        Draws bounding boxes with class labels and confidence scores on a given frame.

        :param frame: The input frame on which to draw the bounding boxes.
        :param bboxes: List of bounding boxes as (x, y, width, height).
        :param class_ids: List of detected class IDs.
        :param confidence_scores: List of confidence scores corresponding to detections.

        :return: The frame with drawn bounding boxes.
        """
        for i, (x, y, w, h) in enumerate(bboxes):
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Prepare label text
            label = f"Class {class_ids[i]}: {confidence_scores[i]:.2f}"

            # Put label text on the frame
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame

    def save_frame(self, frame, frame_number):
        """
        Saves the processed frame with bounding boxes.

        :param frame: The frame with detections drawn.
        :param frame_number: The current frame number.
        """
        if self.save_dir:
            filename = os.path.join(self.save_dir, f"frame_{frame_number}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[INFO] Frame {frame_number} saved at {filename}")

    def run(self) -> None:
        """
        Runs the inference service.

        This method:
        1. Captures frames from the video stream.
        2. Applies object detection to each frame.
        3. Filters the detected bounding boxes using Non-Maximum Suppression (NMS).
        4. Print per-frame detections (i.e, bounding box, class_id, object_score)
        5. Save frame with bounding box overlay to directory.

        **Processing Pipeline:**
        - Capture video frame.
        - Perform object detection using YOLO.
        - Post-process detections to extract bounding boxes and confidence scores.
        - Apply Non-Maximum Suppression to remove redundant detections.
        - Display the predictions and filtered results.

        **Draw Bounding Boxes in frame:**
        ```python
        processed_frame = self.draw_boxes(frame, bboxes, class_ids, scores)

        # Save the frame with detections
        self.save_frame(processed_frame, frame_count)
        ```

        **Example Class Usage:**
        ```python
        service = InferenceService(stream, model, nms)
        service.run()
        ```
        """
        
        # TASK: Implement your Inference service. Use the docstrings to guide you on the
        #       logic behind this implementation. 


# Runner for Inference Service
if __name__ == "__main__":
    print("[INFO] Starting Inference Service...")

    # Initialize Video Processing
    VIDEO_SOURCE = "udp://127.0.0.1:23000"
    print(f"[INFO] Initializing video stream from source: {VIDEO_SOURCE}")
    stream = Preprocessing(VIDEO_SOURCE, drop_rate=60)

    # Initialize Model
    WEIGHTS_PATH = "storage/yolo_models/yolov4-tiny-logistics_size_416_1.weights"
    CONFIG_PATH = "storage/yolo_models/yolov4-tiny-logistics_size_416_1.cfg"
    CLASS_NAMES_PATH = "storage/yolo_models/logistics.names"
    SCORE_THRESHOLD = 0.5

    print("[INFO] Loading YOLO Model...")
    model = Detector(WEIGHTS_PATH, CONFIG_PATH, CLASS_NAMES_PATH, SCORE_THRESHOLD)
    print("[INFO] Model loaded successfully.")

    # Initialize NMS
    IOU_THRESHOLD = 0.4
    print("[INFO] Initializing Non-Maximum Suppression...")
    nms = NMS(SCORE_THRESHOLD, IOU_THRESHOLD)
    print("[INFO] NMS initialized.")

    # Create and run inference service
    print("[INFO] Starting Inference Service loop...")
    service = InferenceService(stream, model, nms, save_dir='output')
    service.run()

    print("[INFO] Inference Service terminated.")
