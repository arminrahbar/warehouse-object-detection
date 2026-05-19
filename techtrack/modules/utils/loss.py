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
        pass 
    
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
        pass

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
        loc_loss = 0 # localization loss
        class_loss = 0 # classification loss
        conf_loss_obj = 0 # with object (or confidence) loss
        conf_loss_noobj = 0 # no object (or confidence) loss
        total_loss = 0 # aggregate loss including loc_loss, class_loss, conf_loss_obj, etc.

        # TASK: Complete this method to compute the Loss function.
        #         This method calculates the localization, objectness 
        #         (or confidence) and classification loss.
        #         This method will be called in the HardNegativeMiner class.
        #         ----------------------------------------------------------
        #         HINT: For simplicity complete use get_predictions(), get_annotations().
        #         You may add class methods to improve the readability of this code. 

        return {
            "total_loss": total_loss, 
            "loc_loss": loc_loss, 
            "conf_loss_obj": conf_loss_obj, 
            "conf_loss_noobj": conf_loss_noobj, 
            "class_loss": class_loss
        }
