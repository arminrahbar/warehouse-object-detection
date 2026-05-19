import os
import math
import pandas as pd
import numpy as np

import os
from glob import glob
import numpy as np
import cv2
import pandas as pd

from ..utils.loss import Loss
from ..inference.model import Detector
from ..inference.nms import NMS


class HardNegativeMiner:
    """
    A class to mine hard negative examples for a given model.
    
    NOTE: DO NOT change this implementation! It's already done for you! All you have to do
    is complete the `sample_hard_negatives` method.

    Attributes:
        model: The model used for prediction.
        nms: Non-maximum suppression object.
        measure: Measure to evaluate predictions.
        dataset_dir: Directory containing the dataset.
        table: DataFrame to store the results.
    """

    def __init__(self, model: Detector, measure: Loss, dataset_dir: str):
        """
        Initialize the HardNegativeMiner with model, measure, and dataset directory.

        Internal Process:
        1. The model, NMS, and loss measure objects are stored as attributes.
        2. The dataset directory is assigned for locating images and annotations.
        3. An empty DataFrame is initialized to store results.

        Args:
            model: The model used for prediction.
            measure: Measure to evaluate predictions.
            dataset_dir: Directory containing the dataset.
        """
        self.model = model
        self.measure = measure
        self.dataset_dir = dataset_dir
        self.table = pd.DataFrame(columns=['annotation_file', 'image_file'] + self.measure.columns)

    def __read_annotations(self, file_path):
        """
        Read annotations from a text file.

        Internal Process:
        1. Opens the annotation file in read mode.
        2. Iterates through each line to extract bounding box information.
        3. Parses class labels and bounding box coordinates.
        4. Returns the list of parsed annotations.

        Args:
            file_path (str): Path to the annotation file.

        Returns:
            list: List of annotations in the format (class_label, x_center, y_center, width, height).
        """
        annotations = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_label = int(parts[0])
                bbox = list(map(float, parts[1:]))
                annotations.append((class_label, *bbox))
        return annotations

    def __predict(self, image):
        """
        Make a prediction on the provided image using the model.

        Internal Process:
        1. The input image is passed to the model for inference.
        2. The model returns predictions in a structured format.
        3. The output predictions are returned to be further processed.

        Args:
            image (ndarray): The image to predict on.

        Returns:
            output: The model's prediction output.
        """
        output = self.model.predict(image)
        return output

    def __construct_table(self):
        """
        Construct a table with image files, annotation files, and measures.

        Internal Process:
        1. Iterates through all image and annotation file pairs.
        2. Reads the corresponding image and annotation data.
        3. Runs the model to generate predictions.
        4. Computes the loss or accuracy measure using the provided function.
        5. Appends the computed results to a DataFrame.
        6. Stores the results for further processing.
        """
        table_rows = []
        for image_file, annotation_file in zip(
                sorted(glob(os.path.join(self.dataset_dir, "*.jpg"))),
                sorted(glob(os.path.join(self.dataset_dir, "*.txt")))):

            image = cv2.imread(image_file)
            annotation = self.__read_annotations(annotation_file)
            prediction = self.__predict(image)

            measures = self.measure.compute(prediction, annotation)            
            table_rows.append({'annotation_file': annotation_file, 'image_file': image_file, **measures})

        # Concatenate all rows at once
        self.table = pd.DataFrame(table_rows)

    def sample_hard_negatives(self, num_hard_negatives, criteria):
        """
        Sample hard negative examples based on the specified criteria.

        Internal Process:
        1. Calls `__construct_table()` to build a dataset of predictions.
        2. Sorts the dataset by the specified criteria in descending order.
        3. Selects the top `num_hard_negatives` as the hardest examples.
        4. Returns the selected samples as a DataFrame.

        Args:
            num_hard_negatives (int): The number of hard negatives to sample.
            criteria (str): The criteria to sort and sample the hard negatives.

        Returns:
            DataFrame: A DataFrame containing the sampled hard negative examples.
        """
        
        # TASK: Complete this method to rank the hard samples.

        
        pass
