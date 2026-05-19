import cv2
import numpy as np
import random


class Augmenter:
    """
    A collection of dataset augmentation methods including transformations, 
    blurring, resizing, and brightness adjustments. 

    NOTE: This class is used to transform data necessary for training TechTrack's models.
          Imagine that the output of `self.transform()` is fed directly to train the model.
    
    The following transformations are included:
    - Horizontal flipping: i.e., def horizontal_flip(**kwargs)
    - Gaussian blurring: i.e., def gaussian_blur_image(**kwargs)
    - Resizing: i.e., def resize(**kwargs)
    - Brightness and contrast adjustments: i.e., def change_brightness(**kwargs)
        - HINT: you may use cv2.addWeighted()

    NOTE: These methods uses **kwargs to accept arbitrary keyword arguments,
    but explicit parameter definitions improve clarity and usability.
    - "**kwargs" reference: https://www.geeksforgeeks.org/args-kwargs-python/

    Finally, Provide a demonstration and visualizations of these methods in `notebooks/augmentation.ipynb`.
    You will define your own keywords for "**kwargs".
    """

    
    @staticmethod
    def horizontal_flip(**kwargs):
        """
        Horizontally flip the image.
        
        """
        pass

    @staticmethod
    def gaussian_blur(**kwargs):
        """
        Apply Gaussian blur to the image.
        
        """
        pass


    @staticmethod
    def resize(**kwargs):
        """
        Resize the image.
        
        """
        pass

    @staticmethod
    def change_brightness(**kwargs):
        """
        Adjust brightness and contrast of the image.
        
        """
        pass

    @staticmethod
    def transform(**kwargs):
        """
        Apply random augmentations from the available methods.
        
        Internal Process:
        1. A list of available augmentation functions is created.
        2. The list is shuffled to introduce randomness.
        3. A random number of augmentations is selected.
        4. The selected augmentations are applied sequentially to the image.
        
        :param image: Input image (numpy array)
        :param kwargs: Additional parameters for transformations (if any)
        :return: Augmented image
        """
        pass
        

"""
EXAMPLE RUNNER:

# Create an instance of Augmenter
augmenter = Augmenter()

kwargs = {"image": your_image, # Numpy type
            ... # Add more...
        }

# Apply random transformations
augmented_image = augmenter.transform(**kwargs)

# Display the original and transformed images
cv2.imshow("Original Image", image)
cv2.imshow("Augmented Image", augmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""