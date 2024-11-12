import typing

import cv2
import numpy as np
from mltu.annotations.images import Image
from mltu.transformers import Transformer


class ImageThresholding(Transformer):
    """Threshold image"""

    def __init__(self, threshold: bool = False):
        """Initialize ImageThresholding

        Args:
            transpose_axis (bool): Whether to transpose axis. Default: False
        """
        self.threshold = threshold

    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[np.ndarray, typing.Any]:
        """Convert each Image to numpy, transpose axis ant normalize to float value"""
        img_numpy = image
        if type(img_numpy) is not np.ndarray:
            img_numpy = image.numpy()

        # Apply Otsu's thresholding
        img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)

        img_numpy = cv2.bilateralFilter(img_numpy, 8, 10, 30)

        _, img = cv2.threshold(img_numpy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Ensure the output is shaped [height, width, channels]
        img = np.expand_dims(img, axis=-1)

        # # img = binary_img.numpy()
        # print(img.shape)

        return img, annotation
