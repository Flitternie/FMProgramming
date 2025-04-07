from __future__ import annotations

import re
import warnings
from typing import Union, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.ops import box_iou
from word2number import w2n

from execution.modules import object_detection, vqa, llm
from execution.utils import show_single_image

"""
This file defines the ImagePatch class, inspired by the implementation in the ViperGPT repository.
"""

class ImagePatch:
    """
    Represents a crop of an image and supports detection, querying, and visual reasoning.

    Attributes
    ----------
    cropped_image : torch.Tensor
        Cropped image tensor (C x H x W).
    left, lower, right, upper : int
        Coordinates of the crop relative to the original image.
    height, width : int
        Dimensions of the cropped image.
    horizontal_center, vertical_center : float
        Center point of the crop (used for sorting, comparison).
    """

    def __init__(
        self,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        left: int = None,
        lower: int = None,
        right: int = None,
        upper: int = None,
        parent_left=0,
        parent_lower=0,
        queues=None,
        parent_img_patch=None
    ):
        """
        Initialize an ImagePatch from a given image or tensor and optional crop coordinates.

        Parameters
        ----------
        image : Image.Image, torch.Tensor, or np.ndarray
            The full image or sub-region to initialize the patch from.
        left, lower, right, upper : int, optional
            Crop coordinates (relative to the image) to extract the patch.
        parent_left, parent_lower : int, optional
            Relative offset if this patch is a crop from another patch.
        queues : tuple, optional
            Reserved for task-specific asynchronous use.
        parent_img_patch : ImagePatch, optional
            Parent patch from which this one is cropped (if applicable).
        """
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(1, 2, 0)
        elif isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image / 255

        if all(v is None for v in [left, lower, right, upper]):
            self.cropped_image = image
            self.left, self.lower = 0, 0
            self.right, self.upper = image.shape[2], image.shape[1]
        else:
            self.cropped_image = image[:, lower:upper, left:right]
            self.left, self.lower = left, lower
            self.right, self.upper = right, upper

        self.height = self.cropped_image.shape[1]
        self.width = self.cropped_image.shape[2]

        if self.height == 0 or self.width == 0:
            raise ValueError("ImagePatch has no area.")

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2
        self.queues = (None, None) if queues is None else queues
        self.parent_img_patch = parent_img_patch
        self.cache = {}

    @property
    def original_image(self):
        """Returns the full original image this patch is derived from."""
        if self.parent_img_patch is None:
            return self.cropped_image
        return self.parent_img_patch.original_image

    def find(self, object_name: str, routing: int) -> List[ImagePatch]:
        """
        Detect instances of an object in the patch.

        Parameters
        ----------
        object_name : str
            Name of the object to detect.
        routing : int
            Index of the model to route to.

        Returns
        -------
        List[ImagePatch]
            List of patches for each detected object.
        """
        coords = object_detection(self.cropped_image, object_name, routing)
        patches = []
        for c in coords:
            try:
                patches.append(self.crop(*c))
            except:
                warnings.warn("Invalid coordinates in object detection.")
        return patches

    def exists(self, object_name: str, routing: int) -> bool:
        """
        Check if a specific object exists in the patch.

        Parameters
        ----------
        object_name : str
            Name or numeric representation of the object.
        routing : int
            Model routing index.

        Returns
        -------
        bool
            True if the object exists in the patch.
        """
        if object_name.isdigit() or object_name.lower().startswith("number"):
            val = w2n.word_to_num(object_name.lower().replace("number", "").strip())
            answer = self.query("What number is written in the image (in digits)?", routing=routing) # Note: Nested method call
            return w2n.word_to_num(answer) == val

        return len(self.find(object_name, routing=routing)) > 0 # Note: Nested method call

    def verify_property(self, object_name: str, attribute: str, routing: int) -> bool:
        """
        Check if a specific object in the patch has a given visual attribute.

        Parameters
        ----------
        object_name : str
            Object of interest.
        attribute : str
            Attribute to verify.
        routing : int
            Model routing index.

        Returns
        -------
        bool
            True if the attribute is verified for the object.
        """
        query = f"is this {object_name} {attribute} ?"
        return "yes" in self.query(query, routing=routing).lower() # Note: Nested method call

    def query(self, question: str, routing: int) -> str:
        """
        Ask a visual question about the image content.

        Parameters
        ----------
        question : str
            Visual question to ask.
        routing : int
            Model routing index.

        Returns
        -------
        str
            Answer from the VQA model.
        """
        return vqa(self.cropped_image, question, routing)

    def crop(self, left: int, lower: int, right: int, upper: int) -> ImagePatch:
        """
        Crop a sub-region of the current patch and return it as a new ImagePatch.

        Parameters
        ----------
        left, lower, right, upper : int
            Coordinates of the desired sub-region.

        Returns
        -------
        ImagePatch
        """
        return ImagePatch(
            self.cropped_image,
            int(left), int(lower), int(right), int(upper),
            self.left, self.lower,
            queues=self.queues,
            parent_img_patch=self
        )

    def overlaps_with(self, left: int, lower: int, right: int, upper: int) -> bool:
        """
        Check if a bounding box overlaps with this patch.

        Returns
        -------
        bool
            True if overlapping.
        """
        return self.left <= right and self.right >= left and self.lower <= upper and self.upper >= lower

    def llm_query(self, question: str, routing: int) -> str:
        """
        Ask a language model a question using external knowledge.

        Parameters
        ----------
        question : str
            The text-based question.
        routing : int
            Model routing index.

        Returns
        -------
        str
            Answer from the language model.
        """
        return llm(question, routing)

    def show(self, size: tuple[int, int] = None):
        """
        Display the cropped image using matplotlib or IPython.

        Parameters
        ----------
        size : tuple, optional
            Target display size (width, height).
        """
        show_single_image(self.cropped_image, size)

    def __repr__(self):
        return f"ImagePatch({self.left}, {self.lower}, {self.right}, {self.upper})"


def distance(patch_a: Union[ImagePatch, float], patch_b: Union[ImagePatch, float]) -> float:
    """
    Compute the distance between two ImagePatches or two float values.

    If patches overlap, returns a negative IoU-based distance.

    Parameters
    ----------
    patch_a, patch_b : ImagePatch or float

    Returns
    -------
    float
    """
    if isinstance(patch_a, ImagePatch) and isinstance(patch_b, ImagePatch):
        a_min = np.array([patch_a.left, patch_a.lower])
        a_max = np.array([patch_a.right, patch_a.upper])
        b_min = np.array([patch_b.left, patch_b.lower])
        b_max = np.array([patch_b.right, patch_b.upper])

        u = np.maximum(0, a_min - b_max)
        v = np.maximum(0, b_min - a_max)
        dist = np.sqrt((u ** 2).sum() + (v ** 2).sum())

        if dist == 0:
            box_a = torch.tensor([patch_a.left, patch_a.lower, patch_a.right, patch_a.upper])[None]
            box_b = torch.tensor([patch_b.left, patch_b.lower, patch_b.right, patch_b.upper])[None]
            dist = -box_iou(box_a, box_b).item()
    else:
        dist = abs(patch_a - patch_b)

    return dist


def to_numeric(string: str, no_string: bool = False):
    """
    Convert a string to an int or float, if possible. Supports:
    - Raw numbers
    - Word numbers (e.g., "two")
    - Strings with units (e.g., "5cm")

    Parameters
    ----------
    string : str
        Input to convert.
    no_string : bool
        If True, raises exception instead of returning original string when numeric conversion fails.

    Returns
    -------
    int, float, or str
    """
    try:
        return int(string)
    except ValueError:
        pass
    try:
        return float(string)
    except ValueError:
        pass

    try:
        return w2n.word_to_num(string)
    except ValueError:
        pass

    # Clean up unwanted characters
    string_re = re.sub("[^0-9.\-]", "", string)
    string_re = string_re.replace('&', '-') if string_re.startswith('-') else string_re

    if '-' in string_re:
        return to_numeric(string_re.split('-')[0], no_string=no_string)

    try:
        return float(string_re) if '.' in string_re else int(string_re)
    except ValueError:
        if no_string:
            raise ValueError(f"Cannot convert {string} to numeric.")
        return string
