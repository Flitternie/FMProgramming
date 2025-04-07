import os
import PIL
import random
import torch
import torchvision.transforms as transforms
from openai import OpenAI

"""
This file contains utility functions and dataset classes for open-form VQA tasks,
"""

PROMPT = '''
            Keep your answer short. Try to answer within 3 words. 
            For numerical answers, use number digits (e.g. 5 instead of five), and returns the number only. 
            If there are multiple answers, separate them with a comma (e.g. cat, dog).
            If you find the question unanswerable based on the image content, output "N/A". 
            For example, if the image content is irrelevant to the question, or the content in the image does not fully and clearly match all the entities, humans, attributes, spatial, logical, and numerical constraints in the question, output "N/A"
            '''

# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #

def equivalence(output, label):
    """
    Checks if the output and label are equivalence based on specific criteria.

    Args:
        output (str): The output to compare.
        label (str): The label to compare with.

    Returns:
        bool: True if the output and label match based on the defined criteria, False otherwise.
    """
    output = str(output)
    label = str(label)
    
    # Case insensitive check
    if label.lower().strip() in output.lower().strip():
        return True
    # Special case for "n/a" and "0" equivalence
    elif label == "n/a" and output == "0":
        return True
    return False

# ---------------------------------------------------------------------------- #
#                             Image Processing Functions                       #
# ---------------------------------------------------------------------------- #

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image to 256x256
    transforms.ToTensor(),          # Convert image to tensor
])

def load_image(img_path):
    """
    Load an image from a file path and transform it into a tensor. Only RGB images are supported.

    Args:
        img_path (str): Path to the image file.

    Returns:
        torch.Tensor or None: Transformed image as a tensor, or None if the image is not RGB.
    """
    img = PIL.Image.open(img_path)
    if img.mode != 'RGB':
        return None
    img = transform(img)
    return img


def prepare_data(data):
    """
    Prepare the image paths and answers from the given dataset.

    Args:
        data (list): List of dictionaries containing image paths and answers.

    Returns:
        tuple: A tuple containing:
            - image_paths (list): List of image file paths.
            - answers (list): List of corresponding answers.
    """
    image_paths = []
    answers = []
    for item in data:
        image_paths.append(item['image'])
        answers.append(item['answer'])
    return image_paths, answers


# ---------------------------------------------------------------------------- #
#                               VQA Dataset Class                              #
# ---------------------------------------------------------------------------- #

class VqaDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for handling open-form VQA tasks. 

    Args:
        data_dir (str): Path to the directory containing images.
        image_data (list): List of dictionaries containing image file paths and associated answers.
        random_seed (int, optional): Seed for shuffling the dataset. Default is 42.
    """

    def __init__(self, data_dir, image_data, random_seed=42):
        self.data = image_data
        self.data_dir = data_dir
        self.random_seed = random_seed
        self._initiate()
        self._shuffle()
    
    def _initiate(self):
        """
        Initialize the dataset by loading images and answers from the data.
        Only RGB images are included.
        """
        self.images = []
        self.labels = []
        self.image_paths = []
        for data in self.data:
            image_path = os.path.join(self.data_dir, data['image'])
            image = load_image(image_path)
            if image is not None:
                self.images.append(image)
                self.labels.append(data['answer'])
                self.image_paths.append(data['image'])
    
    def _shuffle(self):
        """
        Shuffle the dataset with the provided random seed.
        """
        random.seed(self.random_seed)
        assert len(self.images) == len(self.labels)
        self.indices = list(range(len(self.images)))
        random.shuffle(self.indices)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Retrieve an item (image, answer, and path) at the given index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): The image tensor.
                - label (str): The answer associated with the image.
                - image_path (str): Path to the image file.
        """
        idx = self.indices[idx]
        return self.images[idx], self.labels[idx], self.image_paths[idx]
