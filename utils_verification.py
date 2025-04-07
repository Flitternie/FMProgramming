import PIL
import random
import tqdm
import hashlib
import torch
import torchvision.transforms as transforms

"""
This file contains utility functions and dataset classes for binary VQA tasks.
"""

# ---------------------------------------------------------------------------- #
#                           Helper Functions                                   #
# ---------------------------------------------------------------------------- #

def postprocessing(output):
    """
    Convert a string or boolean output into a binary label (0 or 1).

    Args:
        output (str | bool): The output returned by the model.

    Returns:
        int: 1 if "yes" in output or output is True, 0 otherwise.
    """
    if isinstance(output, bool):
        return int(output)
    if "yes" in output.lower():
        output = 1
    else:
        output = 0
    return output

def hash_query(query):
    """
    Generate a hash for a query string using SHA-512.

    Args:
        query (str): Input string.

    Returns:
        int: Hashed integer value.
    """
    return int(hashlib.sha512(query.encode('utf-8')).hexdigest(), 16)

# ---------------------------------------------------------------------------- #
#                             Image Processing Functions                       #
# ---------------------------------------------------------------------------- #

img_dir = './data/coco/train2017/{}.jpg'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def load_image(img_id):
    """
    Load an image by its COCO ID, apply preprocessing.

    Args:
        img_id (int | str): COCO image ID.

    Returns:
        torch.Tensor or None: Preprocessed image or None if not RGB.
    """
    img_path = img_dir.format(str(img_id).zfill(12))
    img = PIL.Image.open(img_path)
    if img.mode != 'RGB':
        return None
    img = transform(img)
    return img

# ---------------------------------------------------------------------------- #
#                              Data Preparation                                #
# ---------------------------------------------------------------------------- #

def prepare_data(query_data, hased_query):
    """
    Load and cache positive and negative images for a given query.

    Args:
        query_data (dict): Dictionary containing keys `positive_images`, `negative_images`.
        hased_query (int): Hashed query string used as filename identifier.

    Returns:
        tuple: (positive_images, positive_image_ids, negative_images, negative_image_ids)
    """
    positive_image_ids = query_data['positive_images']
    try:
        positive_images = torch.load(f'./data/saved_retrieval_imgs/positive_images_{hased_query}.pt', weights_only=False)
    except:
        positive_images = [load_image(i) for i in positive_image_ids]
        torch.save(positive_images, f'./data/saved_retrieval_imgs/positive_images_{hased_query}.pt')

    negative_image_ids = query_data['negative_images']
    try:
        negative_images = torch.load(f'./data/saved_retrieval_imgs/negative_images_{hased_query}.pt', weights_only=False)
    except:
        negative_images = [load_image(i) for i in tqdm.tqdm(negative_image_ids)]
        torch.save(negative_images, f'./data/saved_retrieval_imgs/negative_images_{hased_query}.pt')
    
    return positive_images, positive_image_ids, negative_images, negative_image_ids

# ---------------------------------------------------------------------------- #
#                           Verification Dataset Class                         #
# ---------------------------------------------------------------------------- #

class VerificationDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for handling binary VQA tasks. 

    Args:
        positive_images (list): List of torch.Tensor positive images.
        positive_img_ids (list): Corresponding image IDs for positive images.
        negative_images (list): List of torch.Tensor negative images.
        negative_img_ids (list): Corresponding image IDs for negative images.
        random_seed (int): Seed for deterministic shuffling.
    """
    def __init__(self, positive_images, positive_img_ids, negative_images, negative_img_ids, random_seed=42):
        self.positive_images = positive_images
        self.positive_img_ids = positive_img_ids
        self.negative_images = [img for img in negative_images if img is not None]
        self.negative_img_ids = [img_id for img_id, img in zip(negative_img_ids, negative_images) if img is not None]
        self.random_seed = random_seed
        self._shuffle()
    
    def _shuffle(self):
        """
        Shuffle the combined dataset with consistent indexing.
        """
        random.seed(self.random_seed)
        self.images = self.positive_images + self.negative_images
        self.labels = [1] * len(self.positive_images) + [0] * len(self.negative_images)
        self.img_ids = self.positive_img_ids + self.negative_img_ids
        assert len(self.images) == len(self.labels) == len(self.img_ids)
        self.indices = list(range(len(self.images)))
        random.shuffle(self.indices)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        return self.images[idx], self.labels[idx], self.img_ids[idx]


class ExampleVerificationDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset version for debugging — uses all positive images
    and the first 15 negative images.

    Args:
        positive_images (list): List of torch.Tensor positive images.
        positive_img_ids (list): Corresponding image IDs for positive images.
        negative_images (list): List of torch.Tensor negative images.
        negative_img_ids (list): Corresponding image IDs for negative images.
    """
    def __init__(self, positive_images, positive_img_ids, negative_images, negative_img_ids):
        self.images = positive_images + negative_images[:15]
        self.img_ids = positive_img_ids + negative_img_ids[:15]
        self.labels = [1] * len(positive_images) + [0] * 15

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.img_ids[idx]
