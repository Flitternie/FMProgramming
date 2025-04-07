import os
import io
import base64
import yaml
import torch
import matplotlib.pyplot as plt

from PIL import Image
from typing import Union, List, Dict, Any, Optional
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes as tv_draw_bounding_boxes
from openai import OpenAI

"""
This file contains utility functions and classes for handling images, visualizations,
remote model calls, and config loading. It supports bounding box visualization and 
coco-class conversion for object detection tasks.
"""

# ---------------------------------------------------------------------------- #
#                                  Utilities                                   #
# ---------------------------------------------------------------------------- #

def is_interactive() -> bool:
    """
    Check if the current Python session is interactive (e.g., Jupyter notebook).

    Returns:
        bool: True if interactive, False otherwise.
    """
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def load_config(file_path: str) -> 'YAMLObject':
    """
    Load a YAML config file and convert it into an object.

    Args:
        file_path (str): Path to the YAML config file.

    Returns:
        YAMLObject: An object with config data as attributes.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return YAMLObject(data)


def denormalize(images: torch.Tensor, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225)) -> torch.Tensor:
    """
    Denormalize an image tensor (reverse normalization).

    Args:
        images (Tensor): Normalized image tensor [B, C, H, W].
        means (tuple): Mean values used during normalization.
        stds (tuple): Std values used during normalization.

    Returns:
        Tensor: Denormalized image tensor.
    """
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


# ---------------------------------------------------------------------------- #
#                             Visualization Utils                              #
# ---------------------------------------------------------------------------- #

def show_single_image(
    image: Union[torch.Tensor, Image.Image],
    denormalize_stats: Optional[tuple] = None,
    bgr_image: bool = False,
    save_path: Optional[str] = None,
    size: str = 'small',
    bbox_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Display or save a single image, optionally with bounding boxes.

    Args:
        image (Tensor | PIL.Image): Image to display.
        denormalize_stats (tuple): (mean, std) for denormalizing image.
        bgr_image (bool): Convert BGR image to RGB.
        save_path (str): Path to save the image (optional).
        size (str): 'small', 'large', or 'size_img' for custom sizing.
        bbox_info (dict): Dict with keys 'bboxes', 'labels', and 'colors'.
    """
    if not is_interactive():
        import matplotlib
        matplotlib.use("module://imgcat")

    if size == 'size_img' and isinstance(image, torch.Tensor):
        figsize = (image.shape[2] / 100, image.shape[1] / 100)
    elif size == 'small':
        figsize = (4, 4)
    else:
        figsize = (12, 12)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    if bbox_info:
        image = draw_bounding_boxes(
            image=image,
            bboxes=bbox_info['bboxes'],
            labels=bbox_info['labels'],
            colors=bbox_info['colors'],
            width=5
        )

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()

        if denormalize_stats:
            image = denormalize(image.unsqueeze(0), *denormalize_stats)
        if image.dtype == torch.float32:
            image = image.clamp(0, 1)

        ax.imshow(image.squeeze(0).permute(1, 2, 0))
    else:
        if bgr_image:
            image = image[..., ::-1]
        ax.imshow(image)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()


def draw_bounding_boxes(
    image: Union[torch.Tensor, Image.Image],
    bboxes: Union[List[List[int]], torch.Tensor],
    width: int = 5,
    **kwargs
) -> torch.Tensor:
    """
    Draw bounding boxes on a tensor or PIL image.

    Args:
        image (Tensor | PIL.Image): Image to draw on.
        bboxes (list | Tensor): Bounding boxes [xmin, ymin, xmax, ymax].
        width (int): Line width.
        **kwargs: Additional draw options.

    Returns:
        Tensor: Image tensor with bounding boxes.
    """
    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image)

    if isinstance(bboxes, list):
        bboxes = torch.tensor(bboxes)

    image = (image * 255).to(torch.uint8).cpu()
    height = image.shape[1]

    # Adjust y-coordinates (bottom-up to top-down)
    bboxes = torch.stack([
        bboxes[:, 0],
        height - bboxes[:, 3],
        bboxes[:, 2],
        height - bboxes[:, 1]
    ], dim=1)

    return tv_draw_bounding_boxes(image, bboxes, width=width, **kwargs)


# ---------------------------------------------------------------------------- #
#                                  YAML Class                                  #
# ---------------------------------------------------------------------------- #

class YAMLObject:
    """
    Recursive object that converts a YAML dictionary to an object with attributes.

    Example:
        config = YAMLObject({'foo': {'bar': 42}})
        print(config.foo.bar)  # 42
    """

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                value = YAMLObject(value)
            elif isinstance(value, list):
                value = [YAMLObject(v) if isinstance(v, dict) else v for v in value]
            setattr(self, key, value)

    def serialize(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        return self.__dict__


# ---------------------------------------------------------------------------- #
#                                 Remote MLLM                                  #
# ---------------------------------------------------------------------------- #

class RemoteMLLM:
    """
    Interface for calling remote multimodal LLMs using OpenAI-compatible API.

    Args:
        model_name (str): Name of the model.
        api_key (str): API key.
        base_url (str): API base URL.
        args (YAMLObject): Additional model args (including prompt).
    """

    def __init__(self, model_name: str, api_key: str, base_url: str, args: YAMLObject):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.model_args = args.serialize()
        if 'prompt' in self.model_args:
            self.prompt = self.model_args.pop('prompt')
        else:
            self.prompt = ""

    def encode_base64(self, image: Image.Image) -> str:
        """
        Convert PIL image to base64 JPEG string.

        Args:
            image (PIL.Image): Input image.

        Returns:
            str: Base64-encoded string.
        """
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def __call__(self, image: Image.Image, question: str) -> str:
        """
        Query the model with an image and a question.

        Args:
            image (PIL.Image): Input image.
            question (str): Natural language question.

        Returns:
            str: Answer from the model.
        """
        img_b64 = self.encode_base64(image)

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question + " " + self.prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ],
            **self.model_args
        )

        return completion.choices[0].message.content


# ---------------------------------------------------------------------------- #
#                              COCO Class Utilities                            #
# ---------------------------------------------------------------------------- #

def convert_coco(object_name: str) -> str:
    """
    Normalize object name to match COCO class labels.

    Args:
        object_name (str): Input label.

    Returns:
        str: Standard COCO class label.
    """
    object_name = object_name.lower().strip()

    if object_name not in coco_classes:
        if any(w in object_name for w in ["man", "woman", "child", "boy", "girl"]):
            return "person"
        if "television" in object_name:
            return "tv"
        if "ball" in object_name:
            return "sports ball"
        for word in object_name.split():
            if word in coco_classes:
                return word

    return object_name


# ---------------------------------------------------------------------------- #
#                                COCO Classes                                  #
# ---------------------------------------------------------------------------- #

coco_classes = {
    '__background__': 0,
    'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6,
    'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11,
    'stop sign': 12, 'parking meter': 13, 'bench': 14, 'bird': 15, 'cat': 16,
    'dog': 17, 'horse': 18, 'sheep': 19, 'cow': 20, 'elephant': 21, 'bear': 22,
    'zebra': 23, 'giraffe': 24, 'backpack': 25, 'umbrella': 26, 'handbag': 27,
    'tie': 28, 'suitcase': 29, 'frisbee': 30, 'skis': 31, 'snowboard': 32,
    'sports ball': 33, 'kite': 34, 'baseball bat': 35, 'baseball glove': 36,
    'skateboard': 37, 'surfboard': 38, 'tennis racket': 39, 'bottle': 40,
    'wine glass': 41, 'cup': 42, 'fork': 43, 'knife': 44, 'spoon': 45, 'bowl': 46,
    'banana': 47, 'apple': 48, 'sandwich': 49, 'orange': 50, 'broccoli': 51,
    'carrot': 52, 'hot dog': 53, 'pizza': 54, 'donut': 55, 'cake': 56, 'chair': 57,
    'couch': 58, 'potted plant': 59, 'bed': 60, 'dining table': 61, 'toilet': 62,
    'tv': 63, 'laptop': 64, 'mouse': 65, 'remote': 66, 'keyboard': 67,
    'cell phone': 68, 'microwave': 69, 'oven': 70, 'toaster': 71, 'sink': 72,
    'refrigerator': 73, 'book': 74, 'clock': 75, 'vase': 76, 'scissors': 77,
    'teddy bear': 78, 'hair drier': 79, 'toothbrush': 80
}
