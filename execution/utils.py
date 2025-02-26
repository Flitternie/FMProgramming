import matplotlib.pyplot as plt
import os
import io
import base64
import yaml
import torch
from openai import OpenAI
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes as tv_draw_bounding_boxes
from typing import Union

def is_interactive() -> bool:
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
        else:
            return False
    except NameError:
        return False  # Probably standard Python interpreter
    
# Custom object to parse YAML data
class YAMLObject:
    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                value = YAMLObject(value)
            elif isinstance(value, list):
                value = [YAMLObject(item) if isinstance(item, dict) else item for item in value]
            self.__setattr__(key, value)
    
    def serialize(self) -> dict:
        return self.__dict__

class RemoteMLLM:
    def __init__(self, model_name, api_key, base_url):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name

    def encode_base64(self, image) -> str:
        """Encode an image in tensor to base64 format."""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    
    def __call__(self, image, question) -> str:
        image_base64 = self.encode_base64(image)

        completion = self.client.chat.completions.create(
            model = self.model_name,
            temperature = 0.0,
            max_tokens = 1,
            messages = 
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        }
                    ]
                }
            ]
        )

        return completion.choices[0].message.content

    
def load_config(file_path: str) -> YAMLObject:
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return YAMLObject(data)


def denormalize(images, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225)):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


def show_single_image(image, denormalize_stats=None, bgr_image=False, save_path=None, size='small', bbox_info=None):
    if not is_interactive():
        import matplotlib
        matplotlib.use("module://imgcat")
    if size == 'size_img':
        figsize = (image.shape[2] / 100, image.shape[1] / 100)  # The default dpi of plt.savefig is 100
    elif size == 'small':
        figsize = (4, 4)
    else:
        figsize = (12, 12)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])

    if bbox_info is not None:
        image = draw_bounding_boxes(image, bbox_info['bboxes'], labels=bbox_info['labels'], colors=bbox_info['colors'],
                                    width=5)

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if denormalize_stats is not None:
            image = denormalize(image.unsqueeze(0), *denormalize_stats)
        if image.dtype == torch.float32:
            image = image.clamp(0, 1)
        ax.imshow(image.squeeze(0).permute(1, 2, 0))
    else:
        if bgr_image:
            image = image[..., ::-1]
        ax.imshow(image)

    if save_path is None:
        plt.show()
    # save image if save_path is provided
    if save_path is not None:
        # make path if it does not exist
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)


def draw_bounding_boxes(
        image: Union[torch.Tensor, Image.Image],
        bboxes: Union[list, torch.Tensor],
        width: int = 5,
        **kwargs
):
    """
    Wrapper around torchvision.utils.draw_bounding_boxes
    bboxes: [xmin, ymin, xmax, ymax]
    :return:
    """
    if isinstance(image, Image.Image):
        if type(image) == Image.Image:
            image = transforms.ToTensor()(image)
    if isinstance(bboxes, list):
        bboxes = torch.tensor(bboxes)

    image = (image * 255).to(torch.uint8).cpu()
    height = image.shape[1]
    bboxes = torch.stack([bboxes[:, 0], height - bboxes[:, 3], bboxes[:, 2], height - bboxes[:, 1]], dim=1)
    return tv_draw_bounding_boxes(image, bboxes, width=width, **kwargs)

def convert_coco(object):
    object = object.lower().strip()
    if object not in coco_classes.keys():
        if any(word in ["man", "woman", "child", "boy", "girl"] for word in object.split(" ")):
            return "person"
        if any(word in ["television"] for word in object.split(" ")):
            return "tv"
        if any(word in ["ball"] for word in object.split(" ")):
            return "sports ball"
        else:
            for word in object.split(" "):
                if word in coco_classes.keys():
                    return word
    else:
        return object
        

coco_classes = {
    '__background__': 0,
    'person': 1,
    'bicycle': 2,
    'car': 3,
    'motorcycle': 4,
    'airplane': 5,
    'bus': 6,
    'train': 7,
    'truck': 8,
    'boat': 9,
    'traffic light': 10,
    'fire hydrant': 11,
    'stop sign': 12,
    'parking meter': 13,
    'bench': 14,
    'bird': 15,
    'cat': 16,
    'dog': 17,
    'horse': 18,
    'sheep': 19,
    'cow': 20,
    'elephant': 21,
    'bear': 22,
    'zebra': 23,
    'giraffe': 24,
    'backpack': 25,
    'umbrella': 26,
    'handbag': 27,
    'tie': 28,
    'suitcase': 29,
    'frisbee': 30,
    'skis': 31,
    'snowboard': 32,
    'sports ball': 33,
    'kite': 34,
    'baseball bat': 35,
    'baseball glove': 36,
    'skateboard': 37,
    'surfboard': 38,
    'tennis racket': 39,
    'bottle': 40,
    'wine glass': 41,
    'cup': 42,
    'fork': 43,
    'knife': 44,
    'spoon': 45,
    'bowl': 46,
    'banana': 47,
    'apple': 48,
    'sandwich': 49,
    'orange': 50,
    'broccoli': 51,
    'carrot': 52,
    'hot dog': 53,
    'pizza': 54,
    'donut': 55,
    'cake': 56,
    'chair': 57,
    'couch': 58,
    'potted plant': 59,
    'bed': 60,
    'dining table': 61,
    'toilet': 62,
    'tv': 63,
    'laptop': 64,
    'mouse': 65,
    'remote': 66,
    'keyboard': 67,
    'cell phone': 68,
    'microwave': 69,
    'oven': 70,
    'toaster': 71,
    'sink': 72,
    'refrigerator': 73,
    'book': 74,
    'clock': 75,
    'vase': 76,
    'scissors': 77,
    'teddy bear': 78,
    'hair drier': 79,
    'toothbrush': 80
}
