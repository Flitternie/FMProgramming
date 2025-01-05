import PIL
import tqdm
import json
import urllib
import torch
import torchvision.transforms as transforms
from pycocotools.coco import COCO

from execution.router import *
from execution.models import object_detection_models
from execution.models import vqa_models

def download_image(url):
    img = PIL.Image.open(urllib.request.urlopen(url))
    if img.mode != 'RGB':
        return None
    img = transform(img)
    return img

with open('./EfficientAgentBench/code.json') as f:
    data = json.load(f)

def load_user_program(code):
    program_str = code
    # Execute the program string to create a function
    exec_globals = {}
    exec(program_str, exec_globals)
    return exec_globals['execute_command'], program_str

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

unloader = transforms.ToPILImage()

query_idx = 2

negative_data = torch.load(open(f"./EfficientAgentBench/data/{data[query_idx]['id']}/negative_tensors.pt", "rb"))
negative_images = negative_data[1]

code = data[query_idx]['code']

routing_system = RoutingSystem(*load_user_program(code))

image = transform(unloader(negative_images[0]))

output = routing_system.routing(image)