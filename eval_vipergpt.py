import PIL
import tqdm
import json
import urllib
import torch
import torchvision.transforms as transforms
from pycocotools.coco import COCO

from vipergpt.router import *
from vipergpt.models import object_detection_models
from vipergpt.models import vqa_models

transform = transforms.Compose([
    transforms.ToTensor(),
])

unloader = transforms.ToPILImage()

# Function to load and execute a user program from a JSON file
def load_user_program(code):
    program_str = code
    # Execute the program string to create a function
    exec_globals = {}
    exec(program_str, exec_globals)
    return program_str, exec_globals['execute_command']

def download_image(url):
    img = PIL.Image.open(urllib.request.urlopen(url))
    if img.mode != 'RGB':
        return None
    img = transform(img)
    return img

# Initialize the COCO dataset for instance annotations, may take a few seconds
dataDir='./EfficientAgentBench/coco'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco = COCO(annFile)

# load json
with open('./EfficientAgentBench/code.json') as f:
    data = json.load(f)

baseline_positives = {
    "llava": [],
    "blip": []
}

baseline_negatives = {
    "llava": [],
    "blip": []
}

log = open("log.txt", "a+", buffering=1)
positives, negatives = [], []
for i in data[2:]:
    log.write(f"Query: {i['query']}\n")
    query = i['query']
    print(query)

    code = i['code']
    program_str, execute_command = load_user_program(code)
    routing_system = RoutingSystem(execute_command, program_str)

    print("Start loading images")

    
    negative_data = torch.load(open(f"./EfficientAgentBench/data/{i['id']}/negative_tensors.pt", "rb"))
    negative_images = negative_data[1]

    positive_image_ids = i['positive_images']
    positive_images = []
    for i in positive_image_ids:
        img_tensor = download_image(coco.loadImgs(i)[0]['coco_url'])
        positive_images.append(img_tensor)
    
    print("Finished loading images")

    pbar = tqdm.tqdm(total=len(positive_images) + len(negative_images))

    llava_positive = []
    blip_positive = []
    for image, id in zip(positive_images, positive_image_ids):
        image = unloader(image)
        llava_output = vqa_models.models[-1](image, f"Does this image contain {query.lower()}?")
        blip_output = vqa_models.models[-2](image, f"Does this image contain {query.lower()}?")

        wrapped = routing_system.routing(image)
        output = wrapped(image)

        log.write(f"Img: {id}; Label: 1; LLAVA: {llava_output}; BLIP: {blip_output}; ViperGPT: {output};\n")    
        pbar.update(1)    

    # baseline_positives["llava"].append(llava_positive)
    # baseline_positives["blip"].append(blip_positive)
    # positives.append(positive_images)

    llava_negative = []
    blip_negative = []
    for i in range(len(negative_images)):
        image = unloader(negative_images[i])
        llava_output = vqa_models.models[-1](image, f"Does this image contain {query.lower()}?")
        blip_output = vqa_models.models[-2](image, f"Does this image contain {query.lower()}?")
        llava_negative.append("yes" in llava_output)

        wrapped = routing_system.routing(image)
        output = wrapped(image)

        log.write(f"Img: {negative_data[0][i]}; Label: 0; LLAVA: {llava_output}; BLIP: {blip_output}; ViperGPT: {output};\n")
        pbar.update(1)
    pbar.close()

log.close()