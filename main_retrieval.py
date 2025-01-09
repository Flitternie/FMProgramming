import PIL
import tqdm
import json
import urllib
import random
import torch
import torchvision.transforms as transforms
import numpy as np

from execution.router import *

# set seed
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

unloader = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load annotations
with open('./data/retrieval_data.json') as f:
    data = json.load(f)

# Initialize the COCO dataset for loading images
img_dir='./data/coco/train2017/{}.jpg'


# Function to load and execute a user program from a JSON file
def load_user_program(code):
    program_str = code
    # Execute the program string to create a function
    exec_globals = {}
    exec(program_str, exec_globals)
    return program_str, exec_globals['execute_command']

def load_image(img_id):
    img_path = img_dir.format(str(img_id).zfill(12))
    img = PIL.Image.open(img_path)
    if img.mode != 'RGB':
        return None
    img = transform(img)
    return img


class ImageRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, postive_images, postive_img_ids, negative_images, negative_img_ids, random_seed=42):
        self.positive_images = [unloader(img) for img in postive_images]
        self.negative_images = [unloader(img) for img in negative_images]
        self.positive_img_ids = postive_img_ids
        self.negative_img_ids = negative_img_ids
        self.random_seed = random_seed
        self._shuffle()
    
    def _shuffle(self):
        random.seed(self.random_seed)
        self.images = self.positive_images + self.negative_images
        self.labels = [1] * len(self.positive_images) + [0] * len(self.negative_images)
        self.img_ids = self.positive_img_ids + self.negative_img_ids
        self.indices = list(range(len(self.images)))
        random.shuffle(self.indices)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        return self.images[idx], self.labels[idx], self.img_ids[idx]



for cost_weighting in [0, 0.0001, 0.001, 0.01, 0.1]:
    log = open(f"log_struct_{cost_weighting}.txt", "a+", buffering=1)
    positives, negatives = [], []
    for i in data:
        log.write(f"Query: {i['query']}\n")
        query = i['query']
        print(query)

        code = i['code']
        program_str, execute_command = load_user_program(code)
        routing_system = RoutingSystem(execute_command, program_str, cost_weighting)

        print("Start loading images")

        positive_image_ids = i['positive_images']
        positive_images = []
        for i in positive_image_ids:
            img_tensor = load_image(i)
            positive_images.append(img_tensor)

        negative_image_ids = i['negative_images']
        negative_images = []
        for i in tqdm(negative_image_ids):
            img_tensor = load_image(i)
            negative_images.append(img_tensor)
        
        dataset = ImageRetrievalDataset(positive_images, positive_image_ids, negative_images, negative_image_ids, random_seed=42)
        print("Finished loading images")
        pbar = tqdm.tqdm(total=len(dataset))
        for idx in range(len(dataset)):
            image, label, id = dataset[idx]
            # llava_output = vqa_models.models[-1](image, f"Does this image contain {query.lower()}?")
            # blip_output = vqa_models.models[-2](image, f"Does this image contain {query.lower()}?")
            routed_program, routing_decision, routing_idx = routing_system.routing(image)
            try:
                output = routed_program(image)
            except:
                output = -1

            if int(label) == 1: # Positive, Minority class
                if int(output) == int(label):
                    reward = 100 # True Positive
                else:
                    reward = -100 # False Negative, or -200 to penalize more for missed positives
                
            elif int(label) == 0: # Negative, Majority class
                if int(output) == int(label): 
                    reward = 1 # True Negative
                else:
                    reward = -10 # False Positive, or -20 to penalize more for false positives
            routing_system.update_router(image, routing_idx, reward)

            log.write(f"Img: {id}; Label: {label}; ViperGPT: {output}; Routing: {routing_idx};\n")
            # log.write(f"Img: {id}; Label: 1; LLAVA: {llava_output}; BLIP: {blip_output}; ViperGPT: {output};\n")    
            pbar.update(1)    
        pbar.close()

    log.close()