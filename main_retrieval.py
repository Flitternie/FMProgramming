import PIL
import tqdm
import json
import hashlib
import random
import torch
import torchvision.transforms as transforms
import numpy as np

from execution.router import *
from utils_retrieval import *
from utils import set_seed

set_seed(42)

if __name__ == "__main__":
    # Load annotations
    with open('./data/retrieval_data.json') as f:
        data = json.load(f)

    for cost_weighting in [0, 0.0001, 0.001, 0.01, 0.1]:
        log = open(f"./logs/log_struct_{cost_weighting}.txt", "a+", buffering=1)
        for i in data:
            log.write(f"Query: {i['query']}\n")
            query = i['query']
            print(query)
            # hash the query as a unique identifier
            hased_query = int(hashlib.sha512(query.encode('utf-8')).hexdigest(), 16)

            code = i['code']
            program_str, execute_command = load_user_program(code)
            routing_system = RoutingSystem(execute_command, program_str, cost_weighting)

            print("Start loading images")
            positive_images, positive_image_ids, negative_images, negative_image_ids = prepare_data(i, hased_query)
            dataset = ImageRetrievalDataset(positive_images, positive_image_ids, positive_images, positive_image_ids, random_seed=42)
            print(f"Finished loading images for query: {query}")
            
            pbar = tqdm.tqdm(total=len(dataset))
            for idx in range(len(dataset)):
                image, label, id = dataset[idx]
                # llava_output = vqa_models.models[-1](image, f"Does this image contain {query.lower()}?")
                # blip_output = vqa_models.models[-2](image, f"Does this image contain {query.lower()}?")
                routed_program, routing_decision, routing_idx = routing_system.routing(image)
                try:
                    output = routed_program(image)
                except Exception as e:
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