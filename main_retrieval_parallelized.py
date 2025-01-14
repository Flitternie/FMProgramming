import os
import PIL
import tqdm
import json
import hashlib
import random
import torch
import torchvision.transforms as transforms
import numpy as np
from multiprocessing import Pool, Lock, set_start_method

from execution.router import *
from utils_retrieval import *
from utils import set_seed

set_seed(42)


def init_worker(lock, log_path):
    """Initializer for each worker process."""
    global log_lock, log_file
    log_lock = lock
    log_file = log_path

def process_query(args):
    query_data, cost_weighting, device = args
    query = query_data['query']
    hased_query = int(hashlib.sha512(query.encode('utf-8')).hexdigest(), 16)
    code = query_data['code']
    program_str, execute_command = load_user_program(code)
    routing_system = RoutingSystem(execute_command, program_str, cost_weighting, struct=True)

    try:
        positive_images, positive_image_ids, negative_images, negative_image_ids = prepare_data(query_data, hased_query)
        dataset = ImageRetrievalDataset(positive_images, positive_image_ids, negative_images, negative_image_ids, random_seed=42)
        print(f"Finished loading images for query: {query}")

        for idx in tqdm.tqdm(range(len(dataset))):
            image, label, img_id = dataset[idx]
            routed_program, routing_decision, routing_idx = routing_system.routing(image)
            try:
                output = routed_program(image)
            except Exception as e:
                output = -1

            if int(label) == 1:  # Positive class
                reward = 100 if int(output) == int(label) else -100
            else:  # Negative class
                reward = 1 if int(output) == int(label) else -10

            routing_system.update_router(image, routing_idx, reward)

            with log_lock:
                with open(log_file, "a+") as log:
                    log.write(f"Query: {query}; Img: {img_id}; Label: {label}; ViperGPT: {output}; Routing: {routing_idx};\n")

        print(f"Processing complete for query: {query}")

    except Exception as e:
        print(f"Error processing query {query}: {e}")
        raise e

if __name__ == "__main__":
    # Load annotations
    with open('./data/retrieval_data.json') as f:
        data = json.load(f)

    try:
        set_start_method("spawn", force=True)
    except RuntimeError as e:
        print("Start method already set.", e)

    available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
    cost_weightings = [0, 0.0001, 0.001, 0.01, 0.1]

    # Set the number of workers explicitly
    num_workers = min(len(data), len(available_devices) * 2)  # Example: 2 workers per device
    print(f"Using {num_workers} workers for parallel processing.")

    for cost_weighting in cost_weightings:
        log_lock = Lock()
        log_path = f"./logs/log_struct_parallel_{cost_weighting}.txt"

        with Pool(processes=num_workers, initializer=init_worker, initargs=(log_lock, log_path)) as pool:
            args = [(query_data, cost_weighting, available_devices[i % len(available_devices)]) for i, query_data in enumerate(data)]
            pool.map(process_query, args)
