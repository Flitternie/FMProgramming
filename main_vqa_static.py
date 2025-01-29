import os
import tqdm
import json
import argparse

from execution.modules import initialize
from execution.router import *
from utils_vqa import *
from utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the VQA baseline')
    parser.add_argument('--mode', type=int, default=0, help='0 for cheapest routing, 1 for most expensive routing')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    mode = args.mode
    config = args.config
    assert mode in [0, 1], "Invalid config value. Please choose 0 for cheapest routing, 1 for most expensive routing"
    initialize(config)

    # Load annotations
    data_dir = "./data/streaming_vqa/"
    with open(os.path.join(data_dir, './spatial/meta_annotation_edited.json')) as f:
        data = json.load(f)
    
    file_name = "lowest" if mode == 0 else "highest"
    log = open(f"./logs/vqa_baseline_{file_name}.txt", "a+", buffering=1)
    for i in data:
        set_seed(42)
        
        query = i['query']
        print(query)
        log.write(f"Query: {query}\n")

        code = i['program']
        program_str, execute_command = load_user_program(code)
        routing_system = RoutingSystem(execute_command, program_str, cost_weighting=0)
        query_idx = i['query_index']

        with open(os.path.join(data_dir, f"./spatial/{query_idx}/annotation.json")) as f:
            image_data = json.load(f)
        print("Start loading images")
        dataset = VqaDataset(data_dir, image_data, random_seed=42)
        print(f"Finished loading images for query: {query}")

        pbar = tqdm.tqdm(total=len(dataset))
        for idx in range(len(dataset)):
            image, label, id = dataset[idx]
            routed_program, routing_decision, routing_idx = routing_system.routing(image, config=mode)
            try:
                output, execution_counter, execution_trace = execute_routed_program(routed_program, image)
            except Exception as e:
                warnings.warn(e)
                log.write(f"Img: {id}; Label: {label}; ViperGPT: {e}; Routing: {routing_idx};\n")
                pbar.update(1)
                continue
            
            cost = execution_cost(execution_counter)

            log.write(f"Img: {id}; Label: {label}; ViperGPT: {output}; Routing: {routing_idx}; Cost: {cost};\n")
            pbar.update(1)    
        pbar.close()

    log.close()