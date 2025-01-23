import tqdm
import json
import argparse

from execution.router import *
from utils_vqa import *
from utils import set_seed

set_seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieval baseline')
    parser.add_argument('--config', type=int, default=0, help='0 for cheapest routing, 1 for most expensive routing')
    args = parser.parse_args()
    config = args.config
    assert config in [0, 1], "Invalid config value. Please choose 0 for cheapest routing, 1 for most expensive routing"

    # Load annotations
    with open('./data/vqa_data.json') as f:
        data = json.load(f)
    
    file_name = "lowest" if config == 0 else "highest"
    log = open(f"./logs/log_{file_name}.txt", "a+", buffering=1)
    for i in data:
        log.write(f"Query: {i['query']}\n")
        query = i['query']
        print(query)

        code = i['code']
        program_str, execute_command = load_user_program(code)
        routing_system = RoutingSystem(execute_command, program_str, cost_weighting=0)

        image_paths, answers = prepare_data(i)

        print("Start loading images")
        dataset = VqaDataset(image_paths, answers, random_seed=42)
        print(f"Finished loading images for query: {query}")
        
        pbar = tqdm.tqdm(total=len(dataset))
        for idx in range(len(dataset)):
            image, label = dataset[idx]
            routed_program, routing_decision, routing_idx = routing_system.routing(image)
            try:
                output, execution_counter, execution_trace = execute_routed_program(routed_program, image)
            except Exception as e:
                warnings.warn(e)
                log.write(f"Img: {id}; Gold: {label}; ViperGPT: {e}; Routing: {routing_idx};\n")
                pbar.update(1)
                continue
            
            cost = execution_cost(execution_counter)

            log.write(f"Img: {id}; Label: {label}; ViperGPT: {output}; Routing: {routing_idx}; Cost: {cost};\n")
            pbar.update(1)    
        pbar.close()

    log.close()