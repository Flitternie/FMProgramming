import os
import tqdm
import json
import argparse

from execution.modules import initialize
from execution.router import *
from utils_retrieval import *
from utils import set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the image verification baseline')
    parser.add_argument('--mode', type=int, default=0, help='0 for cheapest routing, 1 for most expensive routing')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--data', type=str, required=True, help='Path to the data')
    parser.add_argument('--log', type=str, required=True, help='Path to the log file')
    args = parser.parse_args()
    mode = args.mode
    assert mode in [0, 1], "Invalid mode value. Please choose 0 for cheapest routing, 1 for most expensive routing"
    initialize(args.config)

    # Load annotations
    with open(args.data) as f:
        data = json.load(f)

    file_name = "lowest" if mode == 0 else "highest"
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, f"baseline_{file_name}.log"), "a+", buffering=1)
    for i in data:
        set_seed(42)
        log.write(f"Query: {i['query']}\n")
        query = i['query']
        print(query)
        # hash the query as a unique identifier
        hased_query = hash_query(query)

        code = i['code']
        program_str, execute_command = load_user_program(code)
        routing_system = RoutingSystem(execute_command, program_str, cost_weighting=0)

        print("Start loading images")
        positive_images, positive_image_ids, negative_images, negative_image_ids = prepare_data(i, hased_query)
        dataset = ImageRetrievalDataset(positive_images, positive_image_ids, negative_images, negative_image_ids, random_seed=42)
        print(f"Finished loading images for query: {query}")
        
        pbar = tqdm.tqdm(total=len(dataset))
        for idx in range(len(dataset)):
            image, label, id = dataset[idx]
            routed_program, routing_decision, routing_idx = routing_system.routing(image, config=mode)
            try:
                output, execution_counter, execution_trace = execute_routed_program(routed_program, image)
            except Exception as e:
                log.write(f"Img: {id}; Label: {label}; Output: {e}; Routing: {routing_idx};\n")
                pbar.update(1)
                continue
            
            cost = execution_cost(execution_counter)
            output = postprocessing(output)

            log.write(f"Img: {id}; Label: {label}; Output: {output}; Routing: {routing_idx}; Cost: {cost};\n")
            pbar.update(1)    
        pbar.close()

    log.close()