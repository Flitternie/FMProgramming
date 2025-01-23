import tqdm
import json
import argparse

from execution.router import *
from utils_vqa import *
from utils import set_seed

set_seed(42)

if __name__ == "__main__":
    # Load arguments
    parser = argparse.ArgumentParser(description='Run the vqa task')
    parser.add_argument('--cost_weighting', type=float, default=0, help='Cost weighting for the routing system')
    args = parser.parse_args()
    cost_weighting = args.cost_weighting

    # Load annotations
    with open('./data/vqa_data.json') as f:
        data = json.load(f)

    log = open(f"./logs/log_struct_new_{cost_weighting}.txt", "a+", buffering=1)
    for i in data:
        log.write(f"Query: {i['query']}\n")
        query = i['query']
        print(query)

        code = i['code']
        program_str, execute_command = load_user_program(code)
        routing_system = RoutingSystem(execute_command, program_str, cost_weighting)

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
            
            if output is None:
                log.write(f"Img: {id}; Label: {label}; ViperGPT: {output}; Routing: {routing_idx}; Cost: {cost};\n")
                pbar.update(1)   
                continue 
            
            reward_mapping = check_execution(execution_trace, routing_system.router.routing_info)
            cost = execution_cost(execution_counter)

            if label.lower().strip() == output.lower().strip():
                reward = 1
            else:
                reward = -1
            routing_system.update_router(image, routing_idx, reward, reward_mapping)

            log.write(f"Img: {id}; Label: {label}; ViperGPT: {output}; Routing: {routing_idx}; Cost: {cost};\n")
            pbar.update(1)    
        pbar.close()

    log.close()