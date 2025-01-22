import argparse
import tqdm
import json

from execution.router import *
from utils_retrieval import *
from utils import set_seed

set_seed(42)

if __name__ == "__main__":
    # Load arguments
    parser = argparse.ArgumentParser(description='Run the retrieval task')
    parser.add_argument('--cost_weighting', type=float, default=0, help='Cost weighting for the routing system')
    args = parser.parse_args()
    cost_weighting = args.cost_weighting

    # Load annotations
    with open('./data/retrieval_data.json') as f:
        data = json.load(f)

    log = open(f"./logs/log_struct_new_{cost_weighting}.txt", "a+", buffering=1)
    for i in data:
        log.write(f"Query: {i['query']}\n")
        query = i['query']
        print(query)
        # hash the query as a unique identifier
        hased_query = hased_query(query)

        code = i['code']
        program_str, execute_command = load_user_program(code)
        routing_system = RoutingSystem(execute_command, program_str, cost_weighting)

        print("Start loading images")
        positive_images, positive_image_ids, negative_images, negative_image_ids = prepare_data(i, hased_query)
        dataset = ImageRetrievalDataset(positive_images, positive_image_ids, negative_images, negative_image_ids, random_seed=42)
        print(f"Finished loading images for query: {query}")
        
        pbar = tqdm.tqdm(total=len(dataset))
        for idx in range(len(dataset)):
            image, label, id = dataset[idx]
            routed_program, routing_decision, routing_idx = routing_system.routing(image)
            try:
                output, execution_counter, execution_info = execute_routed_program(routed_program, image)
            except Exception as e:
                print(e)
                log.write(f"Img: {id}; Label: {label}; ViperGPT: {e}; Routing: {routing_idx};\n")
                pbar.update(1)
                continue
            
            reward_mapping = check_execution(execution_info, routing_system.router.routing_info)
            cost = execution_cost(execution_counter)

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
            routing_system.update_router(image, routing_idx, reward, reward_mapping)

            log.write(f"Img: {id}; Label: {label}; ViperGPT: {output}; Routing: {routing_idx}; Cost: {cost};\n")
            pbar.update(1)    
        pbar.close()

    log.close()