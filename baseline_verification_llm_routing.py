import os
import tqdm
import json
import argparse
import warnings

from execution.modules import initialize
from execution.backend import (
    execute_routed_program,
    RoutingSystem,
    execution_cost
)
from utils_verification import VerificationDataset, hash_query, prepare_data, postprocessing
from utils import set_seed, load_user_program


def run_llm_routing_verification_baseline(cost_weightings, config_path, data_path, log_dir):
    """
    Run the binary VQA task using a bandit-based routing configuration
    for dynamic model selection.

    Args:
        cost_weightings (list): List of cost weighting factors to test for the routing system.
        config_path (str): Path to the YAML config file for model initialization.
        data_path (str): Path to the JSON dataset of queries and image IDs.
        log_dir (str): Directory to store log files.
    """
    # Initialize FM backend with the given configuration
    initialize(config_path)

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Load the dataset of queries and image IDs
    with open(data_path) as f:
        data = json.load(f)

    # Run tests for each specified cost weighting factor
    for cost_weighting in cost_weightings:
        log_path = os.path.join(log_dir, f"log_{cost_weighting}.log")
        log = open(log_path, "a+", buffering=1)
        
        # Process each query and its associated images
        for item in data:
            set_seed(42)
            query = item['query']
            log.write(f"Query: {query}\n")
            print(f"Processing Query: {query}")
            
            # Generate a unique identifier for image caching
            hashed_query = hash_query(query)
            
            # Prepare the program code for this query
            code = f'''def execute_command(image) -> str:\n    image_patch = ImagePatch(image)\n    return image_patch.query("{query}")'''
            program_str, execute_command = load_user_program(code)
            
            # Initialize routing system with bandit configuration
            routing_system = RoutingSystem(
                execute_command=execute_command,
                source=program_str,
                cost_weighting=cost_weighting,
                config="bandit"
            )
            
            # Load associated images for the query
            print("Loading images...")
            pos_imgs, pos_ids, neg_imgs, neg_ids = prepare_data(item, hashed_query)
            dataset = VerificationDataset(
                positive_images=pos_imgs,
                positive_img_ids=pos_ids,
                negative_images=neg_imgs,
                negative_img_ids=neg_ids,
                random_seed=42
            )
            print(f"Finished loading {len(dataset)} images.")
            
            # Iterate through each image and evaluate
            pbar = tqdm.tqdm(total=len(dataset))
            for idx in range(len(dataset)):
                image, label, image_id = dataset[idx]
                
                # Get routing decision for this image
                routed_program, routing_decision, routing_idx = routing_system.routing(image)
                
                try:
                    # Execute the routed program
                    output, execution_counter, execution_trace = execute_routed_program(routed_program, image)
                except Exception as e:
                    warnings.warn(str(e))
                    log.write(f"Img: {image_id}; Label: {label}; Output: {e}; Routing: {routing_idx};\n")
                    pbar.update(1)
                    continue
                
                # Postprocess output for consistency
                output = postprocessing(output)
                
                # Compute cost of execution
                cost = execution_cost(execution_counter)
                
                # Calculate reward based on prediction accuracy and class imbalance
                reward_mapping = None
                if int(label) == 1:  # Positive class (minority)
                    if int(output) == int(label):
                        reward = 1000  # True Positive
                    else:
                        reward = -100  # False Negative
                elif int(label) == 0:  # Negative class (majority)
                    if int(output) == int(label):
                        reward = 1  # True Negative
                    else:
                        reward = -100  # False Positive
                
                # Update routing system with reward feedback
                routing_system.update_router(image, routing_idx, reward, reward_mapping)
                
                # Log detailed execution info
                log.write(f"Img: {image_id}; Label: {label}; Output: {output}; Routing: {routing_idx}; Cost: {cost};\n")
                pbar.update(1)
                
            pbar.close()
            
        log.close()


if __name__ == "__main__":
    # CLI Interface
    parser = argparse.ArgumentParser(description='Run the image verification task with bandit routing')
    parser.add_argument('--cost_weighting', type=float, nargs='+', required=True,
                        help='Cost weighting factors for the routing system')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML config file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the JSON data file')
    parser.add_argument('--log', type=str, required=True,
                        help='Directory to store log files')
    args = parser.parse_args()
    
    run_llm_routing_verification_baseline(
        cost_weightings=args.cost_weighting,
        config_path=args.config,
        data_path=args.data,
        log_dir=args.log
    )