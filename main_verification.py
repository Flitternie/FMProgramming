import os
import tqdm
import json
import argparse
import warnings

from execution.modules import initialize
from execution.backend import (
    execute_routed_program,
    RoutingSystem,
    check_execution,
    execution_cost
)
from utils_verification import VerificationDataset, hash_query, prepare_data, postprocessing
from utils import set_seed, load_user_program


def run_verification_task(cost_weighting_list, config_path, data_path, log_dir):
    """
    Run the binary Visual Question Answering (VQA) task using foundation model programs
    and dynamic routing for cost-efficient inference.

    Args:
        cost_weighting_list (List[float]): Cost trade-offs to balance accuracy and inference cost.
        config_path (str): Path to the YAML config file for model initialization.
        data_path (str): Path to the JSON dataset of queries and image IDs.
        log_dir (str): Directory to store log files.
    """
    # Initialize FM backend with the given configuration
    initialize(config_path)

    # Load the dataset of queries and image IDs
    with open(data_path) as f:
        data = json.load(f)

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Loop through each cost weighting value
    for cost_weighting in cost_weighting_list:
        log_path = os.path.join(log_dir, f"log_{cost_weighting}.log")
        log = open(log_path, "w", buffering=1)

        # Process each query and its associated images
        for item in data:
            set_seed(42)
            query = item['query']
            code = item['code']
            log.write(f"Query: {query}\n")
            print(f"Processing: {query}")

            # Generate a unique identifier for image caching
            hashed_query = hash_query(query)

            # Convert program string into an executable Python function
            program_str, execute_command = load_user_program(code)

            # Initialize routing system
            routing_system = RoutingSystem(
                execute_command=execute_command,
                source=program_str,
                cost_weighting=cost_weighting,
                config="struct_reinforce"
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
                image, label, img_id = dataset[idx]

                # Dynamically select routing for the current image
                routed_program, routing_decision, routing_idx = routing_system.routing(image)

                try:
                    # Execute the routed program
                    output, execution_counter, execution_trace = execute_routed_program(routed_program, image)
                except Exception as e:
                    warnings.warn(str(e))
                    log.write(f"Img: {img_id}; Label: {label}; Output: {e}; Routing: {routing_idx};\n")
                    pbar.update(1)
                    continue

                # Compute cost and format output
                cost = execution_cost(execution_counter)
                output = postprocessing(output)

                # Analyze the routing trace
                reward_mapping = check_execution(execution_trace, routing_system.router.routing_info)

                # Reward assignment
                if int(label) == 1:  # Positive (minority) class
                    if int(output) == int(label):
                        reward = 1000  # True Positive
                    else:
                        reward = -100  # False Negative
                elif int(label) == 0:  # Negative (majority) class
                    if int(output) == int(label):
                        reward = 1  # True Negative
                    else:
                        reward = -100  # False Positive

                # Update routing policy based on the outcome
                routing_system.update_router(image, routing_idx, reward, reward_mapping)

                # Log detailed execution info
                log.write(f"Img: {img_id}; Label: {label}; Output: {output}; Routing: {routing_idx}; Cost: {cost};\n")
                pbar.update(1)

            pbar.close()

        log.close()


if __name__ == "__main__":
    # CLI Interface
    parser = argparse.ArgumentParser(description="Run the binary VQA task")
    parser.add_argument('--cost_weighting', type=float, nargs='+', required=True,
                        help='Cost weighting values for the routing system (e.g. 0.1 0.5 1.0)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML config file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the JSON data file')
    parser.add_argument('--log', type=str, required=True,
                        help='Directory to store log files')
    args = parser.parse_args()

    run_verification_task(
        cost_weighting_list=args.cost_weighting,
        config_path=args.config,
        data_path=args.data,
        log_dir=args.log
    )
