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


def run_static_verification_baseline(mode, config_path, data_path, log_dir):
    """
    Run the binary VQA task using a static routing configuration
    (cheapest or most expensive) for Foundation Model Programs.

    Args:
        mode (int): 0 for cheapest routing, 1 for most expensive routing.
        config_path (str): Path to the YAML config file for model initialization.
        data_path (str): Path to the JSON dataset of queries and image IDs.
        log_dir (str): Directory to store log files.
    """
    assert mode in [0, 1], "Invalid mode. Use 0 for cheapest, 1 for most expensive routing."
    
    # Initialize FM backend with the given configuration
    initialize(config_path)

    config_label = "lowest" if mode == 0 else "highest"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"baseline_{config_label}.log")
    log = open(log_path, "w", buffering=1)

    # Load the dataset of queries and image IDs
    with open(data_path) as f:
        data = json.load(f)

    # Process each query and its associated images
    for item in data:
        set_seed(42)
        query = item['query']
        code = item['code']
        log.write(f"Query: {query}\n")
        print(f"Processing Query: {query}")

        # Generate a unique identifier for image caching
        hashed_query = hash_query(query)

        # Convert program string into an executable Python function
        program_str, execute_command = load_user_program(code)

        # Initialize routing system (uses static backend per `mode`)
        routing_system = RoutingSystem(
            execute_command=execute_command,
            source=program_str,
            cost_weighting=0
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

            # Static routing decision
            routed_program, routing_decision, routing_idx = routing_system.routing(image, config=mode)

            try:
                # Execute the routed program
                output, execution_counter, execution_trace = execute_routed_program(routed_program, image)
            except Exception as e:
                warnings.warn(str(e))
                log.write(f"Img: {image_id}; Label: {label}; Output: {e}; Routing: {routing_idx};\n")
                pbar.update(1)
                continue

            # Compute cost and format output
            cost = execution_cost(execution_counter)
            output = postprocessing(output)

            # Log detailed execution info
            log.write(f"Img: {image_id}; Label: {label}; Output: {output}; Routing: {routing_idx}; Cost: {cost};\n")
            pbar.update(1)

        pbar.close()

    log.close()


if __name__ == "__main__":
    # CLI Interface
    parser = argparse.ArgumentParser(description='Run the binary VQA task with static FM Program baseline')
    parser.add_argument('--mode', type=int, required=True,
                        help='0 for cheapest routing, 1 for most expensive routing')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML config file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the JSON data file')
    parser.add_argument('--log', type=str, required=True,
                        help='Directory to store log files')
    args = parser.parse_args()

    run_static_verification_baseline(
        mode=args.mode,
        config_path=args.config,
        data_path=args.data,
        log_dir=args.log
    )
