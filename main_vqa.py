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
from utils_vqa import VqaDataset, equivalence
from utils import set_seed, load_user_program


def run_vqa_task(cost_weighting_list, config_path, data_root, query_types, log_dir):
    """
    Run the open-form Visual Question Answering (VQA) task using foundation model programs
    and dynamic routing for cost-efficient inference.

    Args:
        cost_weighting_list (List[float]): List of routing cost weights to evaluate.
        config_path (str): Path to YAML configuration file for model initialization.
        data_root (str): Path to root directory containing query types and annotations.
        query_types (List[str]): Types of reasoning queries to evaluate (e.g., spatial, logical).
        log_dir (str): Directory to store log files.
    """
    # Initialize FM backend with the given configuration
    initialize(config_path)

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    print(f"Running VQA baseline for types: {query_types}")

    for cost_weighting in cost_weighting_list:
        for type_name in query_types:
            log_path = os.path.join(log_dir, f"log_{type_name}_{cost_weighting}.log")
            log = open(log_path, "w", buffering=1)

            type_dir = os.path.join(data_root, type_name)
            meta_file = os.path.join(type_dir, 'meta_annotation.json')

            # Load meta queries for this reasoning type
            with open(meta_file) as f:
                data = json.load(f)

            for item in data:
                set_seed(42)
                query = item['query']
                query_idx = item['query_index']
                program_code = item['program']

                print(f"Query: {query}")
                log.write(f"Query: {query}\n")

                # Load image-level annotations for this query
                ann_path = os.path.join(type_dir, f"{query_idx}/annotation.json")
                with open(ann_path) as f:
                    image_data = json.load(f)

                # Compile user program
                program_str, execute_command = load_user_program(program_code)

                # Initialize routing system with reinforcement parameters
                routing_system = RoutingSystem(
                    execute_command=execute_command,
                    source=program_str,
                    cost_weighting=cost_weighting,
                    update_freq=16,
                    batch_size=8
                )

                # Load the VQA dataset for this query
                print("Start loading images...")
                dataset = VqaDataset(data_root, image_data, random_seed=42)
                print(f"Loaded {len(dataset)} images for query: {query}")

                pbar = tqdm.tqdm(total=len(dataset))
                for idx in range(len(dataset)):
                    image, label, img_path = dataset[idx]

                    # Route and execute program
                    routed_program, routing_decision, routing_idx = routing_system.routing(image)
                    try:
                        output, execution_counter, execution_trace = execute_routed_program(routed_program, image)
                    except Exception as e:
                        warnings.warn(str(e))
                        log.write(f"Img: {img_path}; Label: {label}; Output: {e}; Routing: {routing_idx};\n")
                        pbar.update(1)
                        continue

                    # Check trace and compute resource cost
                    reward_mapping = check_execution(execution_trace, routing_system.router.routing_info)
                    cost = execution_cost(execution_counter)

                    # Reward signal: correct = +100, incorrect = -100
                    reward = 100 if equivalence(output, label) else -100
                    routing_system.update_router(image, routing_idx, reward, reward_mapping)

                    # Log result
                    log.write(f"Img: {img_path}; Label: {label}; Output: {output}; Routing: {routing_idx}; Cost: {cost};\n")
                    pbar.update(1)

                pbar.close()
            log.close()


if __name__ == "__main__":
    # CLI Interface
    parser = argparse.ArgumentParser(description='Run the open-form VQA task')
    parser.add_argument('--cost_weighting', type=float, nargs='+', required=True,
                        help='Cost weighting for the routing system')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML config file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the root directory of VQA data')
    parser.add_argument('--type', type=str, nargs='+', default=None,
                        help='Query categories to evaluate (e.g. spatial logical numerical)')
    parser.add_argument('--log', type=str, required=True,
                        help='Directory to store log files')
    args = parser.parse_args()

    # Default to all categories if not specified
    query_types = args.type or ['spatial', 'logical', 'numerical', 'comparison', 'knowledge']

    run_vqa_task(
        cost_weighting_list=args.cost_weighting,
        config_path=args.config,
        data_root=args.data,
        query_types=query_types,
        log_dir=args.log
    )
