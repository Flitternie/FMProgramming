import os
import tqdm
import json
import argparse
import warnings

from execution.modules import initialize
from execution.backend import (
    RoutingSystem,
    execute_routed_program,
    execution_cost
)

from utils_vqa import VqaDataset
from utils import set_seed, load_user_program


def run_static_vqa_baseline(mode, config_path, data_root, query_types, log_dir):
    """
    Run the open-form VQA task using a static FM Program configuration (cheapest or most expensive backend).

    Args:
        mode (int): 0 for cheapest backend, 1 for most expensive backend.
        config_path (str): Path to YAML configuration file for model initialization.
        data_root (str): Path to root directory containing query types and annotations.
        query_types (List[str]): Types of reasoning queries to evaluate (e.g., spatial, logical).
        log_dir (str): Directory to store log files.
    """
    # Initialize FM backend with the given configuration
    initialize(config_path)

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    config_label = "lowest" if mode == 0 else "highest"
    print(f"Running static baseline ({config_label}) for types: {query_types}")

    for type_name in query_types:
        log_path = os.path.join(log_dir, f"baseline_{type_name}_{config_label}.log")
        log = open(log_path, "w+", buffering=1)

        type_dir = os.path.join(data_root, type_name)
        meta_path = os.path.join(type_dir, "meta_annotation.json")

        with open(meta_path) as f:
            query_metadata = json.load(f)

        for query_obj in query_metadata:
            set_seed(42)
            query = query_obj["query"]
            query_idx = query_obj["query_index"]
            program_code = query_obj["program"]

            log.write(f"Query: {query}\n")
            print(f"Processing query: {query}")

            annotation_path = os.path.join(type_dir, f"{query_idx}/annotation.json")
            with open(annotation_path) as f:
                image_data = json.load(f)

            program_str, execute_command = load_user_program(program_code)

            # RoutingSystem will use fixed routing based on mode (0 = lowest, 1 = highest)
            routing_system = RoutingSystem(
                execute_command=execute_command,
                source=program_str,
                cost_weighting=0  # Static baseline ignores this; routing is fixed
            )

            print("Loading dataset...")
            dataset = VqaDataset(data_root, image_data, random_seed=42)
            print(f"Loaded {len(dataset)} images.")

            pbar = tqdm.tqdm(total=len(dataset))
            for idx in range(len(dataset)):
                image, label, img_path = dataset[idx]

                # Static backend routing based on mode
                routed_program, routing_decision, routing_idx = routing_system.routing(image, config=mode)

                try:
                    output, execution_counter, execution_trace = execute_routed_program(routed_program, image)
                except Exception as e:
                    warnings.warn(str(e))
                    log.write(f"Img: {img_path}; Label: {label}; Output: {e}; Routing: {routing_idx};\n")
                    pbar.update(1)
                    continue

                cost = execution_cost(execution_counter)

                log.write(
                    f"Img: {img_path}; Label: {label}; Output: {output}; Routing: {routing_idx}; Cost: {cost};\n"
                )
                pbar.update(1)
            pbar.close()

        log.close()


if __name__ == "__main__":
    # CLI Interface
    parser = argparse.ArgumentParser(description='Run the open-form VQA task with static FM Program baseline')
    parser.add_argument('--mode', type=int, required=True,
                        help='0 for cheapest routing, 1 for most expensive routing')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML config file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the root directory of VQA data')
    parser.add_argument('--type', type=str, nargs='+', default=None,
                        help='Query categories to evaluate (e.g. spatial logical numerical)')
    parser.add_argument('--log', type=str, required=True,
                        help='Directory to store log files')
    args = parser.parse_args()

    query_types = args.type or ['spatial', 'logical', 'numerical', 'comparison', 'knowledge']

    run_static_vqa_baseline(
        mode=args.mode,
        config_path=args.config,
        data_root=args.data,
        query_types=query_types,
        log_dir=args.log
    )
