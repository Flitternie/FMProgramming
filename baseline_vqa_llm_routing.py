import os
import tqdm
import json
import argparse
import warnings
from openai import OpenAI

from execution.modules import initialize
from execution.backend import RoutingSystem

from utils_vqa import VqaDataset, equivalence, PROMPT
from utils import set_seed, load_user_program, encode_base64_content


def run_llm_routing_vqa_baseline(cost_weightings, config_path, data_root, query_types, log_dir):
    """
    Run the open-form VQA task using a bandit-based routing system to dynamically
    select between different model backends.

    Args:
        cost_weightings (list): List of cost weighting factors to test for the routing system.
        config_path (str): Path to YAML configuration file for model initialization.
        data_root (str): Path to root directory containing query types and annotations.
        query_types (List[str]): Types of reasoning queries to evaluate (e.g., spatial, logical).
        log_dir (str): Directory to store log files.
    """
    # Initialize FM backend with the given configuration
    initialize(config_path)
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Running VQA task with bandit routing for types: {query_types}")
    
    # Initialize OpenAI clients for different model endpoints
    client_small = OpenAI(
        api_key="EMPTY",
        base_url="http://128.83.141.189:8001/v1"
    )

    client_large = OpenAI(
        api_key="EMPTY",
        base_url="http://128.83.141.189:8000/v1"
    )
    
    # Test each cost weighting factor
    for cost_weighting in cost_weightings:
        # Process each query type
        for type_name in query_types:
            log_path = os.path.join(log_dir, f"log_{type_name}_{cost_weighting}.log")
            log = open(log_path, "w", buffering=1)
            
            type_dir = os.path.join(data_root, type_name)
            meta_path = os.path.join(type_dir, "meta_annotation.json")
            
            # Load query metadata for this type
            with open(meta_path) as f:
                query_metadata = json.load(f)
            
            # Process each query in the metadata
            for query_obj in query_metadata:
                set_seed(42)
                query = query_obj["query"]
                query_idx = query_obj["query_index"]
                
                log.write(f"Query: {query}\n")
                print(f"Processing query: {query}")
                
                # Load image annotations for this query
                annotation_path = os.path.join(type_dir, f"{query_idx}/annotation.json")
                with open(annotation_path) as f:
                    image_data = json.load(f)
                
                # Prepare program for routing system
                code = f'''def execute_command(image) -> str:\n    image_patch = ImagePatch(image)\n    return image_patch.query("{query}")'''
                program_str, execute_command = load_user_program(code)
                
                # Initialize bandit routing system
                routing_system = RoutingSystem(
                    execute_command=execute_command,
                    source=program_str,
                    cost_weighting=cost_weighting,
                    config="bandit"
                )
                
                # Format the question with the NA prompt
                formatted_question = f"{query} {PROMPT}"
                
                # Load the dataset
                print("Loading dataset...")
                dataset = VqaDataset(data_root, image_data, random_seed=42)
                print(f"Loaded {len(dataset)} images.")
                
                # Process each image in the dataset
                pbar = tqdm.tqdm(total=len(dataset))
                for idx in range(len(dataset)):
                    image, label, img_path = dataset[idx]
                    
                    # Get routing decision for this image
                    routed_program, routing_decision, routing_idx = routing_system.routing(image)
                    
                    # Convert image to base64 for API transmission
                    image_base64 = encode_base64_content(image)
                    
                    # Select client and model based on routing decision
                    client = client_small if int(routing_idx) == 0 else client_large
                    model = "Qwen/Qwen2.5-VL-3B-Instruct" if int(routing_idx) == 0 else "Qwen/Qwen2.5-VL-72B-Instruct"
                    
                    try:
                        # Query the selected model
                        completion = client.chat.completions.create(
                            model=model,
                            temperature=0.0,
                            seed=42,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": formatted_question
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{image_base64}"
                                            },
                                        }
                                    ]
                                }
                            ]
                        )
                        
                        # Process model output
                        output = completion.choices[0].message.content
                        
                        # Calculate cost based on model size
                        cost = 3750 if routing_idx == 0 else 73400
                        
                        # Calculate reward based on answer correctness
                        reward = 100 if equivalence(output, label) else -100
                        
                        # Update bandit router with feedback
                        routing_system.update_router(image, routing_idx, reward, None)
                        
                    except Exception as e:
                        warnings.warn(str(e))
                        log.write(f"Img: {img_path}; Label: {label}; Output: {str(e)}; Routing: {routing_idx};\n")
                        pbar.update(1)
                        continue
                    
                    # Log detailed execution info
                    log.write(f"Img: {img_path}; Label: {label}; Output: {output}; Routing: {routing_idx}; Cost: {cost};\n")
                    pbar.update(1)
                
                pbar.close()
            
            log.close()


if __name__ == "__main__":
    # CLI Interface
    parser = argparse.ArgumentParser(description='Run the VQA task with bandit routing')
    parser.add_argument('--cost_weighting', type=float, nargs='+', required=True,
                        help='Cost weighting factors for the routing system')
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
    
    run_llm_routing_vqa_baseline(
        cost_weightings=args.cost_weighting,
        config_path=args.config,
        data_root=args.data,
        query_types=query_types,
        log_dir=args.log
    )