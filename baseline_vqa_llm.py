import os
import tqdm
import json
import argparse
import warnings
from openai import OpenAI

from utils_vqa import VqaDataset, PROMPT
from utils import set_seed, encode_base64_content


def run_llm_vqa_baseline(model, api_key, base_url, data_root, query_types, cost, log_dir):
    """
    Run the open-form VQA task using a language model via API calls.
    
    Args:
        model (str): Name of the LLM model to use.
        api_key (str): API key for authentication.
        base_url (str): Base URL for the API endpoint.
        data_root (str): Path to root directory containing query types and annotations.
        query_types (List[str]): Types of reasoning queries to evaluate (e.g., spatial, logical).
        cost (int): Cost associated with using this model.
        log_dir (str): Directory to store log files.
    """
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Running VQA baseline for types: {query_types}")
    
    for type_name in query_types:
        log_path = os.path.join(log_dir, f"baseline_{type_name}_{model.replace('/', '_')}.log")
        log = open(log_path, "w", buffering=1)
        
        type_dir = os.path.join(data_root, type_name)
        meta_path = os.path.join(type_dir, "meta_annotation.json")
        
        with open(meta_path) as f:
            query_metadata = json.load(f)
        
        for query_obj in query_metadata:
            set_seed(42)
            query = query_obj["query"]
            query_idx = query_obj["query_index"]
            
            log.write(f"Query: {query}\n")
            print(f"Processing query: {query}")
            
            annotation_path = os.path.join(type_dir, f"{query_idx}/annotation.json")
            with open(annotation_path) as f:
                image_data = json.load(f)
            
            print("Loading dataset...")
            dataset = VqaDataset(data_root, image_data, random_seed=42)
            print(f"Loaded {len(dataset)} images.")
            
            # Format the question with instructions
            formatted_question = f"{query} {PROMPT}"
            
            pbar = tqdm.tqdm(total=len(dataset))
            for idx in range(len(dataset)):
                image, label, img_path = dataset[idx]
                
                # Convert image to base64 for API transmission
                image_base64 = encode_base64_content(image)
                
                try:
                    # Query the LLM with the image
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
                    
                    # Process the output response
                    output = completion.choices[0].message.content
                    
                except Exception as e:
                    warnings.warn(str(e))
                    log.write(f"Img: {img_path}; Label: {label}; Output: error; Cost: {cost};\n")
                    pbar.update(1)
                    continue
                
                # Log detailed execution info
                log.write(f"Img: {img_path}; Label: {label}; Output: {output}; Cost: {cost};\n")
                pbar.update(1)
                
            pbar.close()
            
        log.close()


if __name__ == "__main__":
    # CLI Interface
    parser = argparse.ArgumentParser(description='Run the open-form VQA task with LLM baseline')
    parser.add_argument('--model', type=str, required=True,
                        help='The model to use for the task')
    parser.add_argument('--api_key', type=str, default="EMPTY",
                        help='The API key for authentication')
    parser.add_argument('--base_url', type=str, default="http://128.83.141.189:8000/v1",
                        help='The base URL of the API')
    parser.add_argument('--cost', type=int, required=True,
                        help='Cost associated with using this model')
    parser.add_argument('--data', type=str, default="/home/lynie/img_generation/new_vqa",
                        help='Path to the root directory of VQA data')
    parser.add_argument('--type', type=str, nargs='+', default=None,
                        help='Query categories to evaluate (e.g. spatial logical numerical)')
    parser.add_argument('--log', type=str, required=True,
                        help='Directory to store log files')
    args = parser.parse_args()
    
    # Set default query types if not specified
    query_types = args.type or ['spatial', 'logical', 'numerical', 'comparison', 'knowledge']
    
    run_llm_vqa_baseline(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        data_root=args.data,
        query_types=query_types,
        cost=args.cost,
        log_dir=args.log
    )