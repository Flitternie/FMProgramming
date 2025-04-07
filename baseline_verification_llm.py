import os
import tqdm
import json
import io
import re
import base64
import argparse
import warnings
from PIL import Image
from openai import OpenAI
from torchvision import transforms

from utils_verification import VerificationDataset, hash_query, prepare_data
from utils import set_seed, encode_base64_content


def run_llm_verification_baseline(model, api_key, base_url, data_path, cost, log_dir):
    """
    Run the binary VQA task using a language model via API calls.
    
    Args:
        model (str): Name of the LLM model to use.
        api_key (str): API key for authentication.
        base_url (str): Base URL for the API endpoint.
        data_path (str): Path to the JSON dataset of queries and image IDs.
        cost (int): Cost associated with using this model.
        log_dir (str): Directory to store log files.
    """
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # Create log directory if it doesn't exist
    os.makedirs(os.path.join(log_dir, "llm_routing_baseline"), exist_ok=True)
    log_path = os.path.join(log_dir, "llm_routing_baseline", f"baseline_{model.replace('/', '_')}.log")
    log = open(log_path, "a+", buffering=1)
    
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
        
        # Extract the VQA question from the code
        try:
            question = re.findall(r'query\("(.*?)"\)', code)[0]
            formatted_question = f"In the image, is there {query}? Answer me YES or NO."
            print(f"Question: {formatted_question}")
        except Exception as e:
            warnings.warn(f"Failed to extract question: {str(e)}")
            formatted_question = f"In the image, is there {query}? Answer me YES or NO."
        
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
            
            # Convert image to base64 for API transmission
            image_base64 = encode_base64_content(image)
            
            try:
                # Query the LLM with the image
                completion = client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    seed=42,
                    max_tokens=1,
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
                prediction = 1 if "yes" in output.lower() else 0
                
            except Exception as e:
                warnings.warn(str(e))
                log.write(f"Img: {image_id}; Label: {label}; Prediction: error; Cost: {cost};\n")
                pbar.update(1)
                continue
            
            # Log detailed execution info
            log.write(f"Img: {image_id}; Label: {label}; Prediction: {prediction}; Cost: {cost};\n")
            pbar.update(1)
            
        pbar.close()
    
    log.close()


if __name__ == "__main__":
    # CLI Interface
    parser = argparse.ArgumentParser(description='Run the image verification task using LLM')
    parser.add_argument('--model', type=str, required=True,
                        help='The model to use for the task')
    parser.add_argument('--api_key', type=str, default="EMPTY",
                        help='The API key for authentication')
    parser.add_argument('--base_url', type=str, default="http://localhost:8000/v1",
                        help='The base URL of the API')
    parser.add_argument('--cost', type=int, required=True,
                        help='The cost associated with using this model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the JSON data file')
    parser.add_argument('--log', type=str, default="./logs",
                        help='Directory to store log files')
    args = parser.parse_args()
    
    run_llm_verification_baseline(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        data_path=args.data,
        cost=args.cost,
        log_dir=args.log
    )