import tqdm
import json
import io
import base64
import PIL
import argparse
from openai import OpenAI

from execution.router import *
from utils_retrieval import *
from utils import set_seed

transform = transforms.ToPILImage()
def encode_base64_content(image) -> str:
    """Encode an image in tensor to base64 format."""
    pil_image = transform(image)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


if __name__ == "__main__":
    # Load arguments
    parser = argparse.ArgumentParser(description='Run the image verification task')
    parser.add_argument('--model', required=True, type=str, help='The model to use for the task')
    parser.add_argument('--api_key', default="EMPTY", type=str, help='The path to the API key')
    parser.add_argument('--base_url', default="http://localhost:8000/v1", type=str, help='The base URL of the API')
    parser.add_argument('--cost', type=int, help='The cost of the model')
    parser.add_argument('--data', type=str, required=True, help='Path to the data')
    args = parser.parse_args()

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )

    # Load annotations
    with open(args.data) as f:
        data = json.load(f)

    log = open(f"./logs/baseline_{args.model.replace('''/''', '_')}.log", "a+", buffering=1)
    for i in data:
        set_seed(42)
        log.write(f"Query: {i['query']}\n")
        query = i['query']
        print(query)
        # hash the query as a unique identifier
        hased_query = hash_query(query)

        print("Start loading images")
        positive_images, positive_image_ids, negative_images, negative_image_ids = prepare_data(i, hased_query)
        dataset = ImageRetrievalDataset(positive_images, positive_image_ids, negative_images, negative_image_ids, random_seed=42)
        print(f"Finished loading images for query: {query}")

        question = f"In the image, is there {query}? Answer me YES or NO."
        
        pbar = tqdm.tqdm(total=len(dataset))
        for idx in range(len(dataset)):
            image, label, id = dataset[idx]

            image_base64 = encode_base64_content(image)

            completion = client.chat.completions.create(
                model = args.model,
                temperature = 0.0,
                max_tokens = 1,
                messages = 
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
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

            output = completion.choices[0].message.content

            if "yes" in output.lower():
                output = 1
            else:
                output = 0

            log.write(f"Img: {id}; Label: {label}; Prediction: {output}; Cost: {args.cost};\n")
            pbar.update(1)    
        pbar.close()

    log.close()