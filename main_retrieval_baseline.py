import tqdm
import json
import hashlib
import argparse

from execution.router import *
from utils_retrieval import *
from utils import set_seed


set_seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieval baseline')
    parser.add_argument('--config', type=int, default=0, help='0 for cheapest routing, 1 for most expensive routing')
    args = parser.parse_args()
    config = args.config
    assert config in [0, 1], "Invalid config value. Please choose 0 for cheapest routing, 1 for most expensive routing"

    # Load annotations
    with open('./data/retrieval_data.json') as f:
        data = json.load(f)

    file_name = "lowest" if config == 0 else "highest"
    log = open(f"./logs/log_struct_{file_name}.txt", "a+", buffering=1)
    for i in data[19:]:
        log.write(f"Query: {i['query']}\n")
        query = i['query']
        print(query)
        # hash the query as a unique identifier
        hased_query = int(hashlib.sha512(query.encode('utf-8')).hexdigest(), 16)

        code = i['code']
        program_str, execute_command = load_user_program(code)
        routing_system = RoutingSystem(execute_command, program_str, cost_weighting=0)

        print("Start loading images")
        positive_images, positive_image_ids, negative_images, negative_image_ids = prepare_data(i, hased_query)
        dataset = ImageRetrievalDataset(positive_images, positive_image_ids, negative_images, negative_image_ids, random_seed=42)
        print(f"Finished loading images for query: {query}")
        
        pbar = tqdm.tqdm(total=len(dataset))
        for idx in range(len(dataset)):
            image, label, id = dataset[idx]
            # llava_output = llava(image, f"Does this image contain {query.lower()}?")
            # blip_output = vqa_models.models[-2](image, f"Does this image contain {query.lower()}?")
            routed_program, routing_decision, routing_idx = routing_system.routing(image, config=config) # config=1 for most expensive routing, config=0 for cheapest routing
            try:
                output = routed_program(image)
            except Exception as e:
                output = -1

            log.write(f"Img: {id}; Label: {label}; ViperGPT: {output}; Routing: {routing_idx};\n")
            # log.write(f"Img: {id}; Label: 1; LLAVA: {llava_output}; BLIP: {blip_output}; ViperGPT: {output};\n")    
            pbar.update(1)    
        pbar.close()

    log.close()