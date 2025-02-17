import tqdm
import json

from execution.models import VisualQuestionAnswering
from execution.router import *
from execution.utils import YAMLObject
from utils_retrieval import *
from utils import set_seed


if __name__ == "__main__":
    config = [YAMLObject(
            {
                'name': "ofa-base_3rdparty-zeroshot_vqa",
                'type': "VQA",
                'device': 0,
            }
        )]
    vqa_model = VisualQuestionAnswering(config)

    # Load annotations
    with open('./data/retrieval_data.json') as f:
        data = json.load(f)

    log = open("./logs/baseline_ofa.log", "a+", buffering=1)
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
            output = vqa_model.forward(image, question, 0)
            cost = vqa_model.model_pool[0].cost

            if "yes" in output.lower():
                output = 1
            else:
                output = 0

            log.write(f"Img: {id}; Label: {label}; Prediction: {output}; Cost: {cost};\n")
            pbar.update(1)    
        pbar.close()

    log.close()