# %% 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tqdm import tqdm
import torch
import requests
from PIL import Image
from io import BytesIO
from pycocotools.coco import COCO
from pymilvus import MilvusClient
from transformers import SiglipProcessor, SiglipModel, SiglipTokenizer

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Define the model ID
model_ID = "google/siglip-so400m-patch14-384"
# Get model, processor & tokenizer

siglip_model = SiglipModel.from_pretrained(model_ID).to(device)
# Get the processor
siglip_processor = SiglipProcessor.from_pretrained(model_ID)
# Get the tokenizer
siglip_tokenizer = SiglipTokenizer.from_pretrained(model_ID)


# %%
def get_single_image_embedding(my_image, processor, model, device):
  image = processor(
      text = None,
      images = my_image,
      return_tensors="pt"
      )["pixel_values"].to(device)
  embedding = model.get_image_features(image)
  # convert the embeddings to numpy array
  return embedding.cpu().detach().numpy()

def get_single_text_embedding(text): 
  inputs = siglip_tokenizer(text, return_tensors = "pt").to(device)
  text_embeddings = siglip_model.get_text_features(**inputs)
  # convert the embeddings to numpy array
  embedding_as_np = text_embeddings.cpu().detach().numpy()
  return embedding_as_np


def get_image(image_URL):
   response = requests.get(image_URL)
   image = Image.open(BytesIO(response.content)).convert("RGB")
   return image

def check_valid_URL(image_URL):
    try:
        response = requests.get(image_URL)
        if response.status_code != 200:
            return False
        Image.open(BytesIO(response.content))
        return True
    except Exception:
        return False


client = MilvusClient("image_retrieval_siglip.db")

# %%
# search for all folders under the directory "./EfficientAgentBench/data/"

data_dir = "./EfficientAgentBench/data/"
folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
for image_id in folders:
    print(f"Processing {image_id}")
    positive_path = os.path.join(data_dir, image_id, "positive_tensors.pt")
    negative_path = os.path.join(data_dir, image_id, "negative_tensors.pt")
    positive_tensors = torch.load(positive_path)
    negative_tensors = torch.load(negative_path)

    # Create a collection in quick setup mode
    if client.has_collection(collection_name=f"img_{image_id}"):
        client.drop_collection(collection_name=f"img_{image_id}")
    client.create_collection(
        collection_name=f"img_{image_id}",
        vector_field_name="vector",
        dimension=1152,
        auto_id=False,
        enable_dynamic_field=True,
        metric_type="COSINE",
    )
    pbar = tqdm(total=len(positive_tensors[0]) + len(negative_tensors[0]))

    for img in range(len(positive_tensors[0])):
        img_embedding = get_single_image_embedding(positive_tensors[0][img], siglip_processor, siglip_model, device)
        client.insert(
                f"img_{image_id}",
                {
                    "vector": img_embedding.flatten().tolist(), 
                    "label": 1,
                    "id": positive_tensors[1][img]
                },
            )
        pbar.update(1)
    for img in range(len(negative_tensors[0])):
        try:
            img_embedding = get_single_image_embedding(negative_tensors[0][img], siglip_processor, siglip_model, device)
            client.insert(
                    f"img_{image_id}",
                    {
                        "vector": img_embedding.flatten().tolist(), 
                        "label": 0,
                        "id": negative_tensors[1][img]
                    },
                )
            pbar.update(1)
        except Exception as e:
            print(f"Error: {e} in processing {image_id}")
            pbar.update(1)
    pbar.close()
