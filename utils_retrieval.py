import PIL
import random
import tqdm
import torch
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Function to load and execute a user program from a JSON file
def load_user_program(code):
    program_str = code
    # Execute the program string to create a function
    exec_globals = {}
    exec(program_str, exec_globals)
    return program_str, exec_globals['execute_command']

# Initialize the COCO dataset for loading images
img_dir='./data/coco/train2017/{}.jpg'

def load_image(img_id):
    img_path = img_dir.format(str(img_id).zfill(12))
    img = PIL.Image.open(img_path)
    if img.mode != 'RGB':
        return None
    img = transform(img)
    return img

def prepare_data(query_data, hased_query):
    positive_image_ids = query_data['positive_images']
    try:
        positive_images = torch.load(f'./data/saved_retrieval_imgs/positive_images_{hased_query}.pt', weights_only=False)
    except:
        positive_images = [load_image(i) for i in positive_image_ids]
        torch.save(positive_images, f'./data/saved_retrieval_imgs/positive_images_{hased_query}.pt')

    negative_image_ids = query_data['negative_images']
    try:
        negative_images = torch.load(f'./data/saved_retrieval_imgs/negative_images_{hased_query}.pt', weights_only=False)
    except:
        negative_images = [load_image(i) for i in tqdm.tqdm(negative_image_ids)]
        torch.save(negative_images, f'./data/saved_retrieval_imgs/negative_images_{hased_query}.pt')
    
    return positive_images, positive_image_ids, negative_images, negative_image_ids


class ImageRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, positive_images, postive_img_ids, negative_images, negative_img_ids, random_seed=42):
        self.positive_images = positive_images
        self.positive_img_ids = postive_img_ids
        # Remove non-RGB images
        self.negative_images = [negative_image for negative_image in negative_images if negative_image is not None]
        self.negative_img_ids = [negative_img_id for negative_img_id, negative_image in zip(negative_img_ids, negative_images) if negative_image is not None]
        self.random_seed = random_seed
        self._shuffle()
    
    def _shuffle(self):
        random.seed(self.random_seed)
        self.images = self.positive_images + self.negative_images
        self.labels = [1] * len(self.positive_images) + [0] * len(self.negative_images)
        self.img_ids = self.positive_img_ids + self.negative_img_ids
        self.indices = list(range(len(self.images)))
        random.shuffle(self.indices)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        return self.images[idx], self.labels[idx], self.img_ids[idx]