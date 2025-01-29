import os
import PIL
import random
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


def load_image(img_path):
    img = PIL.Image.open(img_path)
    if img.mode != 'RGB':
        return None
    img = transform(img)
    return img


def prepare_data(data):
    image_paths = []
    answers = []
    for item in data:
        image_paths.append(item['image'])
        answers.append(item['answer'])
    return image_paths, answers


class VqaDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_data, random_seed=42):
        self.data = image_data
        self.data_dir = data_dir
        self.random_seed = random_seed
        self._initiate()
        self._shuffle()
    
    def _initiate(self):
        self.images = []
        self.labels = []
        self.image_paths = []
        for data in self.data:
            image_path = os.path.join(self.data_dir, data['image'])
            image = load_image(image_path)
            if image is not None:
                self.images.append(image)
                self.labels.append(data['answer'])
                self.image_paths.append(data['image'])
    
    def _shuffle(self):
        random.seed(self.random_seed)
        assert len(self.images) == len(self.labels)
        self.indices = list(range(len(self.images)))
        random.shuffle(self.indices)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        return self.images[idx], self.labels[idx], self.image_paths[idx]