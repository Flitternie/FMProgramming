import torch
from routing import Router
from torchvision import datasets, transforms
from tqdm import tqdm

# Data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),   # Resize to 256x256 first
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert 1 channel to 3 channels
])

# Load dataset
# testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

routing_info = {
    "class": 0
}

routing_options = {
    "class": [0] * 10
}

router = Router(routing_info, routing_options, cost_weighting=0)

pbar = tqdm(total=len(testloader))
correct = 0
for i, data in enumerate(testloader):
    inputs, labels = data
    outputs, arm_idx = router.select(inputs)
    prediction = arm_idx
    reward = 100 if int(labels) == prediction else 0
    correct += int(int(labels) == prediction)
    router.update(inputs, arm_idx, reward)
    pbar.update(1)
    pbar.set_description(f"Accuracy: {correct / (i + 1)}")


    
