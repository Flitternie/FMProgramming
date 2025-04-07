import io
import base64
import random
import torch
import numpy as np
import torchvision.transforms as transforms

"""
This file contains utility functions used across experiments.
"""

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to be used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_user_program(code):
    """
    Loads a user program from a string of code, and returns the program as a string 
    along with the function `execute_command` extracted from the code.

    Args:
        code (str): String of code that defines the user program.

    Returns:
        tuple: A tuple containing:
            - program_str (str): The original code.
            - execute_command (Callable): The function `execute_command` from the code.
    """
    program_str = code
    # Execute the program string to create a function
    exec_globals = {}
    exec(program_str, exec_globals)
    return program_str, exec_globals['execute_command']

def encode_base64_content(image) -> str:
    """
    Encode an image tensor to base64 format for API transmission.
    
    Args:
        image: A tensor representation of an image.
        
    Returns:
        str: Base64 encoded string of the image in JPEG format.
    """
    transform = transforms.ToPILImage()
    pil_image = transform(image)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str