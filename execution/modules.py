from execution.models import ObjectDetection, VisualQuestionAnswering, LanguageModel
from execution.utils import load_config

'''
This file contains the main API for the execution module.
'''

# Global model instances and cost information
object_detection_models = None
vqa_models = None
llm_models = None

cost_info = None

# ---------------------------------------------------------------------------- #
#                              Initialization Function                         #
# ---------------------------------------------------------------------------- #

def initialize(config_path):
    """
    Initializes all models (Object Detection, VQA, LLM) and computes their cost info.

    Args:
        config_path (str): Path to the YAML configuration file.

    Raises:
        RuntimeError: If initialization fails due to incorrect config.
    """
    global object_detection_models, vqa_models, llm_models, cost_info

    # Load configuration object from YAML
    config = load_config(config_path)

    # Initialize models with debug mode from config
    object_detection_models = ObjectDetection(config.object_detection, debug=config.debug)
    vqa_models = VisualQuestionAnswering(config.vqa, debug=config.debug)
    llm_models = LanguageModel(config.llm, debug=config.debug)

    # Extract and store model cost information
    cost_info = {
        "object_detection": [model.cost for model in object_detection_models.model_pool],
        "vqa": [model.cost for model in vqa_models.model_pool],
        "llm": [model.cost for model in llm_models.model_pool]
    }

# ---------------------------------------------------------------------------- #
#                            Cost Info Retrieval                               #
# ---------------------------------------------------------------------------- #

def get_cost_info():
    """
    Returns the cost information for all initialized models.

    Returns:
        dict: Contains cost lists for 'object_detection', 'vqa', and 'llm'.

    Raises:
        RuntimeError: If the module is not initialized.
    """
    if cost_info is None:
        raise RuntimeError("Module not initialized. Call initialize() first.")
    
    return cost_info

# ---------------------------------------------------------------------------- #
#                            Object Detection API                              #
# ---------------------------------------------------------------------------- #

def object_detection(image, object_name, routing=None):
    """
    Runs object detection on the given image.

    Args:
        image (torch.Tensor | PIL.Image): Input image for detection.
        object_name (str): Name of the object to detect.
        routing (int, optional): Index of the model to use. Defaults to None.

    Returns:
        list: Coordinates of the detected bounding boxes.
    """
    coordinates, scores = object_detection_models.forward(image, object_name, routing)
    return coordinates

# ---------------------------------------------------------------------------- #
#                      Visual Question Answering (VQA) API                     #
# ---------------------------------------------------------------------------- #

def vqa(image, text, routing=None):
    """
    Runs visual question answering on the given image and text.

    Args:
        image (torch.Tensor | PIL.Image): Input image.
        text (str): Question to ask about the image.
        routing (int, optional): Index of the model to use. Defaults to None.

    Returns:
        str: Model's answer to the question.
    """
    return vqa_models.forward(image, text, routing)

# ---------------------------------------------------------------------------- #
#                          Language Model (LLM) API                            #
# ---------------------------------------------------------------------------- #

def llm(query, routing=None):
    """
    Runs a query through the language model.

    Args:
        query (str): Text query to send to the LLM.
        routing (int, optional): Index of the model to use. Defaults to None.

    Returns:
        str: Response from the language model.
    """
    return llm_models.forward(query, routing)
