from execution.models import ObjectDetection, VisualQuestionAnswering, LanguageModel
from execution.utils import load_config

object_detection_models = None
vqa_models = None
llm_models = None

cost_info = None

def initialize(config):
    global object_detection_models, vqa_models, llm_models, cost_info

    config = load_config(config)

    object_detection_models = ObjectDetection(config.object_detection, debug=config.debug)
    vqa_models = VisualQuestionAnswering(config.vqa, debug=config.debug)
    llm_models = LanguageModel(config.llm, debug=config.debug)

    cost_info = {
        "object_detection": [model.cost for model in object_detection_models.model_pool],
        "vqa": [model.cost for model in vqa_models.model_pool],
        "llm": [model.cost for model in llm_models.model_pool]
    }

def get_cost_info():
    if cost_info is None:
        raise RuntimeError("Module not initialized. Call initialize() first.")
    return cost_info

def object_detection(image, object_name, routing=None):
    coordinates, scores = object_detection_models.forward(image, object_name, routing)    
    return coordinates

def vqa(image, text, routing=None):
    return vqa_models.forward(image, text, routing)

def llm(query, routing=None):
    return llm_models.forward(query, routing)