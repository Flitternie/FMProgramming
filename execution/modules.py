from execution.models import ObjectDetection, VisualQuestionAnswering, LanguageModel

debug_mode = False

object_detection_models = ObjectDetection(debug=debug_mode)
vqa_models = VisualQuestionAnswering(debug=debug_mode)
llm_models = LanguageModel(debug=debug_mode)

cost_info = {
    "object_detection": [model["cost"] for model in object_detection_models.model_pool],
    "vqa": [model["cost"] for model in vqa_models.model_pool],
    "llm": [model["cost"] for model in llm_models.model_pool]
}

def object_detection(image, object_name, routing=None):
    coordinates, scores = object_detection_models.forward(image, object_name, routing)
    return coordinates

def vqa(image, text, routing=None):
    return vqa_models.forward(image, text, routing)

def llm(query, routing=None):
    return llm_models.forward(query, routing)