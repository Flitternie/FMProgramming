import torch
from joblib import Memory

from omegaconf import OmegaConf

from execution.utils import HiddenPrints
from execution.apis import ObjectDetection, VisualQuestionAnswering

configs = [OmegaConf.load('./execution/config.yaml')]
# unsafe_merge makes the individual configs unusable, but it is faster
config = OmegaConf.unsafe_merge(*configs)

cache = Memory('cache/' if config.use_cache else None, verbose=0)
device = "cuda" if torch.cuda.is_available() else "cpu"

object_detection_models = ObjectDetection()
vqa_models = VisualQuestionAnswering()

def object_detection(image, object_name, routing=None):
    return object_detection_models.forward(image, object_name, routing)

def vqa(image, text, routing=None):
    return vqa_models.forward(image, text, routing)

def llm(model, prompt, routing=None):
    raise NotImplementedError