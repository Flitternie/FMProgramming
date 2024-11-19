"""
Adding a new functionality is easy. Just implement your new model as a subclass of BaseModel.
The code will make the rest: it will make it available for the processes to call by using
process(name, *args, **kwargs), where *args and **kwargs are the arguments of the models process() method.
"""

import abc
import openai
import os
import re
import timeit
import torch
import torchvision
import warnings
from PIL import Image
from collections import Counter
from itertools import chain
from joblib import Memory
from torch import hub
from torch.nn import functional as F
from torchvision import transforms
from typing import List, Union
from omegaconf import OmegaConf

from vipergpt.utils import HiddenPrints
from vipergpt.tools import ObjectDetection, VisualQuestionAnswering

configs = [OmegaConf.load('./vipergpt/config.yaml')]
# unsafe_merge makes the individual configs unusable, but it is faster
config = OmegaConf.unsafe_merge(*configs)

cache = Memory('cache/' if config.use_cache else None, verbose=0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------- Base abstract model --------------------------- #

class BaseModel(abc.ABC):
    to_batch = False
    seconds_collect_data = 1.5  # Window of seconds to group inputs, if to_batch is True
    max_batch_size = 10  # Maximum batch size, if to_batch is True. Maximum allowed by OpenAI
    requires_gpu = True

    def __init__(self, gpu_number=0):
        self.dev = f'cuda:{gpu_number}' if device == 'cuda' else device

    # @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        If to_batch is True, every arg and kwarg will be a list of inputs, and the output should be a list of outputs.
        The way it is implemented in the background, if inputs with defaults are not specified, they will take the
        default value, but still be given as a list to the forward method.
        """
        pass

    # @classmethod
    # @abc.abstractmethod
    def name(cls) -> str:
        """The name of the model has to be given by the subclass"""
        pass

    # @classmethod
    def list_processes(cls):
        """
        A single model can be run in multiple processes, for example if there are different tasks to be done with it.
        If multiple processes are used, override this method to return a list of strings.
        Remember the @classmethod decorator.
        If we specify a list of processes, the self.forward() method has to have a "process_name" parameter that gets
        automatically passed in.
        See GPT3Model for an example.
        """
        return [cls.name]

# ------------------------------ Specific models ---------------------------- #


class ObjectDetector(BaseModel):
    name = 'object_detector'

    def __init__(self, gpu_number=0):
        super().__init__(gpu_number)

        with HiddenPrints('ObjectDetector'):
            detection_model = hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).to(self.dev)
            detection_model.eval()

        self.detection_model = detection_model

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        """get_object_detection_bboxes"""
        input_batch = image.to(self.dev).unsqueeze(0)  # create a mini-batch as expected by the model
        detections = self.detection_model(input_batch)
        p = detections['pred_boxes']
        p = torch.stack([p[..., 0], 1 - p[..., 3], p[..., 2], 1 - p[..., 1]], -1)  # [left, lower, right, upper]
        detections['pred_boxes'] = p
        return detections

class CLIPModel(BaseModel):
    name = 'clip'

    def __init__(self, gpu_number=0, version="ViT-L/14@336px"):  # @336px
        super().__init__(gpu_number)

        import clip
        self.clip = clip

        with HiddenPrints('CLIP'):
            model, preprocess = clip.load(version, device=self.dev)
            model.eval()
            model.requires_grad_ = False
        self.model = model
        self.negative_text_features = None
        self.transform = self.get_clip_transforms_from_tensor(336 if "336" in version else 224)

    # @staticmethod
    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    # @staticmethod
    def get_clip_transforms_from_tensor(self, n_px=336):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(n_px),
            self._convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    @torch.no_grad()
    def binary_score(self, image: torch.Tensor, prompt, negative_categories=None):
        is_video = isinstance(image, torch.Tensor) and image.ndim == 4
        if is_video:  # video
            image = torch.stack([self.transform(image[i]) for i in range(image.shape[0])], dim=0)
        else:
            image = self.transform(image).unsqueeze(0).to(self.dev)

        prompt_prefix = "photo of "
        prompt = prompt_prefix + prompt

        if negative_categories is None:
            if self.negative_text_features is None:
                self.negative_text_features = self.clip_negatives(prompt_prefix)
            negative_text_features = self.negative_text_features
        else:
            negative_text_features = self.clip_negatives(prompt_prefix, negative_categories)

        text = self.clip.tokenize([prompt]).to(self.dev)

        image_features = self.model.encode_image(image.to(self.dev))
        image_features = F.normalize(image_features, dim=-1)

        pos_text_features = self.model.encode_text(text)
        pos_text_features = F.normalize(pos_text_features, dim=-1)

        text_features = torch.concat([pos_text_features, negative_text_features], axis=0)

        # run competition where we do a binary classification
        # between the positive and all the negatives, then take the mean
        sim = (100.0 * image_features @ text_features.T).squeeze(dim=0)
        if is_video:
            query = sim[..., 0].unsqueeze(-1).broadcast_to(sim.shape[0], sim.shape[-1] - 1)
            others = sim[..., 1:]
            res = F.softmax(torch.stack([query, others], dim=-1), dim=-1)[..., 0].mean(-1)
        else:
            res = F.softmax(torch.cat((sim[0].broadcast_to(1, sim.shape[0] - 1),
                                       sim[1:].unsqueeze(0)), dim=0), dim=0)[0].mean()
        return res

    @torch.no_grad()
    def clip_negatives(self, prompt_prefix, negative_categories=None):
        if negative_categories is None:
            with open('useful_lists/random_negatives.txt') as f:
                negative_categories = [x.strip() for x in f.read().split()]
        # negative_categories = negative_categories[:1000]
        # negative_categories = ["a cat", "a lamp"]
        negative_categories = [prompt_prefix + x for x in negative_categories]
        negative_tokens = self.clip.tokenize(negative_categories).to(self.dev)

        negative_text_features = self.model.encode_text(negative_tokens)
        negative_text_features = F.normalize(negative_text_features, dim=-1)

        return negative_text_features

    @torch.no_grad()
    def classify(self, image: Union[torch.Tensor, list], categories: list[str], return_index=True):
        is_list = isinstance(image, list)
        if is_list:
            assert len(image) == len(categories)
            image = [self.transform(x).unsqueeze(0) for x in image]
            image_clip = torch.cat(image, dim=0).to(self.dev)
        elif len(image.shape) == 3:
            image_clip = self.transform(image).to(self.dev).unsqueeze(0)
        else:  # Video (process images separately)
            image_clip = torch.stack([self.transform(x) for x in image], dim=0).to(self.dev)

        # if len(image_clip.shape) == 3:
        #     image_clip = image_clip.unsqueeze(0)

        prompt_prefix = "photo of "
        categories = [prompt_prefix + x for x in categories]
        categories = self.clip.tokenize(categories).to(self.dev)

        text_features = self.model.encode_text(categories)
        text_features = F.normalize(text_features, dim=-1)

        image_features = self.model.encode_image(image_clip)
        image_features = F.normalize(image_features, dim=-1)

        if image_clip.shape[0] == 1:
            # get category from image
            softmax_arg = image_features @ text_features.T  # 1 x n
        else:
            if is_list:
                # get highest category-image match with n images and n corresponding categories
                softmax_arg = (image_features @ text_features.T).diag().unsqueeze(0)  # n x n -> 1 x n
            else:
                softmax_arg = (image_features @ text_features.T)

        similarity = (100.0 * softmax_arg).softmax(dim=-1).squeeze(0)
        if not return_index:
            return similarity
        else:
            result = torch.argmax(similarity, dim=-1)
            if result.shape == ():
                result = result.item()
            return result

    @torch.no_grad()
    def compare(self, images: list[torch.Tensor], prompt, return_scores=False):
        images = [self.transform(im).unsqueeze(0).to(self.dev) for im in images]
        images = torch.cat(images, dim=0)

        prompt_prefix = "photo of "
        prompt = prompt_prefix + prompt

        text = self.clip.tokenize([prompt]).to(self.dev)

        image_features = self.model.encode_image(images.to(self.dev))
        image_features = F.normalize(image_features, dim=-1)

        text_features = self.model.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        sim = (image_features @ text_features.T).squeeze(dim=-1)  # Only one text, so squeeze

        if return_scores:
            return sim
        res = sim.argmax()
        return res

    def forward(self, image, prompt, task='score', return_index=True, negative_categories=None, return_scores=False):
        if task == 'classify':
            categories = prompt
            clip_sim = self.classify(image, categories, return_index=return_index)
            out = clip_sim
        elif task == 'score':
            clip_score = self.binary_score(image, prompt, negative_categories=negative_categories)
            out = clip_score
        else:  # task == 'compare'
            idx = self.compare(image, prompt, return_scores)
            out = idx
        if not isinstance(out, int):
            out = out.cpu()
        return out

class BLIPModel(BaseModel):
    name = 'blip'
    to_batch = False
    max_batch_size = 32
    seconds_collect_data = 0.2  # The queue has additionally the time it is executing the previous forward pass

    def __init__(self, size=None, gpu_number=0, half_precision=config.blip_half_precision, blip_v2_model_type=config.blip_v2_model_type):
        super().__init__(gpu_number)

        # from lavis.models import load_model_and_preprocess
        from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, \
            Blip2ForConditionalGeneration
        
        if size == 0:
            blip_v2_model_type = 'blip2-flan-t5-xl'
        elif size == 1:
            blip_v2_model_type = 'blip2-flan-t5-xxl'
        else:
            raise ValueError("Size must be 0 or 1")

        # https://huggingface.co/models?sort=downloads&search=Salesforce%2Fblip2-
        assert blip_v2_model_type in ['blip2-flan-t5-xxl', 'blip2-flan-t5-xl', 'blip2-opt-2.7b', 'blip2-opt-6.7b',
                                      'blip2-opt-2.7b-coco', 'blip2-flan-t5-xl-coco', 'blip2-opt-6.7b-coco']

        with warnings.catch_warnings(), HiddenPrints("BLIP"), torch.cuda.device(self.dev):
            max_memory = {gpu_number: torch.cuda.mem_get_info(self.dev)[0]}

            self.processor = Blip2Processor.from_pretrained(f"Salesforce/{blip_v2_model_type}")
            # Device_map must be sequential for manual GPU selection
            try:
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    f"Salesforce/{blip_v2_model_type}", load_in_8bit=half_precision,
                    torch_dtype=torch.float16 if half_precision else "auto",
                    device_map="sequential", max_memory=max_memory
                )
            except Exception as e:
                # Clarify error message. The problem is that it tries to load part of the model to disk.
                if "had weights offloaded to the disk" in e.args[0]:
                    extra_text = ' You may want to consider setting half_precision to True.' if half_precision else ''
                    raise MemoryError(f"Not enough GPU memory in GPU {self.dev} to load the model.{extra_text}")
                else:
                    raise e

        self.qa_prompt = "Question: {} Short answer:"
        self.caption_prompt = "a photo of"
        self.half_precision = half_precision
        self.max_words = 50

    @torch.no_grad()
    def caption(self, image, prompt=None):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.dev, torch.float16)
        generated_ids = self.model.generate(**inputs, length_penalty=1., num_beams=5, max_length=30, min_length=1,
                                            do_sample=False, top_p=0.9, repetition_penalty=1.0,
                                            num_return_sequences=1, temperature=1)
        generated_text = [cap.strip() for cap in
                          self.processor.batch_decode(generated_ids, skip_special_tokens=True)]
        return generated_text
    
    def pre_question(self, question):
        # from LAVIS blip_processors
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question

    @torch.no_grad()
    def qa(self, image, question):
        inputs = self.processor(images=image, text=question, return_tensors="pt", padding="longest").to(self.dev)
        if self.half_precision:
            inputs['pixel_values'] = inputs['pixel_values'].half()
        generated_ids = self.model.generate(**inputs, length_penalty=-1, num_beams=5, max_length=10, min_length=1,
                                            do_sample=False, top_p=0.9, repetition_penalty=1.0,
                                            num_return_sequences=1, temperature=1)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def forward(self, image, question=None, task='caption'):
        if not self.to_batch:
            image, question, task = [image], [question], [task]

        if len(image) > 0 and 'float' in str(image[0].dtype) and image[0].max() <= 1:
            image = [im * 255 for im in image]

        # Separate into qa and caption batches.
        prompts_qa = [self.qa_prompt.format(self.pre_question(q)) for q, t in zip(question, task) if t == 'qa']
        images_qa = [im for i, im in enumerate(image) if task[i] == 'qa']
        images_caption = [im for i, im in enumerate(image) if task[i] == 'caption']

        with torch.cuda.device(self.dev):
            response_qa = self.qa(images_qa, prompts_qa) if len(images_qa) > 0 else []
            response_caption = self.caption(images_caption) if len(images_caption) > 0 else []

        response = []
        for t in task:
            if t == 'qa':
                response.append(response_qa.pop(0))
            else:
                response.append(response_caption.pop(0))

        if not self.to_batch:
            response = response[0]
        return response


# clip_model = CLIPModel()
# def clip(image, text, task='score', negative_categories=None):
#     print("CLIP")
#     return clip_model.forward(image, text, task=task, negative_categories=negative_categories)
# print("CLIP model loaded")

# glip_model_small = GLIPModel(size=0, gpu_number=0)
# glip_model_large = GLIPModel(size=1, gpu_number=0)
# def glip(image, text, routing):
#     if routing == 0:
#         print("GLIP small")
#         return glip_model_small.forward(image, text)
#     else:
#         print("GLIP large")
#         return glip_model_large.forward(image, text)
# print("GLIP model loaded")

# blip_model_small = BLIPModel(size=0, gpu_number=1)
# blip_model_large = BLIPModel(size=1, gpu_number=1)
# def blip(image, text, task, routing):
#     if routing == 0:
#         print("BLIP small")
#         return blip_model_small.forward(image, text, task=task)
#     else:
#         print("BLIP large")
#         return blip_model_large.forward(image, text, task=task)
# print("BLIP model loaded")


# tcl_model = TCLModel()
# def tcl(image, texts, task='score'):
    # return tcl_model.forward(image, texts, task=task)
# print("TCL model loaded")

# xvlm_model = XVLMModel()
# def xvlm(image, text, task='score', negative_categories=None):
    # return xvlm_model.forward(image, text, task=task, negative_categories=negative_categories)
# print("XVLM model loaded")

# maskrcnn_model = MaskRCNNModel()
# def maskrcnn(image):
    # return maskrcnn_model.forward(image)
# print("MaskRCNN model loaded")

# depth_model = DepthEstimationModel()
# print("Depth model loaded")

object_detection_models = ObjectDetection("cuda:0")
vqa_models = VisualQuestionAnswering("cuda:1")

def object_detection(image, object_name, routing=None):
    return object_detection_models.forward(image, object_name, routing)

def vqa(image, text, routing=None):
    return vqa_models.forward(image, text, routing)

def image_text_matching(image, category, negative_categories=None, routing=None):
    raise NotImplementedError
    model = 'clip'
    if model == 'clip':
        res = clip(image, category, task='score', negative_categories=negative_categories)
    elif model == 'tcl':
        res = tcl(image, category, task='score')
    else:  # xvlm
        task = 'binary_score' if negative_categories is not None else 'score'
        res = xvlm(image, category, task='score', negative_categories=negative_categories)
        res = res.item()
    return res

def image_text_classify(image, text, routing=None):
    raise NotImplementedError
    model = 'clip'
    if model == 'clip':
        selected = clip(image, text, task='classify')
    elif model == 'tcl':
        selected = tcl(image, text, task='classify')
    elif model == 'xvlm':
        res = xvlm(image, text, task='score')
        res = res.argmax().item()
        selected = res
    return selected

def batched_image_text_matching(images, content, negative_categories=None, routing=None):
    raise NotImplementedError
    scores = []
    model = 'clip'
    for cont in content:
        if model == 'clip':
            res = clip([p.cropped_image for p in images], cont, task='compare', return_scores=True)
        else:
            res = xvlm([p.cropped_image for p in images], cont, task='score')
        scores.append(res)
    scores = torch.stack(scores).mean(dim=0)
    scores = scores.argmax().item()  # Argmax over all image patches
    return scores

def llm(model, prompt, routing=None):
    raise NotImplementedError

def depth(image):
    raise NotImplementedError