import re
import torch
import torchvision.transforms as transforms
from openai import OpenAI
from agentlego.apis import load_tool

class ObjectDetection():
    def __init__(self, device=None, debug=False):        
        '''
        model_list:
            See mmdetection(https://github.com/open-mmlab/mmdetection/) for more details.
            'glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco', 232M
            'glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata', 
            'grounding_dino_swin-t_finetune_16xb2_1x_coco', 172M
            'grounding_dino_swin-b_finetune_16xb2_1x_coco', 233M
        '''
        self.device = device
        self.image_processor = transforms.ToPILImage()
        # "glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata",
        # "faster-rcnn_x101-64x4d_fpn_ms-3x_coco",
        self.model_pool = [
            {
                "model": "grounding_dino_swin-t_finetune_16xb2_1x_coco",
                "device": 0,
                "threshold": 0.25,
            },
            # {
            #     "model": "glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco",
            #     "device": 0,
            #     "threshold": 0.50,
            # },
            {
                "model": "grounding_dino_swin-b_finetune_16xb2_1x_coco",
                "device": 1,
                "threshold": 0.25,
            }

        ]
        self.initialize()

    def initialize(self):
        self.models = [load_tool('TextToBbox', model=model["model"], device=model["device"]) for model in self.model_pool]
        print("Object Detection Models Loaded")
    
    def _parse_coordinates(self, text):
        coordinates = []
        text = text.split("\n")
        for t in text:
            # Regular expression to match the pattern
            pattern = r"\((\d+), (\d+), (\d+), (\d+)\), score (\d+)"
            # Use re.match to parse the string
            match = re.match(pattern, t)
            if match:
                x1, y1, x2, y2, score = map(int, match.groups())
                coordinates.append((x1, y1, x2, y2))
        return coordinates
    
    def forward(self, image, object_name, routing):
        assert routing < len(self.models), f"Routing should be less than {len(self.models)}"
        if isinstance(image, torch.Tensor):
            image = self.image_processor(image)
        result = self.models[routing](image, object_name, threshold=self.model_pool[routing]["threshold"], top1=False)
        coordinates = self._parse_coordinates(result)
        print(f"Detected {len(coordinates)} {object_name} in the image")
        return coordinates

class VisualQuestionAnswering():
    '''
    model_list:
        'blip-base_3rdparty_vqa', 361.48M
        'blip2-opt2.7b_3rdparty-zeroshot_vqa', 3770.47M
        'flamingo_3rdparty-zeroshot_vqa', 8.22G
        'llava-7b-v1.5_vqa', 7062.90M
        'ofa-base_3rdparty-finetuned_vqa', 182.24M
        'ofa-base_3rdparty-zeroshot_vqa', 182.24M
        'otter-9b_3rdparty_vqa', 8220.45M
    '''
    def __init__(self, device=None, debug=False):
        self.device = device
        self.model_pool = [
            {
                "name": "ofa-base_3rdparty-zeroshot_vqa",
                "device": 1,
            },
            {
                "name": "blip2-opt2.7b_3rdparty-zeroshot_vqa",
                "device": 0,
            }
        ]
        self.image_processor = transforms.ToPILImage()
        self.initialize()
    
    def initialize(self):
        self.models = [load_tool('VQA', model=model["name"], device=model["device"]) for model in self.model_pool]
        print("Visual Question Answering Models Loaded")
    
    def forward(self, image, question, routing):
        assert routing < len(self.models), f"Routing should be less than {len(self.models)}"
        if isinstance(image, torch.Tensor):
            image = self.image_processor(image)
        return self.models[routing](image, question)
    
class LanguageModel():
    def __init__(self, device=None, debug=False):      
        self.model_pool = [
            {
                "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "prompt": "You are a helpful assistant. You need to answer the question and keep it as short and simple as possible, without any explainations. For example, if the question is about the capital of France, you answer 'Paris'. If the question is about a certain number, return only the number.",
            },
            {
                "model": "meta-llama/Llama-3.3-70B-Instruct",
                "prompt": "You are a helpful assistant. You need to answer the question and keep it as short and simple as possible, without any explainations. For example, if the question is about the capital of France, you answer 'Paris'. If the question is about a certain number, return only the number.",
            }
        ]
        self.initialize()

    def initialize(self):
        # load api_key from the file
        self.api_key = open("api.key", "r").read()
        self.openai = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )
    
    def forward(self, query, routing):
        assert routing < len(self.model_pool), f"Routing should be less than {len(self.model_pool)}"
        completion = self.openai.chat.completions.create(
            model=self.model_pool[routing]["model"],
            messages=[
                {"role": "system", "content": self.model_pool[routing]["prompt"]},
                {"role": "user", "content": query}
            ],
            temperature=0,
            seed=42,
        )
        return completion.choices[0].message.content
