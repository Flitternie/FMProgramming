import re
import torch
import torchvision.transforms as transforms
from openai import OpenAI
from agentlego.apis import load_tool
from execution.utils import convert_coco


class ObjectDetection():
    def __init__(self, config, debug=False):        
        '''
        model_list:
            See mmdetection(https://github.com/open-mmlab/mmdetection/) for more details.
        '''
        self.image_processor = transforms.ToPILImage()
        self.model_pool = config
        self.initialize()

    def initialize(self):
        self.models = [load_tool(model.type, model=model.name, device=model.device) for model in self.model_pool]
        self._count_parameters()
        print("Object Detection Models Loaded")
    
    def _parse_coordinates(self, text):
        coordinates = []
        scores = []
        text = text.split("\n")
        for t in text:
            # Regular expression to match the pattern
            pattern = r"\((\d+), (\d+), (\d+), (\d+)\), score (\d+)"
            # Use re.match to parse the string
            match = re.match(pattern, t)
            if match:
                x1, y1, x2, y2, score = map(int, match.groups())
                coordinates.append((x1, y1, x2, y2))
                scores.append(score)
        return coordinates, scores
    
    def _count_parameters(self):
        for idx in range(len(self.models)):
            self.models[idx].setup()
            num_params = sum(p.numel() for p in self.models[idx]._inferencer.model.parameters())
            self.model_pool[idx].cost = num_params // 1e6
            print(f"Model {idx} has {num_params // 1e6}M parameters")

    def forward(self, image, object_name, routing):
        assert routing < len(self.models), f"Routing should be less than {len(self.models)}"
        if isinstance(image, torch.Tensor):
            image = self.image_processor(image)

        if self.model_pool[routing].type == "TextToBbox":
            result = self.models[routing](image, object_name)
            # coordinates, scores = self._parse_coordinates(result)
            result = result[result.scores > self.model_pool[routing].threshold]
            coordinates, scores = result.bboxes, result.scores

        elif self.model_pool[routing].type == "ObjectDetection":
            result = self.models[routing](image)
            result = result[result.scores > self.model_pool[routing].threshold]
            if object_name != "object":
                object_name = convert_coco(object_name)
                object_name_idx = self.models[routing].classes.index(object_name) if object_name in self.models[routing].classes else -1
                result = result[result.labels == object_name_idx]
            coordinates, scores = result.bboxes.tolist(), result.scores.tolist()
        
        print(f"Detected {len(coordinates)} {object_name} in the image")
        return coordinates, scores


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
    def __init__(self, config, debug=False):
        self.model_pool = config
        self.image_processor = transforms.ToPILImage()
        self.initialize()
    
    def initialize(self):
        self.models = [load_tool(model.type, model=model.name, device=model.device) for model in self.model_pool]
        self._count_parameters()
        print("Visual Question Answering Models Loaded")
    
    def _count_parameters(self):
        for idx in range(len(self.models)):
            self.models[idx].setup()
            num_params = sum(p.numel() for p in self.models[idx]._inferencer.model.parameters())
            self.model_pool[idx].cost = num_params // 1e6
            print(f"Model {idx} has {num_params // 1e6}M parameters")
    
    def forward(self, image, question, routing):
        assert routing < len(self.models), f"Routing should be less than {len(self.models)}"
        if isinstance(image, torch.Tensor):
            image = self.image_processor(image)
        return self.models[routing](image, question)
    
class LanguageModel():
    def __init__(self, config, debug=False):      
        self.model_pool = config
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
            model=self.model_pool[routing].name,
            messages=[
                {"role": "system", "content": self.model_pool[routing].prompt},
                {"role": "user", "content": query}
            ],
            temperature=self.model_pool[routing].prompt.temperature,
            seed=42,
        )
        return completion.choices[0].message.content
