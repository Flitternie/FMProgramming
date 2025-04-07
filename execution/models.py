import re
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from openai import OpenAI

from agentlego.apis import load_tool
from execution.utils import convert_coco, RemoteMLLM

"""
This file contains classes for the object detection, visual question answering (VQA), 
and language model systems used in multimodal agent tasks.
"""

# ---------------------------------------------------------------------------- #
#                            Utility Functions                                 #
# ---------------------------------------------------------------------------- #

def load_model(model):
    """
    Load a model based on type and route to correct interface.

    Args:
        model: Model configuration object with type, name, api_key, base_url, args, and device.

    Returns:
        An instantiated model.
    """
    if "vllm" in model.type:
        return RemoteMLLM(model.name, api_key=model.api_key, base_url=model.base_url, args=model.args)
    else:
        num_gpus = torch.cuda.device_count()
        return load_tool(
            tool_type=model.type,
            model=model.name,
            device=model.device if model.device < num_gpus else model.device % num_gpus
        )

# ---------------------------------------------------------------------------- #
#                          Object Detection Class                              #
# ---------------------------------------------------------------------------- #

class ObjectDetection:
    """
    Manages object detection models and performs bounding box prediction.

    Args:
        config (list): List of model configurations.
        debug (bool): Whether to enable debug visualization.
    
    model_list:
        See https://github.com/open-mmlab/mmdetection/ for details.
    """

    def __init__(self, config, debug=False):
        self.image_processor = transforms.ToPILImage()
        self.model_pool = config
        self.debug = debug
        self.initialize()

    def initialize(self):
        """
        Load and initialize all detection models.
        """
        self.models = [load_model(model) for model in self.model_pool]
        self._count_parameters()
        print("Object Detection Models Loaded")

    def _count_parameters(self):
        """
        Count parameters in each model and update cost metadata.
        """
        for idx in range(len(self.models)):
            self.models[idx].setup()
            num_params = sum(p.numel() for p in self.models[idx]._inferencer.model.parameters())
            self.model_pool[idx].cost = num_params // 1e6
            print(f"Object Detection Model Index {idx}: {self.model_pool[idx].name} has {self.model_pool[idx].cost}M parameters")

    def _parse_coordinates(self, text):
        """
        Parse bounding box coordinates and scores from string output.

        Args:
            text (str): Multiline string containing box data.

        Returns:
            Tuple[List[Tuple[int, int, int, int]], List[int]]: Coordinates and scores.
        """
        coordinates, scores = [], []
        for t in text.split("\n"):
            match = re.match(r"\((\d+), (\d+), (\d+), (\d+)\), score (\d+)", t)
            if match:
                x1, y1, x2, y2, score = map(int, match.groups())
                coordinates.append((x1, y1, x2, y2))
                scores.append(score)
        return coordinates, scores

    def forward(self, image, object_name, routing):
        """
        Perform object detection on a given image.

        Args:
            image (Tensor or PIL.Image): Input image.
            object_name (str): Target object name.
            routing (int): Index of model to use.

        Returns:
            Tuple[List[Tuple[int, int, int, int]], List[float]]: Coordinates and scores.
        """
        assert routing < len(self.models), f"Invalid routing index: {routing}"

        if isinstance(image, torch.Tensor):
            image = self.image_processor(image)

        if self.debug:
            plt.figure(figsize=(4, 4))
            plt.imshow(image)
            plt.title(f"Object: {object_name}")
            plt.axis("off")
            plt.show()

        model_type = self.model_pool[routing].type

        if model_type == "TextToBbox":
            query = "all objects" if object_name == "object" else object_name
            result = self.models[routing](image, query)
            threshold = 0.03 if object_name == "object" else self.model_pool[routing].threshold
            result = result[result.scores > threshold]

        elif model_type == "ObjectDetection":
            result = self.models[routing](image)
            result = result[result.scores > self.model_pool[routing].threshold]
            if object_name != "object":
                object_name = convert_coco(object_name)
                class_idx = self.models[routing].classes.index(object_name) if object_name in self.models[routing].classes else -1
                result = result[result.labels == class_idx]

        coordinates = result.bboxes.tolist() if not isinstance(result.bboxes, list) else result.bboxes
        scores = result.scores.tolist() if not isinstance(result.scores, list) else result.scores
        coordinates = [x for _, x in sorted(zip(scores, coordinates), reverse=True)]

        if self.debug and coordinates:
            plt.figure(figsize=(4, 4))
            plt.imshow(image)
            plt.title(f"Detected {len(coordinates)} {object_name} in the image")
            for i in range(min(5, len(coordinates))):
                x1, y1, x2, y2 = coordinates[i]
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="r", linewidth=1))
                plt.text(x1, y1, f"{scores[i]:.2f}", color="r", fontsize=10)
            plt.axis("off")

        print(f"Detected {len(coordinates)} {object_name} in the image")
        return coordinates, scores

# ---------------------------------------------------------------------------- #
#                      Visual Question Answering Class                         #
# ---------------------------------------------------------------------------- #

class VisualQuestionAnswering:
    """
    Visual Question Answering (VQA) system using multiple model backends.

    Args:
        config (list): List of model configurations.
        debug (bool): Whether to enable debug visualization.

    model_list:
        - 'blip-base_3rdparty_vqa', 361.48M
        - 'blip2-opt2.7b_3rdparty-zeroshot_vqa', 3770.47M
        - 'flamingo_3rdparty-zeroshot_vqa', 8.22G
        - 'llava-7b-v1.5_vqa', 7062.90M
        - 'ofa-base_3rdparty-finetuned_vqa', 182.24M
        - 'ofa-base_3rdparty-zeroshot_vqa', 182.24M
        - 'otter-9b_3rdparty_vqa', 8220.45M
    """

    def __init__(self, config, debug=False):
        self.model_pool = config
        self.debug = debug
        self.image_processor = transforms.ToPILImage()
        self.initialize()

    def initialize(self):
        """
        Load and initialize all VQA models.
        """
        self.models = [load_model(model) for model in self.model_pool]
        self._count_parameters()
        print("Visual Question Answering Models Loaded")

    def _count_parameters(self):
        """
        Count parameters for each VQA model and update cost metadata.
        """
        for idx, model in enumerate(self.models):
            try:
                model.setup()
                num_params = sum(p.numel() for p in model._inferencer.model.parameters())
                self.model_pool[idx].cost = num_params // 1e6
            except Exception as e:
                print(f"Failed to count parameters for model {self.model_pool[idx].name}: {e}")
            print(f"VQA Model Index {idx}: {self.model_pool[idx].name} has {self.model_pool[idx].cost}M parameters")

    def forward(self, image, question, routing):
        """
        Answer a visual question about an image.

        Args:
            image (Tensor or PIL.Image): The input image.
            question (str): The question to ask.
            routing (int): Model index to route to.

        Returns:
            str: Answer to the question.
        """
        assert routing < len(self.models), f"Invalid routing index: {routing}"

        if isinstance(image, torch.Tensor):
            image = self.image_processor(image)

        if self.debug:
            plt.figure(figsize=(4, 4))
            plt.imshow(image)
            plt.title(f"Question: {question}")
            plt.axis("off")
            plt.show()

        # if self.model_pool[routing].cost > 5000:
        #     question += " Answer within 3 words."

        response = self.models[routing](image, question)

        if self.debug:
            plt.figure(figsize=(4, 4))
            plt.imshow(image)
            plt.title(f"Question: {question}\nAnswer: {response}")
            plt.axis("off")
            plt.show()

        return response

# ---------------------------------------------------------------------------- #
#                          Language Model Interface                            #
# ---------------------------------------------------------------------------- #

class LanguageModel:
    """
    OpenAI-compatible Language Model for answering text-only prompts.

    Args:
        config (list): List of LLM configurations.
        debug (bool): Whether to log prompt/response pairs.
    """

    def __init__(self, config, debug=False):
        self.model_pool = config
        self.debug = debug
        self.initialize()

    def initialize(self):
        """
        Initialize OpenAI client with API key.
        """
        self.api_key = open("api.key", "r").read().strip()
        self.openai = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepinfra.com/v1/openai"
        )

    def forward(self, query, routing):
        """
        Query a language model and return the response.

        Args:
            query (str): Prompt string.
            routing (int): Model index to route to.

        Returns:
            str: Language model's response.
        """
        assert routing < len(self.model_pool), f"Invalid routing index: {routing}"

        completion = self.openai.chat.completions.create(
            model=self.model_pool[routing].name,
            messages=[
                {"role": "system", "content": self.model_pool[routing].prompt},
                {"role": "user", "content": query}
            ],
            temperature=self.model_pool[routing].temperature,
            seed=42,
        )

        response = completion.choices[0].message.content

        if self.debug:
            print(f"Query: {query}\nResponse: {response}")

        return response
