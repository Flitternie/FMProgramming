from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
import requests
import matplotlib.pyplot as plt


class ObjectDetection:
    def __init__(self, server_url="http://127.0.0.1:8001", debug=False):
        """
        Initialize the ObjectDetection class to interact with the object detection API.

        Args:
            server_url (str): The base URL of the model-serving FastAPI.
        """
        self.server_url = server_url
        self.model_pool = [
            "owlv2-base-patch16-ensemble",
            "grounding-dino-tiny",
            "grounding-dino-base",
        ]
        self.image_processor = transforms.ToPILImage()
        self.debug = True
        print("Object Detection API Initialized")

    def _parse_detections(self, response):
        """
        Parse bounding box coordinates from the API response.

        Args:
            response (dict): The JSON response from the object detection API.

        Returns:
            List[Tuple[int, int, int, int]]: A list of bounding box coordinates (x1, y1, x2, y2).
        """
        coordinates = []
        detections = response.get("detections", [])
        for detection in detections:
            box = detection["box"]
            coordinates.append((box[0], box[1], box[2], box[3]))
        return coordinates

    def forward(self, image, object_name, routing):
        """
        Perform object detection using the specified model.

        Args:
            image (PIL.Image or torch.Tensor): The input image.
            object_name (str): The name of the object to detect.
            model_name (str): The model to use for object detection.

        Returns:
            List[Tuple[int, int, int, int]]: A list of bounding box coordinates (x1, y1, x2, y2).
        """
        try:
            print(f"Performing Object Detection using {self.model_pool[routing]}")
        except IndexError:
            raise IndexError(f"Invalid routing: {routing}, choose from {self.model_pool}")

        if isinstance(image, torch.Tensor):
            image = self.image_processor(image)

        # Convert PIL Image to bytes
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Send the request to the API
        files = {"image": ("image.jpg", buffer, "image/jpeg")}
        data = {"text": object_name}
        endpoint_url = f"{self.server_url}/object_detection/{self.model_pool[routing]}/"
        print(f"Sending request to {endpoint_url}")

        response = requests.post(endpoint_url, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            coordinates = self._parse_detections(result)
            print(f"Detected {len(coordinates)} {object_name}(s) in the image")

            if self.debug == True:
                detections = result["detections"]
                for detection in detections:
                    label, box, score = detection["label"], detection["box"], detection["score"]
                    cropped_image = image.crop(box)
                    # Plot the cropped image
                    plt.figure(figsize=(4, 4))
                    plt.imshow(cropped_image)
                    plt.title(f"{label} ({score:.2f})")
                    plt.axis("off")
                    plt.show()

            return coordinates
        else:
            raise RuntimeError(f"API request failed with status code {response.status_code}: {response.text}")


class VisualQuestionAnswering:
    def __init__(self, server_url="http://127.0.0.1:8002", debug=False):
        """
        Initialize the VisualQuestionAnswering class to interact with the VQA API.

        Args:
            server_url (str): The base URL of the model-serving FastAPI.
        """
        self.server_url = server_url
        self.model_pool = [
            "vilt-b32-finetuned-vqa",
            "blip-vqa-base",
            "blip2-flan-t5-xl",            
        ]
        self.image_processor = transforms.ToPILImage()
        self.debug = True
        print("Visual Question Answering API Initialized")

    def forward(self, image, question, routing):
        """
        Answer a visual question using the specified model.

        Args:
            image (PIL.Image or torch.Tensor): The input image.
            question (str): The question to answer about the image.
            model_name (str): The model to use for answering the question.

        Returns:
            str: The answer provided by the model.
        """
        try:
            print(f"Performing VQA using {self.model_pool[routing]}")
        except IndexError:
            raise IndexError(f"Invalid routing: {routing}, choose from {self.model_pool}")

        if isinstance(image, torch.Tensor):
            image = self.image_processor(image)
        
        # Convert PIL Image to bytes
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Send the request to the API
        files = {"image": ("image.jpg", buffer, "image/jpeg")}
        data = {"question": question}
        endpoint_url = f"{self.server_url}/vqa/{self.model_pool[routing]}/"
        print(f"Sending request to {endpoint_url}")

        response = requests.post(endpoint_url, files=files, data=data)

        if response.status_code == 200:
            if self.debug:
                # display the image with the question and answer
                plt.imshow(image)
                plt.axis("off")
                plt.title(f"Question: {question}\nAnswer: {response.json().get('answer', 'No answer provided')}")
                plt.show()
            return response.json().get("answer", "No answer provided")
        else:
            raise RuntimeError(f"API request failed with status code {response.status_code}: {response.text}")


# Example usage
if __name__ == "__main__":
    server_url = "http://127.0.0.1:8000"

    # Object Detection Example
    od = ObjectDetection(server_url)
    example_image = Image.open("example.jpg")
    boxes = od.detect_objects(example_image, "cat", "grounding-dino-base")
    print(f"Detected boxes: {boxes}")

    # Visual Question Answering Example
    vqa = VisualQuestionAnswering(server_url)
    answer = vqa.answer_question(example_image, "What is in the image?", "blip2-flan-t5-xl")
    print(f"VQA Answer: {answer}")
