from fastapi import FastAPI, UploadFile, Form
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipForQuestionAnswering, ViltForQuestionAnswering, AutoModelForCausalLM
import asyncio
from io import BytesIO
import logging

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
models_config = {
    # 3.94B params, https://huggingface.co/Salesforce/blip2-flan-t5-xl
    "blip2-flan-t5-xl": {
        "model_class": Blip2ForConditionalGeneration,
        "pretrained_model": "Salesforce/blip2-flan-t5-xl",
        "device": "cuda:2" if torch.cuda.is_available() else "cpu",
        "batch_size": 4,
    },
    # 385M params params, https://huggingface.co/Salesforce/blip-vqa-base
    "blip-vqa-base": {
        "model_class": BlipForQuestionAnswering,
        "pretrained_model": "Salesforce/blip-vqa-base",
        "device": "cuda:2" if torch.cuda.is_available() else "cpu",
        "batch_size": 4,
    },
    # 87.4M params, https://huggingface.co/dandelin/vilt-b32-finetuned-vqa
    "vilt-b32-finetuned-vqa": {
        "model_class": ViltForQuestionAnswering,
        "pretrained_model": "dandelin/vilt-b32-finetuned-vqa",
        "device": "cuda:2" if torch.cuda.is_available() else "cpu",
        "batch_size": 4,
    },
}

# Load models and processors
device_model_map = {}
for model_name, config in models_config.items():
    try:
        processor = AutoProcessor.from_pretrained(config["pretrained_model"])
        model = config["model_class"].from_pretrained(config["pretrained_model"]).to(config["device"])
        device_model_map[model_name] = {
            "processor": processor,
            "model": model,
            "config": config,
            "request_queue": asyncio.Queue(),
        }
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")

# Async queue processor per model
async def batch_processor(model_name):
    model_data = device_model_map[model_name]
    processor = model_data["processor"]
    model = model_data["model"]
    config = model_data["config"]
    request_queue = model_data["request_queue"]

    logger.info(f"Starting batch processor for {model_name}")

    while True:
        try:
            requests = []
            time_limit = 5  # Time limit for batching in seconds
            start_time = asyncio.get_event_loop().time()

            while len(requests) < config["batch_size"]:
                try:
                    remaining_time = time_limit - (asyncio.get_event_loop().time() - start_time)
                    if remaining_time <= 0:
                        break
                    request = await asyncio.wait_for(request_queue.get(), timeout=remaining_time)
                    requests.append(request)
                except asyncio.TimeoutError:
                    break
            
            if requests:
                logger.info(f"Processing {len(requests)} requests for {model_name}")
                images = [req[0]["image"] for req in requests]
                questions = [req[0]["question"] for req in requests]

                try:
                    # Preprocess inputs
                    if model_name.startswith("vilt"):
                        inputs = processor(
                            text=questions, images=images, return_tensors="pt", padding=True, truncation=True
                        ).to(config["device"])
                        outputs = model(**inputs)
                        results = [model.config.id2label[output.argmax(-1).item()] for output in outputs.logits]
                    else:
                        inputs = processor(
                            text=questions, images=images, return_tensors="pt", padding=True, truncation=True
                        ).to(config["device"])
                        outputs = model.generate(**inputs)
                        results = [processor.decode(output, skip_special_tokens=True) for output in outputs]

                    # Send responses back to clients
                    for i, (_, response_future) in enumerate(requests):
                        response_future.set_result(JSONResponse({"answer": results[i]}))
                        logger.info(f"Response sent for request {i} in {model_name}")

                except Exception as e:
                    logger.error(f"Error during batch processing for {model_name}: {e}")
                    for _, response_future in requests:
                        response_future.set_result(
                            JSONResponse(
                                {"error": f"Failed to process batch request for {model_name}."},
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            )
                        )

        except Exception as e:
            logger.error(f"Critical error in batch processor for {model_name}: {e}")

# Start batch processors for each model
for model_name in device_model_map.keys():
    asyncio.create_task(batch_processor(model_name))

@app.post("/vqa/{model_name}/")
async def vqa_endpoint(model_name: str, image: UploadFile, question: str = Form(...)):
    if model_name not in device_model_map:
        logger.error(f"Model {model_name} not found")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found.")

    try:
        content = await image.read()
        image = Image.open(BytesIO(content)).convert("RGB")
        request_data = {"image": image, "question": question}

        logger.info(f"Received request for model {model_name} with question: {question}")

        response_future = asyncio.Future()
        await device_model_map[model_name]["request_queue"].put((request_data, response_future))
        logger.info(f"Request added to queue for model {model_name}")
        return await response_future

    except Exception as e:
        logger.error(f"Error in VQA endpoint for {model_name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction error occurred.")
