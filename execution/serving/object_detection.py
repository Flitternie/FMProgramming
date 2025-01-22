from fastapi import FastAPI, UploadFile, Form
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
import torch
from PIL import Image
from transformers import pipeline
import asyncio
from io import BytesIO
import logging

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-object-detection).
'''

# Model configuration
models_config = {
    # 155M params, https://huggingface.co/google/owlv2-base-patch16-ensemble
    "owlv2-base-patch16-ensemble": {
        "model_id": "google/owlv2-base-patch16-ensemble",
        "device": "cuda:1" if torch.cuda.is_available() else "cpu",
        "batch_size": 4,
    },
    # 172M params, https://huggingface.co/IDEA-Research/grounding-dino-tiny
    "grounding-dino-tiny": {
        "model_id": "IDEA-Research/grounding-dino-tiny",
        "device": "cuda:1" if torch.cuda.is_available() else "cpu",
        "batch_size": 4,
    },
    # 233M params, https://huggingface.co/IDEA-Research/grounding-dino-base
    "grounding-dino-base": {
        "model_id": "IDEA-Research/grounding-dino-base",
        "device": "cuda:1" if torch.cuda.is_available() else "cpu",
        "batch_size": 4,
    },
}

# Load pipelines
device_model_map = {}
for model_name, config in models_config.items():
    detection_pipeline = pipeline(
        "zero-shot-object-detection", model=config["model_id"], device=config["device"]
    )
    device_model_map[model_name] = {
        "pipeline": detection_pipeline,
        "config": config,
        "request_queue": asyncio.Queue(),
    }

# Async queue processor per model
async def batch_processor(model_name):
    model_data = device_model_map[model_name]
    detection_pipeline = model_data["pipeline"]
    config = model_data["config"]
    request_queue = model_data["request_queue"]

    logger.info(f"Starting batch processor for {model_name}")

    while True:
        try:
            requests = []
            time_limit = 5  # Set time limit for batching in seconds
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
                # Extract images and texts in batch
                images = [req[0]["image"] for req in requests]
                texts = [req[0]["text"] for req in requests]
                thresholds = [req[0]["threshold"] for req in requests]

                try:
                    # Process the batch in a single pipeline call
                    batch_results = detection_pipeline(
                        images,
                        candidate_labels=texts,
                        threshold=thresholds[0],
                    )

                    # Prepare JSON responses for each request
                    json_results = []
                    for result in batch_results:
                        detections = []
                        for detection in result:
                            box = detection["box"]
                            detections.append(
                                {
                                    "label": detection["label"],
                                    "score": round(detection["score"], 3),
                                    "box": [
                                        round(box["xmin"], 2),
                                        round(box["ymin"], 2),
                                        round(box["xmax"], 2),
                                        round(box["ymax"], 2),
                                    ],
                                }
                            )
                        json_results.append({"detections": detections})

                    # Resolve futures with the corresponding results
                    for i, (_, response_future) in enumerate(requests):
                        response_future.set_result(JSONResponse(json_results[i]))
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

@app.post("/object_detection/{model_name}/")
async def object_detection(model_name: str, image: UploadFile, text: str = Form(...), threshold: float = Form(0.5)):
    if model_name not in device_model_map:
        logger.error(f"Model {model_name} not found")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found.")

    try:
        content = await image.read()
        image = Image.open(BytesIO(content)).convert("RGB")
        request_data = {"image": image, "text": text, "threshold": threshold}

        logger.info(f"Received request for model {model_name} with text: {text}, threshold: {threshold}")

        response_future = asyncio.Future()
        await device_model_map[model_name]["request_queue"].put((request_data, response_future))
        logger.info(f"Request added to queue for model {model_name}")
        return await response_future

    except Exception as e:
        logger.error(f"Error in predict endpoint for {model_name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction error occurred.")
