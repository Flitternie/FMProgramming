from fastapi import FastAPI, UploadFile, Form
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import asyncio
from io import BytesIO
import base64
import logging

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model and Processor Initialization
model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Async Queue for Batch Processing
request_queue = asyncio.Queue()
batch_size = 1  # Number of requests to batch together

class RequestData(BaseModel):
    image: UploadFile
    text: str


def encode_image(image):
    """Convert an image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/predict/")
async def predict(image: UploadFile, text: str = Form(...)):
    """
    Endpoint to predict grounded object detection.
    """
    try:
        # Read image and add to the queue
        content = await image.read()
        image = Image.open(BytesIO(content)).convert("RGB")
        request_data = {"image": image, "text": text}

        # Put the request in the queue and get the response
        response_future = asyncio.Future()
        await request_queue.put((request_data, response_future))
        return await response_future

    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction error occurred.")


async def batch_processor():
    """
    Process the queued requests in batches, and handle cases where there are fewer requests than the batch size.
    """
    while True:
        try:
            # Collect up to `batch_size` requests from the queue
            requests = []
            time_limit = 5  # Set time limit for waiting in seconds
            start_time = asyncio.get_event_loop().time()

            while len(requests) < batch_size:
                try:
                    remaining_time = time_limit - (asyncio.get_event_loop().time() - start_time)
                    if remaining_time <= 0:
                        break
                    request = await asyncio.wait_for(request_queue.get(), timeout=remaining_time)
                    requests.append(request)
                except asyncio.TimeoutError:
                    break

            if requests:
                if len(requests) == 1:
                    # Handle single request directly without batching
                    single_request = requests[0]
                    image = single_request[0]["image"]
                    text = single_request[0]["text"]

                    try:
                        inputs = processor(
                            images=image, text=text, return_tensors="pt", padding=True
                        ).to(device)

                        with torch.no_grad():
                            outputs = model(**inputs)

                        results = processor.post_process_grounded_object_detection(
                            outputs,
                            inputs.input_ids,
                            box_threshold=0.3,
                            text_threshold=0.3,
                            target_sizes=[image.size[::-1]]
                        )

                        # Convert results to JSON-serializable format
                        detections = []
                        for label, score, box in zip(results["labels"], results["scores"], results["boxes"]):
                            detections.append({
                                "label": label,
                                "score": score.item(),
                                "box": box.cpu().tolist(),
                            })

                        single_request[1].set_result(JSONResponse({"detections": detections}))

                    except Exception as e:
                        logger.error(f"Error during single request processing: {e}")
                        single_request[1].set_result(JSONResponse(
                            {"error": "Failed to process request."}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                        ))

                else:
                    # Prepare inputs for the batch
                    images = [req[0]["image"] for req in requests]
                    texts = [req[0]["text"] for req in requests]

                    try:
                        inputs = processor(
                            images=images, text=texts, return_tensors="pt", padding=True
                        ).to(device)

                        # Perform inference
                        with torch.no_grad():
                            outputs = model(**inputs)

                        # Post-process the results
                        results = processor.post_process_grounded_object_detection(
                            outputs,
                            inputs.input_ids,
                            box_threshold=0.3,
                            text_threshold=0.3,
                            target_sizes=[image.size[::-1] for image in images]
                        )

                        # Convert results to JSON-serializable format
                        json_results = []
                        for i, result in enumerate(results):
                            detections = []
                            for label, score, box in zip(result["labels"], result["scores"], result["boxes"]):
                                detections.append({
                                    "label": label,
                                    "score": score.item(),
                                    "box": box.cpu().tolist(),
                                })
                            json_results.append({"detections": detections})

                        # Send responses back to the clients
                        for i, (_, response_future) in enumerate(requests):
                            response_future.set_result(JSONResponse(json_results[i]))

                    except Exception as e:
                        logger.error(f"Error during batch processing: {e}")
                        for _, response_future in requests:
                            response_future.set_result(JSONResponse(
                                {"error": "Failed to process batch request."}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                            ))

        except Exception as e:
            logger.error(f"Critical error in batch processor: {e}")
