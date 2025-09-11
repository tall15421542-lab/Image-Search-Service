import torch
import time
import constants
import model
import uuid
from datetime import datetime
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, CLIPModel
from fastapi import FastAPI, Depends, HTTPException 
from pydantic import BaseModel, AfterValidator
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from typing import Annotated

DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL, echo=True, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
model.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Load pre-trained CLIP model and processor
loading_start = time.monotonic()

MODEL_NAME="openai/clip-vit-base-patch32"

ml_model = CLIPModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

loading_end = time.monotonic()
print(f"Loading Model Time: {(loading_end - loading_start): .2f} seconds")

loading_embeddings_start = time.monotonic()

store = torch.load(constants.IMAGE_EMBEDDING_STORE_PATH)
image_embeddings = store[constants.IMAGE_EMBEDDINGS_KEY]
image_urls = store[constants.IMAGE_URLS_KEY]

loading_embeddings_end = time.monotonic()
print(f"Loading embeddings Time: {(loading_embeddings_end - loading_embeddings_start): .2f} seconds")

app = FastAPI()

class SearchRequest(BaseModel):
    text: str

class SearchResponse(BaseModel):
    image: str
    inference_id: uuid.UUID 

@app.post("/search", response_model=SearchResponse)
def search_similar_image(search_request: SearchRequest, db: Session = Depends(get_db)):
    search_start = time.monotonic()
    text = search_request.text 

# Preprocess inputs
    preprocess_start = time.monotonic()

    inputs = tokenizer(text=text, return_tensors="pt", padding=True)

    preprocess_end = time.monotonic()
    print(f"Preprocess Time: {(preprocess_end - preprocess_start): .2f} seconds")

# Generate embeddings
    generate_embedding_start = time.monotonic()

    with torch.no_grad():
        text_embeddings = ml_model.get_text_features(**inputs)

    generate_embedding_end = time.monotonic()
    print(f"Generate Embedding Time: {(generate_embedding_end - generate_embedding_start): .2f} seconds")

# Extract and normalize image and text embeddings
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

# Compute cosine similarity
    compute_consine_scores_start = time.monotonic()

    cosine_scores = cosine_similarity(image_embeddings, text_embeddings)

    compute_consine_scores_end = time.monotonic()
    print(f"Compute consine similarity Time: {(compute_consine_scores_end - compute_consine_scores_start): .2f} seconds")

# Output the most similar image
    most_similar_index = torch.argmax(cosine_scores).item()

    search_end = time.monotonic()
    print(f"Search Time: {(search_end - search_start): .2f} seconds")

    image_url = image_urls[most_similar_index]

    inference = model.Inference(model_name = MODEL_NAME, input_text = text, output_image = image_url)
    db.add(inference)
    db.commit()

    return { "image": image_urls[most_similar_index], "inference_id": inference.id }

def is_feedback_valid(feedback: int):
    if feedback != 1 and feedback != -1:
        raise ValueError("Invalid feedback value: {feedback}".format(feedback=feedback))
    return feedback

class InferenceFeedbackRequest(BaseModel):
    inference_id: uuid.UUID 
    feedback: Annotated[int, AfterValidator(is_feedback_valid)] = None

class InferenceResponse(BaseModel):
    model_name: str
    input_text: str
    output_image: str
    feedback: int
    created_at: datetime

@app.post("/inference/feedback", response_model=InferenceResponse)
def giveFeedback(req: InferenceFeedbackRequest, db: Session = Depends(get_db)):
    inference = db.query(model.Inference).filter(model.Inference.id == req.inference_id).first()
    if not inference:
        raise HTTPException(status_code=404, detail="inference not found")

    inference.feedback = req.feedback
    db.commit()

    return {
        "model_name": inference.model_name,
        "input_text": inference.input_text,
        "output_image": inference.output_image,
        "feedback": inference.feedback,
        "created_at": inference.created_at
    }

@app.get("/inference/{inference_id}", response_model=InferenceResponse)
def getInference(inference_id: uuid.UUID, db: Session = Depends(get_db)):
    inference = db.query(model.Inference).filter(model.Inference.id == inference_id).first()
    if not inference:
        raise HTTPException(status_code=404, detail="inference not found")

    return {
        "model_name": inference.model_name,
        "input_text": inference.input_text,
        "output_image": inference.output_image,
        "feedback": inference.feedback,
        "created_at": inference.created_at
    }
