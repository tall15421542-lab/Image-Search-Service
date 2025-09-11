import torch
import time
import constants
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, CLIPModel
from fastapi import FastAPI
from pydantic import BaseModel

# Load pre-trained CLIP model and processor
loading_start = time.monotonic()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

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

@app.post("/search", response_model=SearchResponse)
def search_similar_image(search_request: SearchRequest):
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
        text_embeddings = model.get_text_features(**inputs)

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

    return { "image": image_urls[most_similar_index] }
