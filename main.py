import torch
import os
import time
from PIL import Image
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor
from fastapi import FastAPI
from pydantic import BaseModel

# Load pre-trained CLIP model and processor
loading_start = time.monotonic()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

loading_end = time.monotonic()
print(f"Loading Model Time: {(loading_end - loading_start): .2f} seconds")

# Load images
loading_image_start = time.monotonic()

image_urls = []
for root, dirs, files in os.walk("images"):
    for name in files:
        image_urls.append(os.path.join(root, name))

images = [
    Image.open(image_url) for image_url in image_urls
]

loading_image_end = time.monotonic()
print(f"Loading Image Time: {(loading_image_end - loading_image_start): .2f} seconds")

preprocess_image_start = time.monotonic()

imagePreprocessor = processor(images=images, return_tensors="pt")
with torch.no_grad():
    image_embeddings = model.get_image_features(**imagePreprocessor)

image_embeddings = image_embeddings / image_embeddings.norm(
    dim=-1, keepdim=True
)

preprocess_image_end = time.monotonic()
print(f"Preprocess image embeddings Time: {(preprocess_image_end - preprocess_image_start): .2f} seconds")

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
