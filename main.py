import requests
import torch
import os
from PIL import Image
from torch.nn.functional import cosine_similarity
from transformers import CLIPModel, CLIPProcessor


# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load images
image_urls = []
for root, dirs, files in os.walk("images"):
    for name in files:
        image_urls.append(os.path.join(root, name))


images = [
    Image.open(image_url) for image_url in image_urls
]

# Prepare search term
text = "cat"

# Preprocess inputs
inputs = processor(text=text, images=images, return_tensors="pt", padding=True)

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Extract and normalize image and text embeddings
image_embeddings = outputs.image_embeds / outputs.image_embeds.norm(
    dim=-1, keepdim=True
)
text_embeddings = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)

# Compute cosine similarity
cosine_scores = cosine_similarity(image_embeddings, text_embeddings)

# Output similarity scores
for i, image_url in enumerate(image_urls):
    print(
        f"Cosine similarity between the '{text}' and '{image_url}': {cosine_scores[i]}"
    )

# Output the most similar image
most_similar_index = torch.argmax(cosine_scores).item()
print(f"The most similar image is: {image_urls[most_similar_index]}")

