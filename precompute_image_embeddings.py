import torch
import os
import time
import constants
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

# Load images
image_urls = []
for root, dirs, files in os.walk("images"):
    for name in files:
        image_urls.append(os.path.join(root, name))

images = [
    Image.open(image_url) for image_url in image_urls
]

imagePreprocessor = processor(images=images, return_tensors="pt")
with torch.no_grad():
    image_embeddings = model.get_image_features(**imagePreprocessor)

image_embeddings = image_embeddings / image_embeddings.norm(
    dim=-1, keepdim=True
)

torch.save({constants.IMAGE_EMBEDDINGS_KEY: image_embeddings, constants.IMAGE_URLS_KEY: image_urls}, constants.IMAGE_EMBEDDING_STORE_PATH)
