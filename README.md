# Image-Search-Service
## Build and Run
From the project root directory, run:
```bash
docker compose up -d
```
The first build and run will take longer to complete. 
Subsequent runs should start within seconds.

<img width="1347" height="772" alt="截圖 2025-09-15 下午3 27 01" src="https://github.com/user-attachments/assets/519ed2cd-8165-4891-92e4-9e7542f98b5e" />

## API
- Documentation: After running the service, open [localhost:8000/docs](http://0.0.0.0:8000/docs)
- [Postman Collection](https://github.com/tall15421542-lab/Image-Search-Service/blob/main/image_search_service.postman_collection.json)
### POST /search
Given text, the endpoint returns the most relevant image and its inference ID.
#### curl example
```bash
curl --location 'http://localhost:8000/search' \
--header 'Content-Type: application/json' \
--data '{
    "text": "apple"
}'
```
#### Example response
```json
{
    "image": "images/COCO_val2014_000000002149.jpg",
    "inference_id": "acab354c-c2d9-402b-880f-a58eca62ec51"
}
```

### POST /inference/feedback
Give feedback for an inference. The valid feedback is -1(bad) or 1(good).
Inference ID comes from the `POST /search` response. 
#### curl example
```bash
curl --location 'http://localhost:8000/inference/feedback' \
--header 'Content-Type: application/json' \
--data '{
    "inference_id": "acab354c-c2d9-402b-880f-a58eca62ec51",
    "feedback": 1
}'
```
#### Example Response
```json
{
    "model_name": "openai/clip-vit-base-patch32",
    "input_text": "apple",
    "output_image": "images/COCO_val2014_000000002149.jpg",
    "feedback": 1,
    "created_at": "2025-09-13T01:25:43.645566"
}
```
### GET /inference/{inference_id}
Retrieve the information of an inference.
#### curl example
```bash
curl --location 'http://localhost:8000/inference/acab354c-c2d9-402b-880f-a58eca62ec51'
```
#### Example response
```json
{
    "model_name": "openai/clip-vit-base-patch32",
    "input_text": "apple",
    "output_image": "images/COCO_val2014_000000002149.jpg",
    "feedback": 1,
    "created_at": "2025-09-13T01:25:43.645566"
}
```
