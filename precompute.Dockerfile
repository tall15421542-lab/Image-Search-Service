FROM python:3.13.7-slim-bookworm

RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt pip install --no-cache-dir -r /tmp/requirements.txt
COPY precompute_image_embeddings.py constants.py . 

RUN ls
ENTRYPOINT ["python", "precompute_image_embeddings.py"]
