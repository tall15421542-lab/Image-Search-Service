FROM python:3.13.7-slim-bookworm

RUN --mount=type=bind,source=preprocess/requirements.txt,target=/tmp/requirements.txt pip install --no-cache-dir -r /tmp/requirements.txt

COPY constants constants/ 
COPY preprocess preprocess

RUN ls
ENTRYPOINT ["python", "-m", "preprocess.main"]
