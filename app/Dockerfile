FROM python:3.13.7-slim-bookworm

RUN --mount=type=bind,source=app/requirements.txt,target=/tmp/requirements.txt pip install --no-cache-dir -r /tmp/requirements.txt

COPY constants constants
COPY app/ app/ 

EXPOSE 8000

ENTRYPOINT ["fastapi", "run", "app/main.py", "--port", "8000"]
