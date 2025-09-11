FROM python:3.13.7-slim-bookworm

RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt pip install --no-cache-dir -r /tmp/requirements.txt
COPY main.py model.py constants.py . 

EXPOSE 8000

ENTRYPOINT ["fastapi", "run", "main.py", "--port", "8000"]
