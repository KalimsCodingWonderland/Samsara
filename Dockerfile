# Dockerfile
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all your code (including examples/…)
COPY . .

ENV PORT=8000
EXPOSE 8000

# run your wrapper via uvicorn for clearer logs
CMD ["uvicorn", "main:server._app", "--host", "0.0.0.0", "--port", "8000"]
