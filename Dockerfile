FROM --platform=linux/amd64 python:3.11-slim

# 3-a) system-level deps (ulid uses no compiled code, so this is light)
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# 3-b) python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3-c) project code
COPY . .

# 3-d) default port for local dev; Azure overrides with env PORT
ENV PORT=8080
EXPOSE 8080

CMD ["python", "-m", "examples.search_agent.src.samsara.samsara"]
