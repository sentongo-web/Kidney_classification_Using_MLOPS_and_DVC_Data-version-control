FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by TensorFlow and image libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Install the project package in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE 8080

CMD ["python", "app.py"]
