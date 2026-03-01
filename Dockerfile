FROM python:3.10-slim

# Keeps Python output unbuffered so logs appear immediately in Docker
ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System libraries required by TensorFlow, OpenCV, and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this layer is cached between code changes
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the full project and install the cnnClassifier package
COPY . .
RUN pip install -e .

# Directory for uploaded scan images at runtime
RUN mkdir -p uploads

EXPOSE 7860

CMD ["python", "app.py"]
