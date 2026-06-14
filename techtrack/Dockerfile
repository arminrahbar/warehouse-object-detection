# Use a slim Python image for the containerized inference service
FROM python:3.12-slim-bookworm

# Install system dependencies needed for FFmpeg and OpenCV headless runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first so Docker can cache dependency installation
COPY requirements.txt requirements.txt

# Use opencv-python-headless inside Docker and avoid installing opencv-python GUI package
RUN sed '/^opencv-python$/d' requirements.txt > requirements-docker.txt \
    && echo "opencv-python-headless" >> requirements-docker.txt \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

# Copy application code into the container
COPY . /app/

# UDP port used by the inference stream
EXPOSE 23000/udp

# Run the inference service
CMD ["python", "app.py"]