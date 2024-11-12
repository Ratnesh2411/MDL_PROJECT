# Use the official Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=True
ENV PORT=8080
ENV CUDA_VISIBLE_DEVICES=""

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn google-cloud-storage

# Set the working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir gunicorn google-cloud-storage

# Expose the port that the app will run on
EXPOSE 8080

# Run the application using JSON syntax
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
