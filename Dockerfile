# Use the official Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=True
ENV PORT=8080
ENV CUDA_VISIBLE_DEVICES=""

# Set the working directory
WORKDIR /app

# Install required packages
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy the application
COPY . .

# Verify directory structure
RUN echo "Checking directory structure:" && \
    ls -la && \
    echo "Checking templates:" && \
    ls -la templates/ && \
    echo "Checking classifier:" && \
    ls -la classifier/

# Expose port
EXPOSE 8080

# Run with debug logging
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--log-level", "debug", "app:app"]
