# Use the official Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=True
ENV PORT=8000
ENV CUDA_VISIBLE_DEVICES=""

# Set the working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port that the app will run on
EXPOSE 8000

# Run the application using JSON syntax
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
