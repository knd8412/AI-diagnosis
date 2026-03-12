# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY embedder/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY embedding_service.py .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "embedding_service.py"]