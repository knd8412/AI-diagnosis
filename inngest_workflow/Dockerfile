# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
COPY inngest_workflow/requirements.txt inngest_workflow/requirements.txt

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install \
        --no-cache-dir \
        --index-url https://pypi.org/simple/ \
        -r requirements.txt \
        -r inngest_workflow/requirements.txt
# Copy application code
COPY . .

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
