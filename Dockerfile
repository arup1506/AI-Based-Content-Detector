# Multi-stage lightweight Python Dockerfile for AI Content Detector
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Generate dataset and train model if not present
RUN python data/generate_dataset.py && python train.py

EXPOSE 8501
EXPOSE 5000

# Default entrypoint runs Streamlit; can be overridden for Flask (python flask_app.py)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
