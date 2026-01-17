FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY backend /app

# Hugging Face Spaces typically exposes port 7860 and sets PORT dynamically.
EXPOSE 7860

ENV PINECONE_NAMESPACE=dev
ENV LOG_LEVEL=INFO

# Use the PORT environment variable if provided (e.g. on Hugging Face Spaces),
# otherwise default to 7860. Shell form allows env substitution.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}