# AskMyNotes — container image.
# Note: first build is large (PyTorch + the embedding model + EasyOCR are pulled in).
FROM python:3.11-slim

# System libs needed by PyMuPDF / OpenCV (used by OCR).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The embedding model downloads on first request; set a writable cache dir.
ENV HF_HOME=/app/.cache

# Hosts inject $PORT; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
