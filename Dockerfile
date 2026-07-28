FROM python:3.11-slim

WORKDIR /app

# System deps for pypdf / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the embedding model at build time so the first request isn't slow
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn app.main:app --host 127.0.0.1 --port 8000 & streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
