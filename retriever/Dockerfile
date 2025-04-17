FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --default-timeout=100 \
    --retries=5 \
    torch==2.6.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# uvicorn - чтобы поднять FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
