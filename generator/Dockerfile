FROM tensorflow/tensorflow:2.18.0-gpu

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir \
               --default-timeout=100 \
               --retries=5 \
               torch --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# uvicorn - чтобы поднять FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
