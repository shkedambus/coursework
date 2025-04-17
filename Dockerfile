# Используем готовый образ с Python 3.11, TensorFlow 2.18.0 и поддержкой CUDA/cuDNN
FROM tensorflow/tensorflow:2.18.0-gpu

RUN \
    apt-get update -y \
    && apt-get install -y \
    curl \
    git \
    libzbar0 \
    mupdf \
    libzbar-dev \
    python3 \
    python3-dev \
    python3-pip \
    && pip3 install pip --upgrade 

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir \
               --default-timeout=100 \
               --retries=5 \
               torch --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

CMD ["python", "bot.py"]
