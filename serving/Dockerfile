FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
    transformers==4.46.0 \
    peft==0.13.0 \
    bitsandbytes==0.44.1 \
    accelerate==1.0.0 \
    sentence-transformers==3.2.0 \
    faiss-gpu==1.7.2 \
    fastapi==0.115.0 \
    uvicorn==0.32.0 \
    pydantic==2.9.0

COPY . .

# Model weights and vectorstore are mounted as volumes at runtime:
#   -v /path/to/hf_cache:/root/.cache/huggingface
#   -v /path/to/checkpoints:/checkpoints
#   -v /path/to/vectorstore:/vectorstore

ENV BASE_MODEL=meta-llama/Llama-3.3-70B-Instruct
ENV ADAPTER_PATH=/checkpoints/medrag_v2/final
ENV VECTORSTORE=/vectorstore

EXPOSE 8000

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
