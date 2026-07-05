FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Run install for dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    numpy \
    opencv-python-headless \
    pyyaml \
    torch \
    torchvision \
    ultralytics \
    onnx \
    onnxslim

# Copy source code
COPY . .

# Set python path environment variable
ENV PYTHONPATH=/workspace

# Expose port
EXPOSE 8000

# Start FastAPI dashboard
CMD ["uvicorn", "aerolens.dashboard.api:app", "--host", "0.0.0.0", "--port", "8000"]
