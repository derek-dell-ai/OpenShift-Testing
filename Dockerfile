FROM python:3.11-slim

WORKDIR /app

# Install runtime deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy just the server code
COPY app.py .

# Model will be mounted from a PVC at runtime:
# set a default that matches your mount path
ENV MODEL_DIR=/models/tinyllama/tinyllama-dell-fast-final

EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
