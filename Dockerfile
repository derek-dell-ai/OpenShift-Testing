FROM python:3.11-slim

WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Copy model into the container
COPY tinyllama-dell-fast-final /app/tinyllama-dell-fast-final

# Expose port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
