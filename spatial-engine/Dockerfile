FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for spatial libraries
RUN apt-get update && apt-get install -y \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden in docker-compose)
CMD ["python", "-c", "print('Spatial engine ready')"]
