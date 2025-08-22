# backend/Dockerfile
FROM python:3.11-slim

# Dependencias del sistema mínimas (Pillow, pdfplumber, compilar si hace falta)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala deps
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copia el código
COPY backend ./backend

# Crea carpeta de datos persistentes
RUN mkdir -p /app/data

# Variables por defecto (se sobreescriben por .env / compose)
ENV VSTORE_BACKEND=chroma \
    APP_VERSION=0.1.0 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Ejecuta FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
