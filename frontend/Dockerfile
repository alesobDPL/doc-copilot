FROM python:3.11-slim

WORKDIR /app

COPY frontend/requirements.txt ./frontend/requirements.txt
RUN pip install --no-cache-dir -r frontend/requirements.txt

# Copia la app de Streamlit
COPY frontend ./frontend

# Streamlit sirve en 8501 por defecto
EXPOSE 8501

# BACKEND_URL lo inyecta docker-compose (o .env)
ENV BACKEND_URL=http://localhost:8000 \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
