# Image Python officielle
FROM python:3.11

# Dossier de travail
WORKDIR /app

# Copier requirements
COPY requirements.txt .

# Installer dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Exposer le port FastAPI
EXPOSE 8000

# Lancer l'API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]