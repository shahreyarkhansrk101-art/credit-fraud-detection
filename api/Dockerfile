FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/ api/

# models/ and data/processed/ are provided at runtime via docker-compose
# volumes (populated by the training pipeline container) -- just make
# sure the paths exist so the app doesn't fail to start if they're empty.
RUN mkdir -p models data/processed

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
