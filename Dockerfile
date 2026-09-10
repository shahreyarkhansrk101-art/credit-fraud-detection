FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY src/ src/
COPY README.md .

# Data/model/report folders are created at runtime by the scripts,
# but declare them so they can be bind-mounted from the host.
RUN mkdir -p data/raw data/processed models reports/figures

# Run the full pipeline by default: generate -> clean -> train.
# Override the CMD to run a single stage instead, e.g.:
#   docker run --rm -v $(pwd)/data:/app/data fraud-detection python src/clean_data.py
CMD ["sh", "-c", "python src/generate_data.py && python src/clean_data.py && python src/train_model.py"]
