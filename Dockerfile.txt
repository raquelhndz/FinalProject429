# Use an official Python runtime as a parent image
FROM python:3.11-slim

RUN groupadd -r app && useradd -r -g app app

# Copy local code to the container image
ENV APP_HOME /app
ENV PYTHONUNBUFFERED True
WORKDIR $APP_HOME

# Create a writable cache directory
RUN mkdir -p /app/cache && chown -R app:app /app/cache
ENV HF_HOME /app/cache
RUN chmod 777 /app/cache

# Install Python dependencies and Gunicorn
WORKDIR /app
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir gunicorn

USER app
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')"
USER root

RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*

# Copy the rest of the codebase into the image
COPY --chown=app:app . ./
USER app
EXPOSE 8080
ENV PORT 8080

# Run the web service on container startup
CMD exec gunicorn --bind :$PORT --log-level info --workers 1 --threads 8 --timeout 0 dashinterfacework:server