FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY grade_submission.py .
COPY check_submission.py .
COPY index.html .
COPY spam_train.csv .
COPY spam_test.csv .

# Create submissions directory
RUN mkdir -p submissions

# Default port (overridable via PORT env var)
ENV PORT=5000
EXPOSE 5000

# Run with gunicorn in production
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT} --workers 2 --timeout 120"]
