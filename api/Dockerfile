FROM python:3.10-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY modelos/ ./modelos/

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "api.main:app"]
