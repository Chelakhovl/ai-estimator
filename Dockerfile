FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8001

EXPOSE 8001

CMD ["sh", "-c", "uvicorn app.main:app --host $APP_HOST --port ${PORT:-$APP_PORT} --workers 2"]
