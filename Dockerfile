FROM python:3.13-slim
ADD . /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
CMD gunicorn app:server --bind :${PORT:-8080}
