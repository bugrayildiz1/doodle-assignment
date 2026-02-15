# Base image
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt
RUN pip install --default-timeout=300 --no-cache-dir -i https://pypi.org/simple -r requirements.txt
COPY . .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
