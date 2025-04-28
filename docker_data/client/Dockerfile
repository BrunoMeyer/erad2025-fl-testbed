FROM ghcr.io/mentoredtestbed/mentored-base:latest
WORKDIR /app
COPY requirements.txt .
COPY requirements_client.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_client.txt
COPY . .
RUN mkdir -p /app/logs
CMD ["python", "client.py"]
