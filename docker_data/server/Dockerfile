FROM ghcr.io/mentoredtestbed/mentored-base:latest
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/logs
# CMD ["python", "server.py"]
CMD ["./entrypoint.sh"]
