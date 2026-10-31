FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
USER user
ENV HF_HOME=/home/user/.cache/huggingface PORT=7860
EXPOSE 7860
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
