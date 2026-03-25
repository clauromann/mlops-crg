FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV BACKEND_URL=http://localhost:8000

CMD ["python", "gradio_app.py"]
