FROM python:3.11.3

WORKDIR /app

COPY requirements.txt .

EXPOSE 8000

RUN pip install --no-cache-dir -r requirements.txt 

COPY . .

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]