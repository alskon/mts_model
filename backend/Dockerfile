FROM python:3.10

WORKDIR /app

COPY requirements.txt app/requirements.txt

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000" ,"--reload"]





