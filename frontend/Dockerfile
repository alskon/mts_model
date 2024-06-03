FROM python:3.10

WORKDIR /app

COPY requirements.txt app/requirements.txt

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3001

ENTRYPOINT ["streamlit","run","--server.port","3001"]

CMD ["stream_lit.py"]



