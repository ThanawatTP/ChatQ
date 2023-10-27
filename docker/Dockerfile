FROM python:3.10.9-slim

WORKDIR /text2sql

# RUN pip install --upgrade pip

COPY requirements.txt /text2sql

RUN pip install -r requirements.txt

COPY . /text2sql

# EXPOSE 8000

CMD ["uvicorn", "nlq2sql_api:app","--host", "0.0.0.0", "--port", "8000"]