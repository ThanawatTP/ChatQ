FROM python:3.10.9-slim

WORKDIR /text2sql

# RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "nlq2sql_api:app"]

# CMD python nlq2sql.py --nlq '{ "input" : { "text" : "How many user open the app"} }'