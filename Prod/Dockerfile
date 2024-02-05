FROM python:3.11.7-slim

# Set environment variables
ENV nsql_model_path='models/nsql-350M'
ENV sentence_emb_model_path='models/all-MiniLM-L6-v2'
ENV verbose=True

WORKDIR /text2sql

# RUN pip install --upgrade pip

COPY requirements.txt /text2sql

RUN pip install -r requirements.txt

COPY . /text2sql

# EXPOSE 8000

CMD ["uvicorn", "nlq2sql_api:app","--host", "0.0.0.0", "--port", "8000"]