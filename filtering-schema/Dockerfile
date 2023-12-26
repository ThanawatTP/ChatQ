FROM python:3.10.9-slim

# Set environment variables
ENV schema_description_folder_path='src/schemas/column-descriptions'
ENV schema_data_types_folder_path='src/schemas/column-datatypes'
ENV column_threshold=0.2
ENV table_threshold=0.2
ENV max_select_column=10
ENV filter_table=False
ENV verbose=True

WORKDIR /text2sql

# RUN pip install --upgrade pip

COPY requirements.txt /text2sql

RUN pip install -r requirements.txt

COPY . /text2sql

# EXPOSE 8000

CMD ["uvicorn", "nlq2sql_api:app","--host", "0.0.0.0", "--port", "8000"]