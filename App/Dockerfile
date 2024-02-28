FROM python:3.10.9-slim

# Default environment variables
ENV SENTENCE_EMB_MODEL_PATH="models/all-MiniLM-L6-v2"
ENV NSQL_MODEL_PATH="models/nsql-350M"
ENV use_llm=false
ENV verbose=true
ENV max_n=10


# LLM API key (optional)
ENV OPENAI_API_KEY="sk-"
ENV DEEPSEEK_API_KEY="sk-"
ENV GOOGLE_API_KEY=""
ENV temperature=0
ENV llm_model="gpt-3.5-turbo"


WORKDIR /chatq

# RUN pip install --upgrade pip

COPY requirements.txt /chatq

RUN pip install -r requirements.txt

COPY . /chatq

# EXPOSE 8000

CMD ["uvicorn", "app:app","--host", "0.0.0.0", "--port", "8000"]