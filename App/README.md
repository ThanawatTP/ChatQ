# ChatQ Framework

This folder contains the main components of the ChatQ Text2SQL framework. The framework is designed to convert natural language questions into SQL queries.

## Usage

To build the Docker image for this framework, use the following command:

```bash
docker build -t <YOUR_DOCKER_REPO>/chatq:<TAG_NAME> .
```
Replace `<YOUR_DOCKER_REPO>` and `<TAG_NAME>` with your Docker repository name and tag, respectively. This command will start the ChatQ framework container, exposing port 8000 for accessing the API endpoints.

### Pulling the Docker Image

Alternatively, you can pull the Docker image from the following repositories:

- For ARM architecture:
```bash
  docker pull thanawatthongpia/chatq:nsqlllm-arm
```

- For x86 architecture:
```bash
  docker pull thanawatthongpia/chatq:nsqlllm-x86
```

Ensure that you select the appropriate image based on your system architecture.


## Docker Compose

You can easily set up and run the ChatQ framework using Docker Compose. Below is an example `docker-compose.yml` configuration:

```yaml
version: '3'

services:
  chatq:
    image: your_image_name:tag
    container_name: chatq
    ports:
      - "8000:8000"
    environment:
      OPENAI_API_KEY: "your_openai_api_key"
      GOOGLE_API_KEY: "your_google_api_key"
      DEEPSEEK_API_KEY: "your_deepseek_api_key"
      SENTENCE_EMB_MODEL_PATH: "models/all-MiniLM-L6-v2"
      NSQL_MODEL_PATH: "models/nsql-350M"
      llm_model_name: 'deepseek-coder'
      use_llm: True
      max_n: 10
```


You can now interact with the ChatQ framework by sending requests to the exposed API endpoints.

## Environment Variables

The framework utilizes environment variables for configuration. Below are the default environment variables along with their descriptions:

- `SENTENCE_EMB_MODEL_PATH`: Path to the sentence embedding model. You can download the model from [here](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- `NSQL_MODEL_PATH`: Path to the NSQL model for generating SQL followed by schema and question. You can download the model from [here](https://huggingface.co/NumbersStation/nsql-350M) (for small model size).
- `use_llm`: Whether to use a language model for sentence generation (true/false).
- `verbose`: Enable verbose mode for detailed logging (true/false).
- `max_n`: Maximum number of columns for each table when filtering schema.


Optionally, you can set the following environment variables for additional functionality:

- `OPENAI_API_KEY`: API key for OpenAI language model (optional).
- `DEEPSEEK_API_KEY`: API key for DeepSeek language model (optional).
- `GOOGLE_API_KEY`: API key for Google AI services (optional).
- `temperature`: Temperature parameter for language model sampling.
- `llm_model`: Language model to use (e.g., "gpt-3.5-turbo").

## File Overview

- `app.py`: Main application code responsible for handling user input, processing, and generating SQL queries.
- `api.py`: API code for interacting with the Text2SQL model and exposing endpoints for querying.
- `Module.py`: Module for developing domain-specific logic and preprocessing text data.
- `Dockerfile`: Dockerfile for containerization of the ChatQ framework.

These files work together to provide a seamless experience for converting natural language queries into SQL queries.
