import requests
import os
import time
import json

ai_platform_endpoint = os.environ['AI_PLATFORM_ENDPOINT']
nlq2sql_model_prefix = os.environ['NLQ2SQL_MODEL_PREFIX']

#######################################
# Encode API
#######################################
def encode(text):
  response = requests.post(f'{ai_platform_endpoint}/embeddings', 
    json={
      'model': f'{nlq2sql_model_prefix}-embedding',
      'version': "0",
      'input': text
    }
  ).json()

  return response['data'][0]['embedding']

#######################################
# Generate SQL API - Require looping to wait for processing result
#######################################
def generate_sql(prompt):
  response = requests.post(f'{ai_platform_endpoint}/seq2seq', 
    json={
      'model': f'{nlq2sql_model_prefix}-llm',
      'version': "0",
      'input': {
        'text': prompt
      }
    }
  ).json()

  reference_id = response['referenceId']

  result = "processing"
  total_wait_time = 0
  wait_time = 3

  while result == "processing" and total_wait_time < 120:
    print('start loop -', total_wait_time)

    #Start with wait 3 seconds for processing and increasing
    time.sleep(wait_time)   
    total_wait_time += wait_time
    wait_time += 1

    response = requests.get(f'{ai_platform_endpoint}/seq2seq/{reference_id}').json()

    result = response['message'].lower()
    print(f"result: {result}")


  final_response = json.loads(result)
  return final_response['data'][0]['text']