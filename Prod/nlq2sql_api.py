import json, warnings, os
from pprint import pprint
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class EncodeInput(BaseModel):
    input: str
    class Config:
        schema_extra = {
            'example' : {
                "input" : "text to encode"
            }
        }

class GenerateSqlInput(BaseModel):
    text: str

class GenerateSqlApiInput(BaseModel):
    input: GenerateSqlInput
    class Config:
        schema_extra = {
            'example' : {
                "input" : {
                  "text": "prompt string"
                }
            }
        }

print("Loading NSQL model...")

# sentence_emb_model_path = os.environ.get('sentence_emb_model_path')
# nsql_model_path = os.environ.get('nsql_model_path')

nsql_model_path='models/nsql-350M'
sentence_emb_model_path='models/all-MiniLM-L6-v2'

sentence_emb_model = SentenceTransformer(sentence_emb_model_path)
tokenizer = AutoTokenizer.from_pretrained(nsql_model_path)
llm_model = AutoModelForCausalLM.from_pretrained(nsql_model_path)
verbose = True #bool(os.environ.get('verbose').lower() == 'true')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/encode")
async def encode(encode_input: EncodeInput):
  text = encode_input.input
  embedding = sentence_emb_model.encode(text)
  output = {
        "object": "list",
        "data": [ {
                "object": "embedding",
                "index": 0,
                "embedding": embedding.tolist()
            } ],
        "model": "all-MiniLM-L6-v2",
        "usage": { 
            "prompt_tokens": 0,
            "total_tokens": 0
        }
    }
  return output


@app.post("/generate_sql")
async def encode(generate_input: GenerateSqlApiInput):
  prompt = generate_input.input.text
  print(prompt)
  with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      input_ids = tokenizer(prompt, return_tensors="pt").input_ids
      generated_ids = llm_model.generate(input_ids, max_length=1000)
      sql_result = tokenizer.decode(generated_ids[0], skip_special_tokens=True).split('\n')[-1]

      output = {
        "object": "list",
        "data": [ {
                "object": "sql",
                "index": 0,
                "text": sql_result,
                "reason": "pending"
            } ],
        "model": "text-2-sql",
        "usage": { 
            "prompt_tokens": 8,
            "total_tokens": 8
        }
      }

  if verbose:
    print("=== SQL QUERY ===")
    print(sql_result)
    print()

  return output