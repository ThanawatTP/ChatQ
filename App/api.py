import warnings
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

sentence_emb_model_path = os.environ.get('SENTENCE_EMB_MODEL_PATH')
nsql_model_path = os.environ.get('NSQL_MODEL_PATH')

sen_emb = SentenceTransformer(sentence_emb_model_path)
tokenizer = AutoTokenizer.from_pretrained(nsql_model_path)
model = AutoModelForCausalLM.from_pretrained(nsql_model_path)

def encode(text):
    return sen_emb.encode(text).tolist()

def generate_nsql_sql(prompt):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        generated_ids = model.generate(input_ids, max_length=1000)
        sql = tokenizer.decode(generated_ids[0], skip_special_tokens=True).split('\n')[-1]
    return sql

"""
Note:   Using Transformers library for its robust pre-trained models
        for NLP tasks like NL to SQL conversion. Plan to transition to API
        calling in the future to separate concerns and improve scalability.
"""