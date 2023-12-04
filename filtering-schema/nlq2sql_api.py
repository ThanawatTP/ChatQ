import json, warnings, yaml, os
from Description_base_linking import SchemaLinking
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class ModelInput(BaseModel):
    input: TextInput
    class Config:
        schema_extra = {
            'example' : {
                "input" : { "text" : "How many unique events have occurred?"}
            }
        }


with open('nlq2sql_parameters.yaml', 'r') as yaml_file:
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)

schema_link = SchemaLinking()
print("Loading NSQL model...")
tokenizer = AutoTokenizer.from_pretrained(params['nsql_model_path'])
model = AutoModelForCausalLM.from_pretrained(params['nsql_model_path'])
verbose = True
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("READY!!!!")

def create_prompt(question, used_schema):
    full_sql = ""
    for table, columns in used_schema.items():
        if not len(columns): continue       # pass this table when no column
        primary_keys = schema_link.schema_datatypes[table]["JOIN_KEY"]["PK"]
        foreign_keys = list(schema_link.schema_datatypes[table]["JOIN_KEY"]["FK"].keys())
        join_table_key = primary_keys + foreign_keys
        
        sql = f"CREATE TABLE {table} ("
        for column in columns:
            if column in join_table_key and len(join_table_key): join_table_key.remove(column)
            try:
                sql += f' {column} {schema_link.schema_datatypes[table][column]},'
            except KeyError: 
                print(f"KeyError :{column}")
                
        if len(join_table_key): # key for join of table are not selected
            for column in join_table_key:
                sql += f' {column} {schema_link.schema_datatypes[table][column]},'

        # All table contain PK (maybe)
        if len(primary_keys):
            sql += 'PRIMARY KEY ('
            for pk in primary_keys: sql += f'"{pk}" ,'
            sql = sql[:-1] + ")"
        if len(foreign_keys):
            for fk, ref_table in schema_link.schema_datatypes[table]["JOIN_KEY"]["FK"].items():
                sql += f', FOREIGN KEY ("{fk}") REFERENCES "{ref_table}" ("{fk}"),'

        sql = sql[:-1] + " )\n\n"
        full_sql += sql
    prompt = full_sql + "-- Using valid SQLite, answer the following questions for the tables provided above."
    prompt = prompt + '\n' + '-- ' + question
    prompt = prompt + '\n' + "SELECT"

    if verbose: print(prompt)

    return prompt

def gen_sql(prompt):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        generated_ids = model.generate(input_ids, max_length=1000)
        sql = tokenizer.decode(generated_ids[0], skip_special_tokens=True).split('\n')[-1]
    return sql

@app.post("/nlq")
async def pipeline_process(nlq: ModelInput):
    question = nlq.dict()['input']['text']
    used_schema = schema_link.filter_schema(question, column_threshold=0.3, table_threshold=0.2, 
                                            filter_tables=False, max_select_columns=False)
    prompt = create_prompt(question, used_schema)
    sql_result = gen_sql(prompt)
    table_col_sql = schema_link.table_col_of_sql(sql_result)
    reason = ""
    for table, cols in table_col_sql.items():
        for file_name in os.listdir(os.environ.get('schema_description_folder_path')):
            if file_name.startswith(table):
                table_description_file = os.path.join(os.environ.get('schema_description_folder_path'), file_name)
                # condition when file name start with same name 
                with open(table_description_file, 'r') as jsonfile:
                    table_description = json.load(jsonfile)
                    if table_description['table'] == table : break
        else: continue

        table_reason = f"Table - {table}\t: {table_description['description']}\n"
        if len(cols):       # have columns of table
            col_reason = "\n".join([f"\tColumn - {c}\t: {table_description['columns'][c]}" for c in cols])
        else: col_reason = ""
        reason += str(table_reason + col_reason + "\n\n")
    
    output = {
        "object": "list",
        "data": [ {
                "object": "sql",
                "index": 0,
                "text": sql_result,
                "reason": reason
            } ],
        "model": "text-2-sql",
        "usage": { 
            "prompt_tokens": 8,
            "total_tokens": 8
        }
    }

    if verbose:
        print("=== QUESTION ===")
        print(question)
        print()
        print("=== SQL QUERY ===")
        print(sql_result)
        print()
        print("=== REASON ===")
        print(reason)
        print()

    return output
    
# @app.delete("/delete_table")
# async def remove_table(table_name: str):
#     if table_name in schema_link.table_desc_vectors:
#         schema_link.remove_table(table_name)
#     else:
#         return "Table not found"
