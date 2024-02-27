import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Modules import SchemaLinking
from api import generate_nsql_sql
from openai import OpenAI
import google.generativeai as genai

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

class TextInput(BaseModel):
    text: str
    domain_name: str

class ModelInput(BaseModel):
    input: TextInput
    class Config:
        schema_extra = {
            'example' : {
                "input" : { 
                    "text" : "Display the user id and revenue of user who has the highest total transactions id in pointx_fbs_rpt_dly table",
                    "domain_name": "pointx"}
            }
        }


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# for docker environment
llm_model_name = "deepseek-coder"
max_n = 10
temperature = 0
verbose = True
use_nsql = True
use_llm = True


llm_stop = ['\n\n']
system_content_fillmask = """You are a helpful assistant for generate SQL query from user-specified questions and schema. 
User has some SQL where the [MASK] columns, condition values and tables are syntaxed and User wants you to respond to output that populates the [MASK] column of the SQL input followed by the question and schema description (name - description - data type).
If you don't know which column to fill in Do not include columns that you have created yourself. And only columns defined from the schema must be used. 
Do not use columns from other tables or schema. must also be used from the same table defined in the input.
If you must enter conditional values Please decide the format or value based on the sample values of that column.
If that column has many too long category value please decide base on column description.
please return only the answer of sql string query result!!! ('SELECT...')
"""

few_shot_prompt_mask = """For example
table :     cat - this table contain cat information 
columns :    id - number for identify cat | number
            name - name of cat | text
            age - age of cat | number
            birth_date - pet birthday in format 'YYYY-MM-DD' | datetime
            gender - gender of cat (male, female) | text

question : Show me number of cat for each gender which born before March 23, 2011.
input : SELECT [MASK], COUNT([MASK]) FROM [MASK] WHERE [MASK] < [MASK] GROUP BY [MASK] ;
output : SELECT gender, COUNT(*) FROM cat WHERE birth_date < '2011-03-23' GROUP BY gender;

"""

def create_nsql_prompt(schema_link:object, question:str, used_schema:dict) -> str:
    """
    Generate a prompt for applying into SQL generation model based on the question and schema.

    Parameters:
    schema_link (object): The instance of the class containing schema information.
    question (str): The question for which the prompt is generated.
    used_schema (dict): A dictionary containing tables as keys and lists of columns as values after filtering the schema.

    Returns:
    str: A prompt for applying into SQL generation model.

    Example:
    prompt = create_prompt(schema_instance, "What are the total sales?", 
                          { 'sales': {'date' : 0.3, 'amount' : 0.61}, 
                            'products': {'name' : 0.23, 'price' : 0.57}})
    print(prompt)

    CREATE TABLE sales ( date DATE, amount INT,PRIMARY KEY ("date") )
    -- Using valid SQLite, answer the following questions for the tables provided above.
    -- What are the total sales?
    SELECT
    """
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
                sql += f' {column} {schema_link.schema_datatypes[table]["COLUMNS"][column]},'
            except KeyError: 
                print(f"KeyError :{column}")
                
        if len(join_table_key): # key for join of table are remaining
            for column in join_table_key:
                sql += f' {column} {schema_link.schema_datatypes[table]["COLUMNS"][column]},'

        # All table contain PK (maybe)
        if len(primary_keys):
            sql += 'PRIMARY KEY ('
            for pk_type in primary_keys: sql += f'"{pk_type}" ,'
            sql = sql[:-1] + "),"

        if len(foreign_keys):
            for fk, ref_table_column in schema_link.schema_datatypes[table]["JOIN_KEY"]["FK"].items():
                sql += f' FOREIGN KEY ("{fk}") REFERENCES "{list(ref_table_column.keys())[0]}" ("{list(ref_table_column.values())[0]}"),'

        sql = sql[:-1] + " )\n\n"
        full_sql += sql

    prompt = full_sql + "-- Using valid SQLite, answer the following questions for the tables provided above."
    prompt = prompt + '\n' + '-- ' + question
    prompt = prompt + '\n' + "SELECT"

    return prompt

async def LLM_gensql(full_prompt:str, system_content:str, llm_model:str) -> str:
        """
        Generate SQL query followed by prompt

        Parameters:
        prompt (str): prompt for generate result
        llm_model (str): model-service name for generate result

        Returns:
        str: The complete SQL query.
        """
        if llm_model in ['gemini-pro']:
            try:
                gemini_prompt = system_content + full_prompt
                genai.configure(api_key=GOOGLE_API_KEY)
                gemini_model = genai.GenerativeModel(llm_model)
                gemini_model.temperature = 0
                response = gemini_model.generate_content(gemini_prompt)
                return response.text
            
            except Exception as e:
                 return f"Google AI Error : {e}"
            
        elif llm_model in ['gpt-3.5-turbo', 'gpt-4-0125-preview']:
            
            API_KEY = OPENAI_API_KEY
            base_url = "https://api.openai.com/v1"
            # return None
        
        elif llm_model in ['deepseek-coder', 'deepseek-chat']:
            API_KEY = DEEPSEEK_API_KEY
            base_url = "https://api.deepseek.com/v1"
        try:
            
            client = OpenAI(api_key=API_KEY, base_url=base_url)
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                        {"role": "system",
                            "content": system_content},
                        {"role": "user", 
                            "content": full_prompt},
                        ],
                stop=['\n'],
                temperature=0
            )
            return response.choices[0].message.content
        
        except Exception as e:
            return f"API Error :{e}"
            
def get_reason(schema_link:object, sql_result:str) -> str:
    """
    Get the reason message related to the selected columns and tables from the schema based on the SQL query.

    Parameters:
    schema_link (object): The instance of the class containing schema information.
    sql_result (str): The SQL query result for which the reason message is generated.

    Returns:
    str: The reason message explaining the selection of columns and tables from the schema.

    Example:
    get_reason(schema_instance, "SELECT column1, column2 FROM table1 WHERE column3 = 'value'")

    Table - table1 : Description of table1
        Column - column1 : Description of column1
        Column - column2 : Description of column2
        Column - column3 : Description of column3
    """

    table_col_sql = schema_link.table_col_of_sql(sql_result)
    reason = ""

    for table, cols in table_col_sql.items():
        _df = schema_link.column_info_df[schema_link.column_info_df['Table'] == table][['Column', 'Description']].drop_duplicates()
        table_reason = f"Table - {table}\t: {schema_link.table_descriptions[table]['text']}\n"
        if len(cols):       # have columns of table
            col_reason = "\n".join([f"\tColumn - {c}\t: {_df.loc[_df['Column'] == c, 'Description'].values[0]}" for c in cols])
        else: col_reason = ""
        reason += str(table_reason + col_reason + "\n\n")

    return reason

def create_llm_prompt(schema_link:object, used_schema:dict, question:str, masked_query:str, 
                      few_shot:str=few_shot_prompt_mask, is_marked:bool=True, is_fewshot:bool=True) -> str:

    full_prompt = ""
    for table_name, column_score in used_schema.items():
        _df = schema_link.column_info_df[schema_link.column_info_df['Table'] == table_name][['Column', 'Description']].drop_duplicates()
        full_prompt += f"\ntable : {table_name} - {schema_link.table_descriptions[table_name]['text']}\ncolumns:"

        for column_name in column_score:
            full_prompt += f"\t{column_name} - {_df[_df['Column'] == column_name]['Description'].values[0]}"
            full_prompt += f" | {schema_link.schema_datatypes[table_name]['COLUMNS'][column_name]}\n"

    full_prompt += f"question : {question}\n"
    if is_marked: full_prompt += f"input : {masked_query}"
    if is_fewshot: full_prompt = few_shot + full_prompt
    
    return full_prompt + "\nquery : "


def load_domain(domain_name):

    with open(f"domain/{domain_name}/embedded_data.json", "r") as f:
        domain = json.load(f)
    return domain

@app.post("/nlq")
async def pipeline(nlq: ModelInput):

    question = nlq.dict()['input']['text']
    domain_name = nlq.dict()['input']['domain_name']
    
    domain = load_domain(domain_name)
    # domain_tables = list(domain['tables'].keys())
    schema_link = SchemaLinking(domain)
    domain_tables = schema_link.table_selected(question)

    used_schema = schema_link.filter_schema(question, domain_tables, max_n=max_n)
    nsql_prompt = create_nsql_prompt(schema_link, question, used_schema)
    sql_result = generate_nsql_sql(nsql_prompt)

    if use_llm:
        masked_query = schema_link.masking_query(sql_result)
        llm_prompt = create_llm_prompt(schema_link, used_schema, question, masked_query)
        print("Waiting LLM response")
        sql_result = await LLM_gensql(llm_prompt, system_content_fillmask, llm_model_name)

    reason = get_reason(schema_link, sql_result)

    if verbose:
        print("========= QUESTION =========")
        print(question)
        print()
        print("========= SQL =========")
        print(sql_result)
        print()
        print("========= REASON =========")
        print(reason)
        print()
        print("========= SELECTED TABLES =========")
        print(domain_tables)
        print()
        print("========= SCHEMA =========")
        print(used_schema)
        print()

    output = {
        "object": "list",
        "data": [ {
                "object": "sql",
                "index": 0,
                "text": sql_result,
                "reason": reason
            } ],
        "model": llm_model_name,
        "usage": { 
            "prompt_tokens": len(llm_prompt),
            "total_tokens": len(llm_prompt) + len(sql_result)
        }
    }

    return output