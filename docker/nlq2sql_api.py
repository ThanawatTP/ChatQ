import time, argparse, json, warnings, yaml
from Description_base_linking import SchemaLinking
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import Body, FastAPI

def column_index(use_cols,ls_all_col_w_dtype):
    col_ind = {}
    for i,col in enumerate(ls_all_col_w_dtype):
        if col[0] in use_cols: col_ind[col[0]] = i
    return list(set(col_ind.values()))

def sql_create_table(ls_col, ls_all_col_w_dtype):
    ls_col_ind = column_index(ls_col,ls_all_col_w_dtype)
    sql = "CREATE TABLE pointx_fbs_txn_rpt_dly ("
    used_cols = [ls_all_col_w_dtype[i] for i in ls_col_ind ]
    for c_name, c_dtype in used_cols:
        sql = sql + c_name + ' ' + c_dtype + ','
    sql = sql[:-1] + ')'
    return sql

def gen_promp(q, sql):
    promp = sql + '\n' + "-- Using valid SQLite, answer the following questions for the tables provided above."
    promp = promp + '\n' + '-- ' + q
    promp = promp + '\n' + "SELECT"
    return promp

def gen_sql(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=500)
    resp = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return resp


with open('nlq2sql_parameters.yaml', 'r') as yaml_file:
        params = yaml.load(yaml_file, Loader=yaml.FullLoader)

warnings.filterwarnings("ignore")
schema_link = SchemaLinking()
tokenizer = AutoTokenizer.from_pretrained(params['nsql_model_path'])
model = AutoModelForCausalLM.from_pretrained(params['nsql_model_path'])
verbose = True
app = FastAPI()

@app.post("/nlq")
async def pipeline_process(nlq=Body()):

    threshold = 0.2
    question = nlq['input']['text']

    # schema column : data type
    with open(params['pointx_datatype_path'], 'r') as json_file:
        col_type = json.load(json_file)
        ls_all_col_w_dtype = [[k, col_type[k]] for k in col_type]

    selected_columns = schema_link.get_columns_threshold(question, threshold)
    created_table = sql_create_table(selected_columns, ls_all_col_w_dtype)
    prompt = gen_promp(question, created_table)

    resp =  gen_sql(prompt)
    sql_result = resp.split('\n')[-1]
    selected_columns = schema_link.columns_from_query(sql_result)

    for column in selected_columns:
        if column not in schema_link.schema_columns:
            selected_columns.remove(column)
    
    col_descriptions = schema_link.description_of_columns(selected_columns)
    reason = ""
    for col, desc in zip(selected_columns, col_descriptions):
        reason += f"Column: {col} is selected because it is similar to the question.\nDescription: {desc}\n"
    
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
        # print("SELECTED COLUMNS:",len(selected_columns),"FROM",len(ls_all_col_w_dtype))

    return output

    


# if __name__ == '__main__':
#     # start = time.time()
#     arg_parser = argparse.ArgumentParser()
#     arg_parser.add_argument('--nlq', required=True, help="Input question")
#     arg_parser.add_argument('--threshold', type=float, default=0.2, help="Minimum sentence similarity threshold score")
#     arg_parser.add_argument('--param_yaml_path', type=str, default='nlq2sql_parameters.yaml', help="Parameter config file")
#     arg_parser.add_argument('--verbose', type=bool, default=True)
#     args = arg_parser.parse_args()

#     warnings.filterwarnings("ignore")
#     with open(args.param_yaml_path, 'r') as yaml_file:
#         params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    
#     schema_link = SchemaLinking()
#     tokenizer = AutoTokenizer.from_pretrained(params['nsql_model_path'])
#     model = AutoModelForCausalLM.from_pretrained(params['nsql_model_path'])
#     verbose = True
#     pipeline_process(args.nlq, args.threshold)
#     # print(f'e2e processing time = {time.time()-start:.2f} seconds')


