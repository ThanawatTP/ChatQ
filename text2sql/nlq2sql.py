import time, argparse, json, warnings, yaml
from Description_base_linking import SchemaLinking
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    print("=== SQL QUERY ===")
    print(resp.split('\n')[-1])
    print()
    return resp

def pipeline_process(nlq, threshold):

    print("=== QUESTION ===")
    print(nlq)
    print()

    # schema column : data type

    with open(params['pointx_datatype_path'], 'r') as json_file:
        col_type = json.load(json_file)
        ls_all_col_w_dtype = [[k, col_type[k]] for k in col_type]

    # start = time.time()
    selected_columns = schema_link.get_columns_threshold(nlq, threshold)
    # print(f'Filtering column processing time = {time.time()-start:.2f} seconds')
    # start = time.time()
    created_table = sql_create_table(selected_columns, ls_all_col_w_dtype)
    prompt = gen_promp(nlq, created_table)
    # print(f'Gen prompt processing time = {time.time()-start:.2f} seconds')

    start = time.time()
    resp =  gen_sql(prompt)
    stop = time.time()
    duration = stop-start
    print("SELECTED COLUMNS:",len(selected_columns),"FROM",len(ls_all_col_w_dtype))
    print(f'Gen SQL Processing time = {duration:.2f} seconds')


if __name__ == '__main__':
    start = time.time()
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--nlq', type=str, required=True, help="Input question")
    arg_parser.add_argument('--threshold', type=float, default=0.2, help="Minimum sentence similarity threshold score")
    arg_parser.add_argument('--param_yaml_path', type=str, default='/Users/thanawatthongpia/Desktop/code/fine-tuning/docker/nlq2sql_parameters.yaml', help="Parameter config file")
    args = arg_parser.parse_args()

    warnings.filterwarnings("ignore")
    with open(args.param_yaml_path, 'r') as yaml_file:
        params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    
    schema_link = SchemaLinking()
    tokenizer = AutoTokenizer.from_pretrained(params['nsql_model_path'])
    model = AutoModelForCausalLM.from_pretrained(params['nsql_model_path'])
    pipeline_process(args.nlq, args.threshold)
    print(f'e2e processing time = {time.time()-start:.2f} seconds')


