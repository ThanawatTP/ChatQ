import json
import boto3
import os
from Description_base_linking import SchemaLinking
from api import encode, generate_sql

domain_bucket =  os.environ['DOMAIN_BUCKET']
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
result_table = dynamodb.Table(os.environ["RESULT_TABLE_NAME"])


##############################################
# Write Chat Result
##############################################
def write_chat_result(reference_id, status, response_data):
  result_table.put_item(
    Item={
          "referenceId": reference_id,
          "status": status,
          "response": json.dumps(response_data)
    },
  )

##############################################
# Create Prompt for Generate_SQL api call
##############################################
def create_prompt(schema_link, question, used_schema):
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
            sql = sql[:-1] + ")"
        if len(foreign_keys):
            for fk, ref_table in schema_link.schema_datatypes[table]["JOIN_KEY"]["FK"].items():
                sql += f', FOREIGN KEY ("{fk}") REFERENCES "{ref_table}" ("{fk}"),'

        sql = sql[:-1] + " )\n\n"
        full_sql += sql
    prompt = full_sql + "-- Using valid SQLite, answer the following questions for the tables provided above."
    prompt = prompt + '\n' + '-- ' + question
    prompt = prompt + '\n' + "SELECT"

    return prompt

##############################################
# Load Embeded data from S3 for specific domain
##############################################
def load_domain(domain_name):
  response = s3_client.get_object(
    Bucket=domain_bucket, 
    Key=f"{domain_name}/embedded_data.json"
  )

  data = response['Body'].read().decode()
  return json.loads(data)

##############################################
# Get Reason Message related to result SQL
##############################################
def get_reason(schema_link, sql_result):
  table_col_sql = schema_link.table_col_of_sql(sql_result)

  reason = ""
  for table, cols in table_col_sql.items():
      table_reason = f"Table - {table}\t: {schema_link.table_descriptions[table]}\n"
      if len(cols):       # have columns of table
          col_reason = "\n".join([f"\tColumn - {c}\t: {schema_link.schema_descriptions[table][c]}" for c in cols])
      else: col_reason = ""
      reason += str(table_reason + col_reason + "\n\n")

  return reason

##############################################
# Main Lambda Handler
##############################################
def lambda_handler(event, context):
  print('event', event)

  detail = event['detail']
  print(detail)

  reference_id = detail['referenceId']
  domain_name = detail['domain']
  question = detail['question']

  domain = load_domain(domain_name)

  schema_link = SchemaLinking(domain)

  used_schema = schema_link.filter_schema(question,
                                          column_threshold= 0.2, 
                                          table_threshold= 0.2, 
                                          max_select_columns= 10, 
                                          filter_tables=False)
  prompt = create_prompt(schema_link, question, used_schema)
  sql_result = generate_sql(prompt)
  print(sql_result)

  reason = get_reason(schema_link, sql_result)
  print(reason)

  data = {
    'question': question,
    'domain': domain_name,
    'sql': sql_result,
    'reason': reason
  }

  write_chat_result(reference_id, "success", {'data': data})