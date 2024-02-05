import os, json, re
from torch import Tensor
import torch
from api import encode

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

class SchemaLinking():
    def __init__(self, domain):
        self.domain = domain
        self.split_pattern = r'[\s\n;().]'
        self.verbose = False
        self.table_desc_vectors = {}     # { table1: vector , ...}
        self.table_descriptions = {}    # { table1: description, ...}
        self.schema_desc_vectors = {}    # { table1: { column1: vector, ...}}
        self.schema_descriptions = {}    # { table1: { column1: description, ...}}
        self.schema_datatypes = {}       # { table1: { column1: datatype, ...}}

        for table_name, table in domain['tables'].items():
          self.table_desc_vectors[table_name] = table['description']['vector']
          self.table_descriptions[table_name] = table['description']['text']
          self.schema_desc_vectors[table_name] = {}
          self.schema_descriptions[table_name] = {}
          self.schema_datatypes[table_name] = table['datatypes']

          for col_name, col_data in table['columns'].items():
            self.schema_desc_vectors[table_name][col_name] = col_data['vector']
            self.schema_descriptions[table_name][col_name] = col_data['text']

    def filter_schema(self, question:str, 
                      column_threshold:float = 0.4, 
                      table_threshold:float = 0.3, 
                      max_select_columns:int = 5, 
                      filter_tables:bool = False):
        
        question_emb = encode(question)
        used_schemas = {}
        found_table = []            # table found in question
        found_columns = []          # column found in question

        # string matching with table, column and question tokens
        for token in question.split():
            
            if token in self.schema_desc_vectors.keys():
                print("Table string match  ---->", token)
                found_table.append(token)
            for table, column in self.schema_desc_vectors.items():
                if token in column.keys(): 
                    found_columns.append(token)
                    print("Column matching  --->",token)
        
        if filter_tables:       #filter table before
            used_tables = []
            for table_name, table_vector in self.table_desc_vectors.items():
                if cos_sim(table_vector, question_emb) >= table_threshold: 
                    used_tables.append(table_name)
        else: used_tables = list(self.table_desc_vectors.keys())     # filtering schema with all columns

        for table in used_tables:
            if table in found_table: table_offset = 0.1         # offset score for selected column in this table
            else: table_offset = 0
            used_schemas[table] = {}
            for column, column_vector in self.schema_desc_vectors[table].items():
                sim_score = cos_sim(column_vector, question_emb)
                if sim_score >= (column_threshold - table_offset):
                    used_schemas[table][column] = round(float(sim_score),3)
                if column in found_columns:
                    used_schemas[table][column] = 1.0
            if max_select_columns and len(used_schemas[table]) > max_select_columns:
                # Select the top k largest values from the dictionary
                used_schemas[table] = dict(sorted(used_schemas[table].items(), key=lambda item: item[1], reverse=True)[:max_select_columns])

        if self.verbose:
            print("QUESTION\t", question)
            print("COLUMN THRESHOLD\t", column_threshold)
            print("TABLE THRESHOLD\t\t", table_threshold)
            print("MAX SELECTED COLUMN\t", max_select_columns)
            print("FILTER TABLE\t\t", filter_tables)
            # print("USED SCHEMAS\t", used_schemas)
            print("")

        return used_schemas

    def table_col_of_sql(self, sql_query):
        selected_schema = {}       # { table : [columns] }
        query_split = re.split(self.split_pattern, sql_query)
        for table in self.schema_desc_vectors.keys():
            if table in query_split:
                selected_col = []
                for col in self.schema_desc_vectors[table].keys():
                    if col in query_split: selected_col.append(col)
                selected_schema[table] = selected_col
        return selected_schema

