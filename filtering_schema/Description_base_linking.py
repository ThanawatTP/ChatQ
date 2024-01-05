import os, json, re
from sentence_transformers import SentenceTransformer, util

class SchemaLinking():
    def __init__(self):
        
        self.verbose = bool(os.environ.get('verbose').lower() == 'true')
        self.split_pattern = r'[\s\n;().]'
        self.table_desc_vectors = {}     # { table1: vector , ...}
        self.schema_desc_vectors = {}    # { table1: { column1: vector, ...}}
        self.schema_datatypes = {}       # { table1: { column1: datatype, ...}}
        self.sentence_emb_model = SentenceTransformer(os.environ.get('sentence_emb_model_path'))

    def selected_domain(self, schema_description_folder_path, schema_data_types_folder_path):
        schema_descriptions_files_path = sorted(os.listdir(schema_description_folder_path))
        schema_datatypes_files_path = sorted(os.listdir(schema_data_types_folder_path))
        assert len(schema_descriptions_files_path) == len(schema_datatypes_files_path), "Unequal number of files"

        # init class object
        for description_path, datatype_path in zip(schema_descriptions_files_path, schema_datatypes_files_path):
            table1 = "_".join(description_path.split("_")[:-1])
            table2 = "_".join(datatype_path.split("_")[:-1])
            assert table1 == table2, f"Datatype and Description files not match with {table1} and {table2}"
            self.join_schema(schema_description_path=os.path.join(schema_description_folder_path, description_path),
                             schema_datatype_path=os.path.join(schema_data_types_folder_path, datatype_path))
        
        if self.verbose:
            print("Description files\t",schema_descriptions_files_path )
            print("DataType files\t", schema_datatypes_files_path)

    
    def join_schema(self, schema_description_path:str, schema_datatype_path:str):
        with open(schema_description_path) as jsonfile:
            new_schema_description = json.load(jsonfile)
        with open(schema_datatype_path) as jsonfile:
            new_schema_datatype = json.load(jsonfile)
        
        table_name = new_schema_description['table']
        table_vector = self.sentence_emb_model.encode(new_schema_description['description'])
        self.table_desc_vectors[table_name] = table_vector

        self.schema_datatypes[table_name] = new_schema_datatype
        column_vectors = {}
        for col, desc in new_schema_description["columns"].items():
            column_vectors[col] = self.sentence_emb_model.encode(desc)
        self.schema_desc_vectors[table_name] = column_vectors
    
    def remove_table(self, table_name):
        del self.table_desc_vectors[table_name]
        del self.schema_desc_vectors[table_name]
        del self.schema_datatypes[table_name]
        return True
    
    def filter_schema(self, question:str, 
                      column_threshold:float = 0.4, 
                      table_threshold:float = 0.3, 
                      max_select_columns:int = 5, 
                      filter_tables:bool = False):
        question_emb = self.sentence_emb_model.encode(question)
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
                if util.cos_sim(table_vector, question_emb) >= table_threshold: 
                    used_tables.append(table_name)
        else: used_tables = list(self.table_desc_vectors.keys())     # filtering schema with all columns

        for table in used_tables:
            if table in found_table: table_offset = 0.1         # offset score for selected column in this table
            else: table_offset = 0
            used_schemas[table] = {}
            for column, column_vector in self.schema_desc_vectors[table].items():
                sim_score = util.cos_sim(column_vector, question_emb)
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
        


