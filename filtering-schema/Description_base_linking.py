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
        self.schema_descriptions_path = sorted(os.listdir(os.environ.get('schema_description_folder_path')))
        self.schema_datatypes_path = sorted(os.listdir(os.environ.get('schema_data_types_folder_path')))
        assert len(self.schema_descriptions_path) == len(self.schema_datatypes_path)
        # init class object
        for description_path, datatype_path in zip(self.schema_descriptions_path, self.schema_datatypes_path):
            table1 = "_".join(description_path.split("_")[:-1])
            table2 = "_".join(datatype_path.split("_")[:-1])
            assert table1 == table2, "Not same table"
            self.join_schema(schema_description_path=os.path.join(os.environ.get('schema_description_folder_path'), description_path),
                             schema_datatype_path=os.path.join(os.environ.get('schema_data_types_folder_path'), datatype_path))
        
        if self.verbose:
            print("Description Path\t",self.schema_descriptions_path )
            print("DataType Path\t", self.schema_datatypes_path)

    
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
        

    # # set vector of schema description
    # def set_vector_embedding(self):
    #     self.schema_description_df['Description_vector'] = self.schema_description_df.apply(lambda row: self.sentence_embed(row['Description']), axis=1)

    # # set the format of new schema when apply new table. then concat exist schema
    # # | Table_name  |   Column  |   Description |   Description_vector  |

    # def set_schema(self,table_name:str="pointx_fbs_rpt_dly", schema_description_file_name:str='src/pointx_fbs_rpt_dly_description.csv'):
    #     self.schema_temp = pd.read_csv(schema_description_file_name)
    #     self.schema_temp['Table_name'] = table_name
    #     self.schema_temp.insert(0, 'Table_name', self.schema_temp.pop('Table_name'))
    #     self.schema_temp['Description_vector'] = self.schema_temp.apply(lambda row: self.sentence_embed(row['Description']), axis=1)
    #     assert self.schema_temp.columns.tolist() == self.schema_description_df.columns.tolist(), "Schema columns not match"
    #     self.schema_description_df = pd.concat([self.schema_description_df, self.schema_temp], ignore_index=True)
    #     self.schema_columns = self.schema_description_df['Column'].tolist()

    # # embedding sentence to vector 
    # def sentence_embed(self,sentence:str):
    #     sentence_embeddings = self.model.encode(sentence)
    #     return sentence_embeddings 
    
    # # get similarity scores between question and all column descriptions
    # def question_schema_columns_scores(self,question:str):
    #     q_emb = self.sentence_embed(question)
    #     scores = np.array([float(util.cos_sim(q_emb, des)) for des in self.schema_description_df['Description_vector'].tolist()])
    #     return scores
    
    # # append category values after the description
    # def append_description_by_category(self,category_columns:list):
    #     data_df = pd.read_csv('src/pointx_fbs_rpt_dly.csv')
    #     for col_name in category_columns:
    #         cats = set(data_df[col_name].values.tolist())
    #         cats = [c for c in cats if isinstance(c, str)]
    #         add_description = ", ".join(cats)
    #         idx = self.schema_description_df['Column'] == col_name, 'Description'
    #         self.schema_description_df.loc[idx] += add_description
    #     self.set_vector_embedding()
    
    # # return the list of description of columns
    # def description_of_columns(self,columns:list):
    #     return self.schema_description_df[self.schema_description_df['Column'].isin(columns)]['Description'].tolist()

    # # return similarity score of question and specific column
    # def ques_col_similarity(self,question:str,column:str):
    #     assert column in self.schema_columns, "Column not in schema"
    #     col_emb = self.schema_description_df[self.schema_description_df['Column'] == column]['Description_vector'].values[0]
    #     q_emb = self.sentence_embed(question)
    #     score = float(util.cos_sim(q_emb, col_emb))
    #     return score
    
    # # return a list of columns covering the similarity score threshold
    # def get_columns_threshold(self,question:str, threshold=0.2):
    #     scores = self.question_schema_columns_scores(question)
    #     columns = self.schema_description_df.iloc[np.where(scores >= threshold)]['Column'].tolist()
    #     return columns
    
    # # fine-tune sentence embedding model
    # def fine_tune_model(self,questions_columns_list:list,epochs:int):
    #     train_examples = []
    #     for question, columns in questions_columns_list:
    #         columns_description = " ".join(self.description_of_columns(columns))
    #         data = [question, columns_description]
    #         train_examples.append(InputExample(texts=data))

    #     train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
    #     train_loss = losses.MultipleNegativesRankingLoss(model=self.model)
    #     self.model.fit(train_objectives=[(train_dataloader, train_loss)],
    #                                   epochs=epochs, show_progress_bar=False) 
        
    #     print("Fine-tune model done!")
