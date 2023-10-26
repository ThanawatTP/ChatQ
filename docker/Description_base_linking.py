import pandas as pd
import numpy as np
import sqlparse, time, yaml
# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader

class SchemaLinking():
    def __init__(self):
        self.verbose = False
        yaml_file_path = 'nlq2sql_parameters.yaml'
        with open(yaml_file_path, 'r') as yaml_file:
            self.params = yaml.load(yaml_file, Loader=yaml.FullLoader)
        # self.tokenizer = AutoTokenizer.from_pretrained('models/all-MiniLM-L6-v2')
        # self.model = AutoModel.from_pretrained('models/all-MiniLM-L6-v2')
        self.model = SentenceTransformer(self.params['sentence_emb_model_path'])
        self.schema_description_df = pd.DataFrame(columns=['Table_name','Column','Description','Description_vector'])
        self.schema_columns = self.schema_description_df['Column'].tolist()
        self.set_schema("pointx_fbs_rpt_dly", self.params['pointx_description_path'])
        self.sql_extract_token_type = {
            sqlparse.sql.IdentifierList, sqlparse.sql.Where,
            sqlparse.sql.Having, sqlparse.sql.Comparison, sqlparse.sql.Function,
            sqlparse.sql.Parenthesis, sqlparse.sql.Operation, sqlparse.sql.Case
        }

    def set_vector_embedding(self):
        self.schema_description_df['Description_vector'] = self.schema_description_df.apply(lambda row: self.sentence_embed(row['Description']), axis=1)

    def set_schema(self,table_name:str="pointx_fbs_rpt_dly", schema_description_file_name:str='src/pointx_fbs_rpt_dly_description.csv'):
        self.schema_temp = pd.read_csv(schema_description_file_name)
        self.schema_temp['Table_name'] = table_name
        self.schema_temp.insert(0, 'Table_name', self.schema_temp.pop('Table_name'))
        self.schema_temp['Description_vector'] = self.schema_temp.apply(lambda row: self.sentence_embed(row['Description']), axis=1)
        assert self.schema_temp.columns.tolist() == self.schema_description_df.columns.tolist(), "Schema columns not match"
        self.schema_description_df = pd.concat([self.schema_description_df, self.schema_temp], ignore_index=True)
        self.schema_columns = self.schema_description_df['Column'].tolist()

    def sentence_embed(self,sentence:str):
        sentence_embeddings = self.model.encode(sentence)
        return sentence_embeddings 
    
    def question_schema_columns_scores(self,question:str):
        q_emb = self.sentence_embed(question)
        scores = np.array([float(util.cos_sim(q_emb, des)) for des in self.schema_description_df['Description_vector'].tolist()])
        return scores
    
    def append_description_by_category(self,category_columns:list):
        data_df = pd.read_csv('src/pointx_fbs_rpt_dly.csv')
        for col_name in category_columns:
            cats = set(data_df[col_name].values.tolist())
            cats = [c for c in cats if isinstance(c, str)]
            add_description = ", ".join(cats)
            idx = self.schema_description_df['Column'] == col_name, 'Description'
            self.schema_description_df.loc[idx] += add_description
        self.set_vector_embedding()
    
    def description_of_columns(self,columns:list):
        return self.schema_description_df[self.schema_description_df['Column'].isin(columns)]['Description'].tolist()


    def ques_col_similarity(self,question:str,column:str):
        assert column in self.schema_columns, "Column not in schema"
        col_emb = self.schema_description_df[self.schema_description_df['Column'] == column]['Description_vector'].values[0]
        q_emb = self.sentence_embed(question)
        score = float(util.cos_sim(q_emb, col_emb))
        return score
    
    def get_columns_threshold(self,question:str, threshold=0.2):
        scores = self.question_schema_columns_scores(question)
        columns = self.schema_description_df.iloc[np.where(scores >= threshold)]['Column'].tolist()
        return columns
    
    def fine_tune_model(self,questions_columns_list:list,epochs:int):
        train_examples = []
        for question, columns in questions_columns_list:
            columns_description = " ".join(self.description_of_columns(columns))
            data = [question, columns_description]
            train_examples.append(InputExample(texts=data))

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
        train_loss = losses.MultipleNegativesRankingLoss(model=self.model)
        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                                      epochs=epochs, show_progress_bar=False) 
        
        print("Fine-tune model done!")

    ##### Expirement testing #####
    
    def choose_col_onehot(self,columns):
        # return one hot vector of columns to choose
        assert self.schema_columns is not None, "Schema not set"
        col_vec = np.zeros(len(self.schema_columns))
        for col in columns:
            idx = self.schema_columns.index(col)
            col_vec[idx] = 1
        return col_vec
    
    def accuracy_of_onehot(self,choose_onehot_vector,expect_onehot_vector):
        #compare accuracy of two one-hot vectors
        correct_matches = np.sum((choose_onehot_vector == 1) & (expect_onehot_vector == 1))
        total_elements = np.sum(expect_onehot_vector == 1)
        if total_elements == 0:
            accuracy = 0.0  # Avoid division by zero
        else:
            accuracy = correct_matches / total_elements

        return accuracy

    def similarity_threshold(self,question,selected_columns):
        scores = self.question_schema_columns_scores(question)
        col_labels_index = self.schema_description_df[self.schema_description_df['Column'].isin(selected_columns)].index.tolist()
        col_labels_score = scores[col_labels_index]
        min_threshold = np.min(col_labels_score)
        n_columns = len(self.get_columns_threshold(question,min_threshold))

        print("QUESTION:\t",question)
        print("EXPECT COLUMNS:\t",selected_columns)
        print("MIN THRESHOLD:\t",min_threshold)
        print(f"CHOOSE RELATE COLUMN WITHIN THRESHOLD (FROM {len(scores)} COLUMNS):",n_columns)
        print()
        return n_columns

    def columns_from_query(self,sql_query):

        if type(sql_query) == str:
            sql_query = sqlparse.parse(sql_query)[0]
        columns = []
        for token in sql_query.tokens:
            
            if isinstance(token, sqlparse.sql.Identifier):
                if len(str(token).lower().split('as')) > 1:
                    columns.extend(self.columns_from_query(token))
                elif str(token).lower() in self.schema_columns:
                    columns.append(str(token))
                    
            elif isinstance(token, tuple(self.sql_extract_token_type)):
                columns.extend(self.columns_from_query(token))

        return columns