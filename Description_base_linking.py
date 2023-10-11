import pandas as pd
import numpy as np
import sqlparse, time
from sentence_transformers import SentenceTransformer, util

class SchemaLinking():
    def __init__(self):
        self.verbose = False
        self.Sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.schema_description = pd.read_excel('src/New_query_Description.xlsx',header=1)[['Column','Description']].dropna().reset_index(drop=True)
        self.description_emb = [self.Sentence_transformer.encode(des) for des in self.schema_description['Description'].tolist()]
        self.schema_columns = self.schema_description['Column'].tolist()
        self.sql_extract_token_type = {
            sqlparse.sql.IdentifierList, sqlparse.sql.Where,
            sqlparse.sql.Having, sqlparse.sql.Comparison, sqlparse.sql.Function,
            sqlparse.sql.Parenthesis, sqlparse.sql.Operation, sqlparse.sql.Case
        }
    
    def question_columns_scores(self,question):
        q_emb = self.Sentence_transformer.encode(question)
        scores = np.array([float(util.cos_sim(q_emb, des)) for des in self.description_emb])
        return scores
    
    def accuracy_of_vecs(self,choose_vector,expect_vector):
        #compare accuracy of two one-hot vectors
        correct_matches = np.sum((choose_vector == 1) & (expect_vector == 1))
        total_elements = np.sum(expect_vector == 1)
        if total_elements == 0:
            accuracy = 0.0  # Avoid division by zero
        else:
            accuracy = correct_matches / total_elements

        return accuracy
    
    def choose_col_vec(self,columns):
        # return one hot vector of columns to choose
        col_vec = np.zeros(len(self.schema_columns))
        for col in columns:
            idx = self.schema_columns.index(col)
            col_vec[idx] = 1
        
        if self.verbose:
            print("CHOOSE COLUMNS:\t",columns)
            # print("CHOOSE COLUMNS VECTOR:\t",col_vec)
            # print('VALIDATE CHOOSE COL BY IDX:\t',[self.schema_columns[int(id)] for id,c  in enumerate(col_vec) if int(c)])
            print()

        return col_vec
    
    
    def q_c_similarity(self,question,column):
        assert column in self.schema_columns, "Column not in schema"
        idx = self.schema_columns.index(column)
        col_emb = self.description_emb[idx]
        q_emb = self.Sentence_transformer.encode(question)
        score = float(util.cos_sim(q_emb, col_emb))
        return score


    def get_columns(self,question, threshold=0.2):
        scores = self.question_columns_scores(question)
        columns = self.schema_description.iloc[np.where(scores >= threshold)]['Column'].tolist()
        return columns

    def similarity_threshold(self,question,selected_columns):
        scores = self.question_columns_scores(question)
        col_labels_index = self.schema_description[self.schema_description['Column'].isin(selected_columns)].index.tolist()
        col_labels_score = scores[col_labels_index]
        min_threshold = np.min(col_labels_score)
        n_columns = len(self.schema_description.iloc[np.where(scores >= min_threshold)])

        if self.verbose:
            print("QUESTION:\t",question)
            print("EXPECT COLUMNS:\t",selected_columns)
            print("MIN THRESHOLD:\t",min_threshold)
            # print("MAX THRESHOLD:\t",np.max(col_labels_score))
            # print("MAX SCORE COLUMN:\t",self.schema_description.iloc[np.argmax(scores)]['Column'])
            # print("DESCIPTION:\t",self.schema_description.iloc[np.argmax(scores)]['Description'])
            # print("SCORE:\t",np.max(scores))
            print(f"CHOOSE RELATE COLUMN WITHIN THRESHOLD (FROM {len(scores)} COLUMNS):",n_columns)
            print()
        
        return n_columns
    
    def columns_from_query(self,sql_query):

        if type(sql_query) == str:
            sql_parse = sqlparse.parse(sql_query)[0]
        else:
            sql_parse = sql_query

        columns = []
        for token in sql_parse.tokens:
            
            if isinstance(token, sqlparse.sql.Identifier):
                if len(str(token).lower().split('as')) > 1:
                    columns.extend(self.columns_from_query(token))
                elif str(token).lower() in self.schema_columns:
                    columns.append(str(token))
                    
            elif isinstance(token, tuple(self.sql_extract_token_type)):
                columns.extend(self.columns_from_query(token))

        return list(set(columns))

