import pandas as pd
import numpy as np
import sqlparse, time
from sentence_transformers import SentenceTransformer, util

class SchemaLinking():
    def __init__(self):
        self.Sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.schema_description = pd.read_excel('src/New_query_Description.xlsx',header=1)[['Column','Description']].dropna().reset_index(drop=True)
        self.description_emb = [self.Sentence_transformer.encode(des) for des in self.schema_description['Description'].tolist()]
        self.schema_columns = self.schema_description['Column'].tolist()
        self.sql_extract_token_type = {
            sqlparse.sql.IdentifierList, sqlparse.sql.Where,
            sqlparse.sql.Having, sqlparse.sql.Comparison, sqlparse.sql.Function,
            sqlparse.sql.Parenthesis, sqlparse.sql.Operation, sqlparse.sql.Case
        }

    def similarity_threshold(self,question,selected_columns):
        q_emb = self.Sentence_transformer.encode(question)
        scores = np.array([float(util.cos_sim(q_emb, des)) for des in self.description_emb])
        col_labels_index = self.schema_description[self.schema_description['Column'].isin(selected_columns)].index.tolist()
        col_labels_score = scores[col_labels_index]
        min_threshold = np.min(col_labels_score)

        print("QUESTION:\t",question)
        print("EXPECT COLUMNS:\t",selected_columns)
        print("MIN THRESHOLD:\t",min_threshold)
        print("MAX THRESHOLD:\t",np.max(col_labels_score))
        print("MAX SCORE COLUMN:\t",self.schema_description.iloc[np.argmax(scores)]['Column'])
        print("DESCIPTION:\t",self.schema_description.iloc[np.argmax(scores)]['Description'])
        print("SCORE:\t",np.max(scores))

        n_columns = len(self.schema_description.iloc[np.where(scores >= min_threshold)])
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

        return columns


if __name__ == "__main__":
    print("Start")

