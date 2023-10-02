import pandas as pd
import numpy as np
import sqlparse, time
from sentence_transformers import SentenceTransformer, util

class SchemaLinking():
    def __init__(self):
        self.Sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.schema_description = pd.read_excel('src/New_query_Description.xlsx',header=1)[['Column','Description']].dropna().reset_index(drop=True)
        self.description_emb = [self.Sentence_transformer.encode(des) for des in self.schema_description['Description'].tolist()]
    
    def get_col_max_score(self,question,col_labels):
        q_emb = self.Sentence_transformer.encode(question)
        scores = np.array([float(util.cos_sim(q_emb, des)) for des in self.description_emb])
        col_labels_index = self.description_emb[self.description_emb['Column'].isin(col_labels)].index.tolist()
        col_labels_score = scores[col_labels_index]
        min_threshold = np.min(col_labels_score)

        print("QUESTION:\t",question)
        print("EXPECT COLUMNS:\t",col_labels)
        print("MIN THRESHOLD:\t",min_threshold)
        # print("MAX THRESHOLD:\t",np.max(col_labels_score))
        # print("MAX SCORE COLUMN:\t",description_df.iloc[np.argmax(scores)]['Column'])
        # print("DESCIPTION:\t",description_df.iloc[np.argmax(scores)]['Description'])
        # print("SCORE:\t",np.max(scores))

        n_columns = len(self.description_emb.iloc[np.where(scores >= min_threshold)])
        print(f"CHOOSE RELATE COLUMN WITHIN THRESHOLD (FROM {len(scores)} COLUMNS):",n_columns)
        print()
        
        return n_columns
    
    def get_query_column(self,sql_query):
        columns = []
        ignore = ['over','extract','desc','datediff','dayofweek','cnt','dateadd','max','min','sum','count','getdate','timestampdiff','weekday','having','month','year','day','date','avg','team_tds.tds_intern.pointx_fbs_txn_rpt_dly']
        for token in sql_query.tokens:
            if str(token).lower() in ignore:
                continue
            if isinstance(token, sqlparse.sql.Identifier):
                if len(str(token).lower().split('as')) > 1:
                    columns += self.get_query_column(token)
                elif '"' in str(token): #ignore condition value
                    continue
                else:
                    columns.append(str(token))
            elif (isinstance(token, sqlparse.sql.IdentifierList) or isinstance(token, sqlparse.sql.Where) or 
                  isinstance(token, sqlparse.sql.Having) or isinstance(token, sqlparse.sql.Comparison) or 
                  isinstance(token, sqlparse.sql.Function) or isinstance(token, sqlparse.sql.Parenthesis) or 
                  isinstance(token,sqlparse.sql.Operation)):
                columns += self.get_query_column(token)
            if str(token).lower() not in pointx_cols:
                continue
                
        return columns


if __name__ == "__main__":
    print("Start")
    description_df = pd.read_excel('src/New_query_Description.xlsx',header=1)
    description_df = description_df[['Column','Description']].dropna().reset_index(drop=True)
    compare_df = pd.read_csv('src/compare_result.csv')
    pointx_cols = pd.read_csv('src/pointx_fbs_rpt_dly.csv').columns.to_list()

    descriptions = description_df['Description'].tolist()
    questions = compare_df['Question'].to_list()
    sql_queries = compare_df['SQL'].to_list()

