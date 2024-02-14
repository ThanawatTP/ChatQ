import re, math, torch
from torch import Tensor
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

sen_emb = SentenceTransformer("../models/all-MiniLM-L6-v2")

def encode(text):
    return sen_emb.encode(text).tolist()

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
        
        df_data = { 'Table' : [],
                    'Column' : [],
                    'Description' : [],
                    'Vector' : [],
                    'Class_name' : []}

        self.schema_classes = dict()
        self.schema_datatypes = {}      # { table1: { column1: datatype, ...}}
        self.table_descriptions = {}    # { table1: description, ...}
        self.sql_condition = {'=', '>', '<', '>=', '<=', '<>', '!='}

        # preparing object variable
        for table, table_info in domain['tables'].items():
            self.schema_classes[table] = table_info['class_labels']
            self.schema_datatypes[table] = table_info['datatypes']
            self.table_descriptions[table] = table_info['description']
            for col in table_info['columns']:
                df_data['Table'].append(table)
                df_data['Column'].append(col)
                df_data['Description'].append(table_info['columns'][col]['text'])
                df_data['Vector'].append(table_info['columns'][col]['vector'])
                df_data['Class_name'].append(table_info['columns'][col]['column_classes'])

        self.column_info_df = pd.DataFrame(df_data).explode('Class_name')
        self.schema_columns_lower = set(self.column_info_df['Column'].str.lower().values)
        self.schema_tables_lower = set(self.column_info_df['Table'].str.lower().values)

        
    
    
    def most_relate_topic(self, text:str, table:str, top_n:int=10, base_n:int=1) -> dict:
        """
        Determine the most related topics to the given text based on schema classes.

        Parameters:
        text (str): The sentence for which related topics need to be determined.
        table (str): The table representing the schema classes against which the sentence will be compared.
        top_n (int): The maximum number of selects for all topics to be selected based on relevance score. Defaults to 10.
        base_n (int): The minimum number of each topic must be selected. Defaults to 1.

        Returns:
        dict: A dictionary where keys are topic names and values are the count of how many times each topic should be selected.

        Example:
        SchemaLinking.most_relate_topic("example text", "example_table")
        {'topic1': 3, 'topic2': 2, 'topic3': 1}
        """

        text_vec = encode(text)
        # apply cosin similarity score for each column followed by question
        topic_scores = [float(cos_sim(info['vector'], text_vec)) for info in self.schema_classes[table].values()]
        # select number of topics based on score probability
        probs = (topic_scores / np.sum(topic_scores)) * top_n
        topic_selected = { key: max(base_n, math.ceil(score)) for key, score in zip(self.schema_classes[table].keys(), probs)}
        
        return topic_selected
    

    def filter_schema(self, question:str, specific_tables:list, max_n:int=10) -> dict:
        """
        Filter the schema to obtain only the columns of each specified table to be used for generating SQL based on a question.

        Parameters:
        question (str): The question for which the SQL schema needs to be filtered.
        specific_tables (list): List of specific tables for which columns should be selected.
        max_n (int): The maximum number of columns to select per table.

        Returns:
        dict: A dictionary containing selected columns for each table along with their relevance scores.

        Example:
        SchemaLinking.filter_schema("example question", ["table1", "table2"])
        {'table1': {'column1': 0.845, 'column2': 0.723}, 'table2': {'column3': 0.912, 'column4': 0.654}}
        """

        # filtered ued column
        _df = self.column_info_df[['Table', 'Column', 'Vector', 'Class_name']]
        _df = _df[_df['Table'].isin(specific_tables)]

        question_vector = encode(question)
        # apply similarity score of each column followed by question
        _df['Score'] = _df['Vector'].apply(lambda x: float(cos_sim(x, question_vector)))

        # check string matching conditions
        columns_match = []
        for word in question.split():
            if word.lower() in self.schema_columns_lower:
                columns_match.append(word)
            if word.lower() in self.schema_tables_lower and word not in specific_tables:
                specific_tables.append(word)

        # if columns match
        if columns_match:
            print("String matching", columns_match)
            _df.loc[_df['Column'].isin(columns_match), 'Score'] = 1.0
        
        # get the number for selecting each topic
        topic_selected = dict()
        for table in specific_tables:
            table_select = (max_n // len(specific_tables))
            topic_selected.update(self.most_relate_topic(question, table, top_n=table_select, base_n=1))
        
        # prepare used schema each table
        used_schema = {table : dict() for table in specific_tables}
        used_cols = []

        # select the top column n number followed by the highest score.
        for topic, num in topic_selected.items():
            selected_col_index = _df[_df['Class_name'] == topic]['Score'].sort_values(ascending=False).head(num).index
            used_cols.extend(_df.loc[selected_col_index, 'Column'].to_list())

        used_cols = list(set(used_cols))

        for i, row in _df[_df['Column'].isin(used_cols)].iterrows():
            used_schema[row['Table']][row['Column']] = round(row['Score'],3)

        # Primary keys and foreign keys are always selected when using more than one table
        if len(specific_tables) > 1:
            for table in specific_tables:
                table_pk = self.schema_datatypes[table]["JOIN_KEY"]["PK"]
                table_fk = self.schema_datatypes[table]["JOIN_KEY"]["FK"]
                for fk, ref_table_column in table_fk.items():
                    if list(ref_table_column.keys())[0] not in specific_tables: del table_fk[fk]
                column_keys = table_pk + list(table_fk.keys())
                for col in column_keys:
                    if col not in used_schema[table].keys():
                        used_schema[table][col] = 0.5

        return used_schema

    def table_col_of_sql(self, sql_query:str) -> dict:
        """
        Extract tables and their corresponding columns from the given SQL query.

        Parameters:
        sql_query (str): The SQL query from which tables and columns need to be extracted.

        Returns:
        dict: A dictionary containing tables as keys and lists of columns as values.

        Example:
        SchemaLinking.table_col_of_sql("SELECT column1, column2 FROM table1 WHERE column3 = 'value'")
        {'table1': ['column1', 'column2', 'column3']}
        """
        
        selected_schema = {}
        query_split = re.split(self.split_pattern, sql_query)
        for table in self.schema_datatypes.keys():
            if table in query_split:
                selected_col = []
                for col in self.schema_datatypes[table]['COLUMNS'].keys():
                    if col in query_split: selected_col.append(col)
                selected_schema[table] = selected_col

        return selected_schema

    def masking_query(self, sql_query:str, condition_value_mask:bool=True) -> str:
        """
        Mask specified columns and optionally condition values in the given SQL query.

        Parameters:
        sql_query (str): The SQL query to be masked.
        condition_value_mask (bool): Whether to mask condition values. Defaults to True.

        Returns:
        str: The masked SQL query.

        Example:
        SchemaLinking.masking_query("SELECT column1, column2 FROM table1 WHERE column3 = 'value'")
        SELECT [MASK], [MASK] FROM [MASK] WHERE [MASK] = [MASK]
        """

        if '*' in sql_query: sql_query = sql_query.replace('*', "[MASK]")
        query_split = re.split(r'(?<=[() .,;])|(?=[() .,;])', sql_query)
        mask_next = False

        for i in range(len(query_split)):
            token = query_split[i].lower()
            # prepare mask condition value
            if token.lower() == 'where': mask_next = True
            if condition_value_mask and mask_next and (token in self.sql_condition and i + 1 < len(query_split)):
                step_mask_next = 1
                # find the condition value
                while query_split[i + step_mask_next] == ' ': step_mask_next += 1
                query_split[i + step_mask_next] = "[MASK]"
            
            if token in self.schema_columns_lower or token in self.schema_tables_lower:
                query_split[i] = "[MASK]"

        return "".join(query_split)
    