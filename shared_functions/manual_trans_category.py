import pandas as pd 
import numpy as np 
from collections import Counter

def manual_categorize_transactions(raw_transaction_data:pd.DataFrame) -> pd.DataFrame:
    ''' Manually categorize [raw_transaction_data] for ML training data '''
    transaction_cateogry_dict = {
        "Entertainment":"AMC",
        "Grocery":"Aldi,Trader Joe's,Wegmans,NC700",
        "Online":"Amazon,eBay,",
        "Wholesale":"Walmart,Target"
    }
    print(transaction_cateogry_dict.get('Grocery'))

def count_most_frequent_words(data:pd.DataFrame,df_col_to_countFreqWords:str):
    '''Count the most frequent words in given column of a dataframe'''
    most_common_words = Counter(" ".join(data[df_col_to_countFreqWords].str.lower()).split()).most_common(200)
    most_common_df = pd.DataFrame(data=most_common_words).reset_index()
    most_common_df = most_common_df.drop(labels=['index'],axis=1)
    most_common_df = most_common_df.rename(columns={0:'word',1:'count'})
    most_common_df.to_csv(path_or_buf='/Users/samchernov/Desktop/Personal/Financials/Transaction_Export/mostCommonTrans200.csv')
    print(most_common_df)
    