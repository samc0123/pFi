import pandas as pd
import numpy as np
import glob
import os 

# Function takes in csv file with manually reviewed 
    # categories and process them into a dataframe


def group_trans_month_cat(path_masterTrans:str) -> pd.DataFrame:
    '''Group transaction dataframe by month and category'''

    # Fetch the dataframe
    df_transactions = pd.read_csv(filepath_or_buffer=path_masterTrans)
    

    # Group by category, then calculate total spending of each category
    group_df_trans = df_transactions.groupby(by=['category','year','month']).sum()
    group_df_trans['amount'] = group_df_trans['amount'].round(2) # round money to 2 places
    group_df_trans.sort_values(by=['year','month'],axis='index',ascending=True,inplace=True)
    
    return group_df_trans


def main():
    # Transaction file location
    path_transFolder = r'/Users/samchernov/Desktop/Personal/Financials/Transaction_Export/pFi/Master_transaction_file/'

    # Group by month and category
    grouped_df = group_trans_month_cat(path_masterTrans=path_transFolder+'master_transactions_manReview.csv')
    grouped_df.to_csv(path_or_buf=path_transFolder+'/grouped_master_trans.csv')
if __name__ == "__main__":
    print('Now categorizing transactions')
    main()