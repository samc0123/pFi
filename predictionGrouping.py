import pandas as pd
import numpy as np
import glob
import os 

# Function takes in csv file with manually reviewed 
    # categories and process them into a dataframe


def group_trans_month_cat(df_transactions:pd.DataFrame) -> pd.DataFrame:
    '''Group transaction dataframe by month and category'''

    

    # Group by category, then calculate total spending of each category
    group_df_trans = df_transactions.groupby(by=['category','year','month'],as_index=False).sum()
    group_df_trans['amount'] = group_df_trans['amount'].round(2) # round money to 2 places
    group_df_trans.sort_values(by=['year','month'],axis='index',ascending=True,inplace=True)
    
    return group_df_trans


def main():
    # Transaction file location
    path_transFolder = os.environ.get('PATH_TRANS_FOLDER')

    # Group by month and category
    grouped_df = group_trans_month_cat(path_masterTrans=path_transFolder+'master_transactions_manReview.csv')
    grouped_df.to_csv(path_or_buf=path_transFolder+'/grouped_master_trans.csv')
if __name__ == "__main__":
    print('Now categorizing transactions')
    main()
