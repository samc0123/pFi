import pandas as pd
import numpy as np
import glob
import os 

from shared_functions import df_manipulations,manual_trans_category


def main():
    # Import all csv files into dataframe
    print("test entering main")
    path_test = r'/Users/samchernov/Desktop/Personal/Financials/Transaction_Export/pFi/trans_sheets_test' # use your path
    path_training_file = r'/Users/samchernov/Desktop/Personal/Financials/Transaction_Export/pFi/trans_sheets_train/manual_training_trans_cats_ident/predictedCats_knn.csv'

    transaction_df = df_manipulations.read_transactions_to_mainFrame(main_path=path_test,fType='.csv')
    print(transaction_df)

    columns_indep_dep = {
        'training_independent':'nameTrans',
        'training_dependent':'category',
        'actual_independent':'merchant'
    }
    training_data = pd.read_csv(filepath_or_buffer=path_training_file)
    df_trans_w_categories = manual_trans_category.transaction_knn_model(training_dataset=training_data,actual_dataset=transaction_df,columns_indep_dep = columns_indep_dep)
    df_trans_w_categories.to_csv(path_or_buf='/Users/samchernov/Desktop/Personal/Financials/Transaction_Export/pFi/knnPredictions.csv')

    """ for new function, after doing manual review on predeictions
    # Group by category, then calculate total spending of each category
    group_df_trans = df_trans_w_categories.groupby(by=['category','year','month']).sum()
    group_df_trans['amount'] = group_df_trans['amount'].round(2) # round money to 2 places
    group_df_trans.to_csv() """

if __name__ == "__main__":
    print('this is a test')
    main()
    