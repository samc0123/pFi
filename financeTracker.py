import pandas as pd
import numpy as np
import glob
import os 

from shared_functions import df_manipulations,manual_trans_category


def main():
    # Import all csv files into dataframe
    print("test entering main")
    path_train = r'/Users/samchernov/Desktop/Personal/Financials/Transaction_Export/pFi/trans_sheets_train' # use your path
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
    msg_test = manual_trans_category.transaction_knn_model(training_dataset=training_data,actual_dataset=transaction_df,columns_indep_dep = columns_indep_dep)
    msg_test.to_csv(path_or_buf='/Users/samchernov/Desktop/Personal/Financials/Transaction_Export/pFi/knnPredictions.csv')

if __name__ == "__main__":
    print('this is a test')
    main()
    