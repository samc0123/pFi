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

    transaction_df = df_manipulations.read_transactions_to_mainFrame(main_path=path_test,fType='.csv')
    print(transaction_df)

    #manual_trans_category.count_most_frequent_words(data=transaction_df,df_col_to_countFreqWords="Payee",countWords=400)

    training_data_knn = pd.read_csv(filepath_or_buffer='/Users/samchernov/Desktop/Personal/Financials/Transaction_Export/pFi/trans_sheets_train/predictedCats_knn.csv')
    msg_test = manual_trans_category.transaction_knn_model(training_dataset=training_data_knn,actual_dataset=transaction_df)

if __name__ == "__main__":
    print('this is a test')
    main()
    