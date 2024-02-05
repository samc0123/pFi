import pandas as pd
import numpy as np
import glob
import os 
from tkinter import *
from tkinter import ttk

from shared_functions import df_manipulations,manual_trans_category

## TODO: 
    # create tkinter front-end to allow users to select path for training\
        # and test files from their respective locations
    

def main():
    # Import all csv files into dataframe
    print("test entering main")
    path_master_transFile = os.environ.get('PATH_MASTER)
    path_newBankStatements = os.environ.get('PATH_NEW_TRANS)
    testing_file_path = os.environ.get('PATH_TESTING_FILE')

    transaction_df = df_manipulations.read_transactions_to_mainFrame(main_path=path_newBankStatements,fType='.csv')
    print(transaction_df)

    columns_indep_dep = {
        'training_independent':'merchant',
        'training_dependent':'category',
        'actual_independent':'merchant'
    }
    training_data = pd.read_csv(filepath_or_buffer=path_master_transFile)
    df_trans_w_categories = manual_trans_category.transaction_knn_model(training_dataset=training_data,actual_dataset=transaction_df,columns_indep_dep = columns_indep_dep)

    # Write new predictions to the master_transactions_manReview.csv file 
    df_trans_w_categories.to_csv(path_or_buf=path_master_transFile,mode='a',\
        index=False,header=False)
    

    

if __name__ == "__main__":
    print('this is a test')
    main()
    
