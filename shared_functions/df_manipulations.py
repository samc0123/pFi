import pandas as pd
import numpy as np
import glob
import os 

def read_transactions_to_mainFrame(main_path:str,fType:str) -> pd.DataFrame:
    '''Returns a concatenated dataframe of '''
    li = [] # list of final dataframes 

    path_csv = os.path.join(main_path , f"*{fType}")
    all_files = glob.glob(path_csv)
    compare_cols = ['Posted Date', 'Reference Number', 'Payee', 'Address', 'Amount']
    final_cols = ['datePosted','merchant','amount']
    for file in all_files:
        temp_df = pd.read_csv(filepath_or_buffer=file,index_col=None, header=0)

        if np.array_equiv(a1=compare_cols,a2=temp_df.columns.values):
            temp_df.drop(['Reference Number','Address'],axis=1,inplace=True)
        else:
            temp_df.drop(['Transaction Date','Category','Type','Memo'],axis=1,inplace=True)

        
        temp_df.columns = final_cols
        li.append(temp_df)

    transaction_frame = pd.concat(objs=li,axis=0,ignore_index=True)
    return(transaction_frame)