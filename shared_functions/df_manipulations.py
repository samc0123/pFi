import pandas as pd
import numpy as np
import glob
import os 

def read_transactions_to_mainFrame(main_path:str,fType:str) -> pd.DataFrame:
    '''Returns a concatenated dataframe of '''
    li = [] # list of final dataframes 

    path_csv = os.path.join(main_path , f"*{fType}")
    all_files = glob.glob(path_csv)
    for file in all_files:
        temp_df = pd.read_csv(filepath_or_buffer=file,index_col=None, header=0)
        li.append(temp_df)

    transaction_frame = pd.concat(objs=li,axis=0,ignore_index=True)
    return(transaction_frame)