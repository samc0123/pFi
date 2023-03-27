import pandas as pd
import numpy as np
import glob
import os 

def read_transactions_to_mainFrame(main_path:str,fType:str) -> pd.DataFrame:
    '''Returns a concatenated dataframe of '''
    # Read all files of given type from given path
    path_csv = os.path.join(main_path , f"*{fType}")
    all_files = glob.glob(path_csv)
    
    # Specify column structure from downloaded banks CSVs
    cols_downloaded_from_bank_statement = {
        'Chase_cc':np.asarray(['Transaction Date','Post Date','Description',\
            'Category','Type','Amount','Memo']),
        'BofA_cc':np.asarray(['Posted Date','Reference Number','Payee','Address'\
            ,'Amount']),
        'BofA_deposits': np.asarray(['Date','Description','Amount','Running Bal.'])
    }
    # Specify columns needed for processing 
    cols_to_keep_from_bank_statemts = {
        'Chase_cc':np.asarray(['Transaction Date','Description','Amount']),
        'BofA_cc':np.asarray(['Posted Date','Payee','Amount']),
        'BofA_deposits':np.asarray(['Date','Description','Amount'])
    }
    # Specify final dataframe column names 
    final_cols = ['datePosted','merchant','amount']


    # Import files into list for concatenation
    li = [] # list of final dataframes 

    for file in all_files:
        # Read current file into temporary dataframe
        temp_df = pd.read_csv(filepath_or_buffer=file,index_col=None, header=0)
        
        # Find statement type
        statement_type = [k for k,v in \
            cols_downloaded_from_bank_statement.items()\
                if np.array_equiv(a1=np.asarray(temp_df.columns.values),a2=v)]
        
        # Keep the appropriate columns and add dataframe to list for concat
        temp_df = temp_df[cols_to_keep_from_bank_statemts[statement_type.pop()]]
        temp_df.columns = final_cols 
        li.append(temp_df)

    transaction_frame = pd.concat(objs=li,axis=0,ignore_index=True)
    transaction_frame['amount'] = transaction_frame['amount'].replace(',','',regex=True)
    return(transaction_frame)