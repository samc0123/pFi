import pandas as pd
import numpy as np
import glob
import os 
from datetime import datetime as dt 
import csv
import sys

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

 

    # Specify latest dates ran from the statements to avoid duplication
    latest_dates_added_master_file = {
    }
    fPathdates = main_path+'/transDates/lastest_statement_dates.csv'
    with open(file=fPathdates,mode='r') as ld:
        readCSV_latestStatementDates = csv.DictReader(ld)
        for col in readCSV_latestStatementDates:
            latest_dates_added_master_file = col
    
    ld.close()
    print(latest_dates_added_master_file)

    # Import files into list for concatenation
    li = [] # list of final dataframes 

    for file in all_files:
        # Read current file into temporary dataframe
        temp_df = pd.read_csv(filepath_or_buffer=file,index_col=None, header=0)
        
        # Find statement type by checking headers of current statemnt and comparing with dict
        statement_type = [k for k,v in \
            cols_downloaded_from_bank_statement.items()\
                if np.array_equiv(a1=np.asarray(temp_df.columns.values),a2=v)].pop()
        
        # Keep the appropriate columns and add dataframe to list for concat
            # by popping the statement type and then using this val in the dict
            # to return the appropriate columns to keep from temp_df
        temp_df = temp_df[cols_to_keep_from_bank_statemts[statement_type]]
        # Reassign column names
        temp_df.columns = final_cols 

        # Find the earliest and latest dates, alert of potential duplicate statement
        earliest_date = pd.to_datetime(temp_df['datePosted'].min())
        latest_date = pd.to_datetime(temp_df['datePosted'].max())
        latest_date_for_curStatement = pd.to_datetime(latest_dates_added_master_file[statement_type])
        if earliest_date < latest_date_for_curStatement:
            print('Warning: Potentially duplicate statement. Please review\
                master transaction file manually.')
            input('Press Enter to continue...')
        # Modify the current statements latest date as the current latest date
        if latest_date_for_curStatement < latest_date:
            latest_dates_added_master_file[statement_type] = latest_date.strftime(format="%Y-%m-%d")

        # Add current df to be concated for final list of transactions
        li.append(temp_df)

    # Write latest dates to tracking file
    namesFields = list(latest_dates_added_master_file.keys())
    with open(file=fPathdates,mode='w') as editDates:
        datesCSVwriter = csv.DictWriter(f=editDates,fieldnames=namesFields)
        datesCSVwriter.writeheader()
        datesCSVwriter.writerow(latest_dates_added_master_file)
    
    # Concat & format frame to return
    transaction_frame = pd.concat(objs=li,axis=0,ignore_index=True)
    # Replace commas in thousands #'s
    transaction_frame['amount'] = transaction_frame['amount'].replace(',','',regex=True)
    return(transaction_frame)