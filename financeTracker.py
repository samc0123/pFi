import pandas as pd
import numpy as np
import glob
import os 

from shared_functions import df_manipulations,manual_trans_category


def main():
    # Import all csv files into dataframe
    print("test entering main")
    path = r'/Users/samchernov/Desktop/Personal/Financials/Transaction_Export/pFi' # use your path

    transaction_df = df_manipulations.read_transactions_to_mainFrame(main_path=path,fType='.csv')
    print(transaction_df)

    manual_trans_category.manual_categorize_transactions(raw_transaction_data=transaction_df)


if __name__ == "__main__":
    print('this is a test')
    main()
    