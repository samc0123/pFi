import pandas as pd
import numpy as np
import glob
import os 

# Function takes in csv file with manually reviewed 
    # categories and process them into a dataframe


""" for new function, after doing manual review on predeictions
    # Group by category, then calculate total spending of each category
    group_df_trans = df_trans_w_categories.groupby(by=['category','year','month']).sum()
    group_df_trans['amount'] = group_df_trans['amount'].round(2) # round money to 2 places
    group_df_trans.to_csv() """


def main():
    pass

if __name__ == "__main__":
    print('this is a test')
    main()