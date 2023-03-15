import pandas as pd 
import numpy as np 

def manual_categorize_transactions(raw_transaction_data:pd.DataFrame) -> pd.DataFrame:
    ''' Manually categorize [raw_transaction_data] for ML training data '''
    transaction_cateogry_dict = {
        "Entertainment":"AMC",
        "Grocery":"Aldi,Trader Joe's,Wegmans,NC700",
        "Online":"Amazon,eBay,",
        "Wholesale":"Walmart,Target"
    }
    print(transaction_cateogry_dict.get('Grocery'))