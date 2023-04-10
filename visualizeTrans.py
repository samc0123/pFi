import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 

# Load pandas dataframe into file 
trans_csv_path = r'/Users/samchernov/Desktop/Personal/Financials/Transaction_Export/pFi/Master_transaction_file/grouped_master_trans.csv'
df_trans_visualize = pd.read_csv(filepath_or_buffer=trans_csv_path)

# Plot spending by category
fig_bar = plt.figure()

trans_catSpend = df_trans_visualize.groupby(by='category')['amount'].sum()
trans_catSpend.drop(labels='Transfer Between Accounts',axis=0,inplace=True)
trans_catSpend_dict = trans_catSpend.to_dict()
labels_pie,vals_pie = list(trans_catSpend_dict.keys()),np.abs(list(trans_catSpend_dict.values()))

plt.bar(x=labels_pie,height=vals_pie)
plt.xticks(rotation=45,ha='right')
plt.show()




