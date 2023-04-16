import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import sys
from datetime import datetime


def generateBarChart(df_trans_visualize:pd.DataFrame,figHeight:int=100,figWidth:int=100) -> plt.figure:
    # Filter data based on date range
    
    
    # Plot spending by category, bar chart 
    fig_bar = plt.figure()

    trans_catSpend = df_trans_visualize.groupby(by='category')['amount'].sum()
    try:
        trans_catSpend.drop(labels=['Transfer Between Accounts','Paycheck','Payment'],axis=0,inplace=True)
    except KeyError:
        pass
    trans_catSpend_dict = trans_catSpend.to_dict()
    labels_pie,vals_pie = list(trans_catSpend_dict.keys()),np.abs(list(trans_catSpend_dict.values()))

    plt.bar(x=labels_pie,height=vals_pie)
    plt.xticks(rotation=45,ha='right',wrap=True)
    
    return fig_bar

def generatePie_freeCash(df_trans_visualize:pd.DataFrame,figHeight:int=100,figWidth:int=100) -> plt.figure:
    '''Generate pie chart of free cash flow '''

    fig_pie = plt.figure()

    trans_catSpend = df_trans_visualize.groupby(by='month')['amount'].sum()
    try:
        trans_catSpend.drop(labels=['Transfer Between Accounts','Payment'],axis=0,inplace=True)
    except KeyError:
        return(fig_pie)
    
    freeCashFlow = trans_catSpend['amount']



if __name__ == '__main__':
    print('visulization engine has no main method')
    sys.exit(0)




