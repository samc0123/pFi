import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import sys
from datetime import datetime


def generateBarChart(from_date,to_date,df_trans_visualize:pd.DataFrame,figHeight:int=100,figWidth:int=100) -> plt.figure:
    # Filter data based on date range 
    df_trans_visualize = df_trans_visualize[(pd.to_datetime(arg=f'{df_trans_visualize["year"]}-{df_trans_visualize["month"]}-01')>=from_date) & \
        (pd.to_datetime(arg=f'{df_trans_visualize["year"]}-{df_trans_visualize["month"]}-30')<=to_date)]
    
    # Plot spending by category, bar chart 
    fig_bar = plt.figure()

    trans_catSpend = df_trans_visualize.groupby(by='category')['amount'].sum()
    trans_catSpend.drop(labels='Transfer Between Accounts',axis=0,inplace=True)
    trans_catSpend_dict = trans_catSpend.to_dict()
    labels_pie,vals_pie = list(trans_catSpend_dict.keys()),np.abs(list(trans_catSpend_dict.values()))

    plt.bar(x=labels_pie,height=vals_pie)
    plt.xticks(rotation=45,ha='right',wrap=True)
    
    return fig_bar



if __name__ == '__main__':
    print('visulization engine has no main method')
    sys.exit(0)




