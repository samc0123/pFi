import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import sys
from datetime import datetime


def generateBarChart(df_trans_visualize:pd.DataFrame,figHeight:int=7,figWidth:int=10) -> plt.figure:
    # Filter data based on date range
    
    
    # Plot spending by category, bar chart 
    fig_bar,(ax_bar,ax_table,ax_table_2)= plt.subplots(nrows=3,gridspec_kw=dict(height_ratios=[2,1,1]))
    fig_bar.set_figheight(figHeight)
    fig_bar.set_figwidth(figWidth)
    
    trans_catSpend = df_trans_visualize.groupby(by='category')['amount'].sum()
    trans_catSpend_dict = trans_catSpend.to_dict()
    labels_bar,vals_bar = list(trans_catSpend_dict.keys()),np.round(np.abs(list(trans_catSpend_dict.values())),decimals=2)

    bars = ax_bar.bar(x=labels_bar,height=vals_bar)
    ax_bar.set_xticklabels(labels_bar,rotation=45,ha='right',wrap=True)
    ax_bar.bar_label(bars,labels = vals_bar,rotation=90)

    
    
    ax_table.axis(False)
    the_table = ax_table.table(cellText=[vals_bar[0:int(len(vals_bar)/2)]],colLabels=labels_bar[0:int(len(labels_bar)/2)],
          loc='bottom')
    #TODO: split table in half 
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(6)
    the_table.auto_set_column_width(col=list(range(int(len(labels_bar)/2))))
    
    ax_table_2.axis(False)
    the_table_2 = ax_table_2.table(cellText=[vals_bar[int(len(vals_bar)/2)+1:len(vals_bar)-1]],colLabels=labels_bar[int(len(labels_bar)/2)+1:len(labels_bar)-1],
          loc='bottom')
    #TODO: split table in half 
    the_table_2.auto_set_font_size(False)
    the_table_2.set_fontsize(8)
    the_table_2.auto_set_column_width(col=list(range(int(len(labels_bar)/2))))        

    fig_bar.subplots_adjust(bottom=0.1)
    return fig_bar

def generateLine_freeCash(df_trans_visualize:pd.DataFrame,figHeight:int=100,figWidth:int=100) -> plt.figure:
    '''Generate line chart of free cash flow '''

    fig_pie = plt.figure()


    trans_monthSpend = df_trans_visualize.groupby(by=['year','month'])['amount'].sum()
    
    
    # Start forming the pie chart 
    trans_monthSpend_dict = trans_monthSpend.to_dict()
    label_pie,vals_pie = range(len(list(trans_monthSpend_dict.keys()))),list(trans_monthSpend_dict.values())
    print(label_pie,vals_pie)
    # Create the chart 
    plt.bar(x=label_pie,height=vals_pie)
    plt.xticks(label_pie,list(trans_monthSpend_dict.keys()))

    return fig_pie





if __name__ == '__main__':
    print('visulization engine has no main method')
    sys.exit(0)




