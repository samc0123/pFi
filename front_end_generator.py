from tkinter import *
from tkinter import messagebox
import pandas as pd
import sys
import numpy as np 
import matplotlib as mpl
from datetime import datetime as dt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)

from predictionGrouping import group_trans_month_cat
from visualizeTrans import generateBarChart

class datesFromSlider:
    def __init__(self,from_month = None,from_year = None,to_month = None,to_year = None) -> None:
        '''Create a constructor class storing the parameters from the slider'''
        self.from_month = from_month
        self.from_year = from_year
        self.to_month = to_month
        self.to_year = to_year



    def checkValidConstructor(self):
        self.from_date = pd.to_datetime(arg=f'{self.from_year}-{self.from_month}-01')
        self.to_date = pd.to_datetime(arg=f'{self.to_year}-{self.to_month}-28')

        if self.from_date > self.to_date:
            messagebox.showerror(title='Invalid Date Selection',message='From Date must be less than or equal to to date')
        else:
            messagebox.showinfo(title='Date Selection Configured',message=f'Dates between {self.from_date.strftime("%Y-%m-%d")} and {self.to_date.strftime("%Y-%m-%d")} configured')
    
    def printVals(self):
        print(f'From month:{self.from_month} & From Year:{self.from_year}\
            To Month:{self.to_month} & To Year:{self.to_year}')
    

def retrieveDateValues_fromSliders():
    '''Pull date filters from sliders into class'''
    sliderVals.from_month=slider_dates_month_from.get()
    sliderVals.from_year=slider_dates_year_from.get()
    sliderVals.to_month=slider_dates_month_to.get()
    sliderVals.to_year=slider_dates_year_to.get()
    
    
    sliderVals.checkValidConstructor()
    sliderVals.printVals()

def filter_df_by_date(df_to_filter:pd.DataFrame) -> pd.DataFrame:
    '''Filter dataframe based on attributes set in the slider values'''
    df_to_filter['day'] = int(1) # add day to convert to datetime
    df_to_filter['date_filter'] = pd.to_datetime(dict(\
        year=df_to_filter.year, \
            month=df_to_filter.month, \
                day=df_to_filter.day))
    df_to_filter = df_to_filter[(\
        df_to_filter['date_filter'] >= sliderVals.from_date\
    ) & (\
        df_to_filter['date_filter'] <= sliderVals.to_date)]
    
    return(df_to_filter)

def createPlot_catSpend_bar():
    '''Generate bar graph for category spending'''
    ## Spending by category 

    # Created category grouping dataframe 
    df_cat_and_month_grouped = group_trans_month_cat(df_transactions=df_masterTrans)
    df_cat_filtered = filter_df_by_date(df_to_filter=df_cat_and_month_grouped)
    
    print(df_cat_filtered)
    # Create bar graph to plot
    figBar = generateBarChart(df_trans_visualize=df_cat_filtered)

    # Add canvas to display graph
    canvas = FigureCanvasTkAgg(figBar,master = window)  
    canvas.draw()
    canvas.get_tk_widget().grid(row=0,column=3,rowspan=3,pady=30)

def createPlot_freeCash_pie():
    '''Create a pie chart of free cash flow per month'''

    # Created category grouping dataframe 
    df_cat_and_month_grouped = group_trans_month_cat(df_transactions=df_masterTrans)
    df_cat_filtered = filter_df_by_date(df_to_filter=df_cat_and_month_grouped)
    
    print(df_cat_filtered)

    figPie = None # TODO: finish plotting the pie chart for free cash flow



# Import master_transFile for information querying 
masterTransPath = r'/Users/samchernov/Desktop/Personal/Financials/Transaction_Export/pFi/Master_transaction_file/master_transactions_manReview.csv'
df_masterTrans = pd.read_csv(filepath_or_buffer=masterTransPath)

# Intialize the tkinter window
window = Tk()
window.geometry('1200x800')
window.minsize(width=400,height=400)
window.maxsize(width=1600,height=800)
window.title('Bob the Budgetary')


## Format sliders for date selection, use button to retrieve values

# Sliders from
slider_dates_month_from = Scale(master=window,from_=df_masterTrans['month'].min(),to=df_masterTrans['month'].max(),label='Select Month From',\
    orient=HORIZONTAL,length=200)
slider_dates_month_from.grid(column=0,row=0)
slider_dates_year_from = Scale(master=window,from_=df_masterTrans['year'].min(),to=df_masterTrans['year'].max(),label='Select Year From',\
    orient=HORIZONTAL,length=200)
slider_dates_year_from.grid(column=0,row=1)
# Sliders to 
slider_dates_month_to = Scale(master=window,from_=df_masterTrans['month'].min(),to=df_masterTrans['month'].max(),label='Select Month To',\
    orient=HORIZONTAL,length=200)
slider_dates_month_to.grid(column=1,row=0)
slider_dates_year_to = Scale(master=window,from_=df_masterTrans['year'].min(),to=df_masterTrans['year'].max(),label='Select Year To',\
    orient=HORIZONTAL,length=200)
slider_dates_year_to.grid(column=1,row=1)


# Get date values 
sliderVals = datesFromSlider()
getDatesvals_but = Button(master=window, text='Confirm Date Range',command=retrieveDateValues_fromSliders)
getDatesvals_but.grid(column=0,row=2,columnspan=2)



### Display three visuals: spending by category, spending by month, spending by month by category 

# Bar chart category per month
but_generate_catSpend_bar = Button(master=window,text='Generate Category Spending Chart',command=createPlot_catSpend_bar)
but_generate_catSpend_bar.grid(row=4,column=0,columnspan=2)

# Pie chart free cash flow
but_generate_catSpend_bar = Button(master=window,text='Generate Free Cash Flow',command=createPlot_freeCash_pie)
but_generate_catSpend_bar.grid(row=5,column=0,columnspan=2)




window.mainloop()



