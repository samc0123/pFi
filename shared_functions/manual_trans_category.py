import pandas as pd 
import numpy as np 
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tss
from sklearn.preprocessing import LabelEncoder

def manual_categorize_transactions(raw_transaction_data:pd.DataFrame) -> pd.DataFrame:
    ''' Manually categorize [raw_transaction_data] for ML training data '''
    transaction_cateogry_dict = {
        "Entertainment":"AMC",
        "Grocery":"Aldi,Trader Joe's,Wegmans,NC700",
        "Online":"Amazon,eBay,",
        "Wholesale":"Walmart,Target"
    }
    print(transaction_cateogry_dict.get('Grocery'))

def count_most_frequent_words(data:pd.DataFrame,df_col_to_countFreqWords:str,countWords:int):
    '''Count the most frequent words in given column of a dataframe'''
    most_common_words = Counter(" ".join(data[df_col_to_countFreqWords].str.lower()).split()).most_common(countWords)
    most_common_df = pd.DataFrame(data=most_common_words).reset_index()
    most_common_df = most_common_df.drop(labels=['index'],axis=1)
    most_common_df = most_common_df.rename(columns={0:'word',1:'count'})
    most_common_df.to_csv(path_or_buf=f'/Users/samchernov/Desktop/Personal/Financials/Transaction_Export/mostCommonTrans{countWords}.csv')
    print(most_common_df)

def transaction_knn_model(training_dataset:pd.DataFrame,actual_dataset:pd.DataFrame) :
    '''Run knn algo on training_dataset to try to predict values in actual_dataset'''
    # Create encoder labels for each categorical dataset
    lblEncodeX = LabelEncoder()
    lblEncodey = LabelEncoder()
    lblEncodeact = LabelEncoder()
    
    # Split training data for model based on word frequency in transactions
    X = lblEncodeX.fit_transform(training_dataset["nameTrans"]).reshape(-1,1)  
    y = lblEncodey.fit_transform(training_dataset["category"]).reshape(-1,1).ravel() # needed for proper shaping to the model
    x_actual = lblEncodeact.fit_transform(actual_dataset["Payee"]).reshape(-1,1) # actual dataset to predict on
    X_train,X_test,y_train,y_test = tss(X,y,test_size=0.2,random_state=1,stratify=y) # random state - matches the training set breakdown in\
        # test, stratify ensures breakdown across all columns 
   
    # Create instance of the knn algorithm
    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X_train,y_train)

    
    # Test model accuracy 
    print(knn.score(X_test,y_test))
    preds = knn.predict(X_test)
    preds = lblEncodey.inverse_transform(preds)
    vals = lblEncodeX.inverse_transform(X_test)
    print(vals,preds)

    ## TODO: continue to test model, accuracy is pretty good (73%), but test data transactions spit out incorrect cateogries completely 
    # Predict categories for the model according to the training data

    """  preds = knn.predict(x_actual)
    preds = lblEncodey.inverse_transform(preds)
    vals = lblEncodeact.inverse_transform(x_actual)
     
     # Put the output into a dictorionary 
    data_predictedCategories_forTrans = {
        "nameTrans": vals,
        "category": preds
    }

    # Create dataframe and csv file for outputting predictions 
    df_predictedCategories = pd.DataFrame(data=data_predictedCategories_forTrans)
    df_predictedCategories.to_csv(path_or_buf='/Users/samchernov/Desktop/Personal/Financials/Transaction_Export/pFi/predictedCats_knn.csv')
    return("Test succeeded") """