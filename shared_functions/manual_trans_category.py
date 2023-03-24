import pandas as pd 
import numpy as np 
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tss
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing




def count_most_frequent_words(data:pd.DataFrame,df_col_to_countFreqWords:str,countWords:int):
    '''Count the most frequent words in given column of a dataframe'''
    most_common_words = Counter(" ".join(data[df_col_to_countFreqWords].str.lower()).split()).most_common(countWords)
    most_common_df = pd.DataFrame(data=most_common_words).reset_index()
    most_common_df = most_common_df.drop(labels=['index'],axis=1)
    most_common_df = most_common_df.rename(columns={0:'word',1:'count'})
    most_common_df.to_csv(path_or_buf=f'/Users/samchernov/Desktop/Personal/Financials/Transaction_Export/mostCommonTrans{countWords}.csv')
    print(most_common_df)

def transaction_knn_model(training_dataset:pd.DataFrame,actual_dataset:pd.DataFrame,columns_indep_dep:dict)  -> pd.DataFrame: 
    '''Run knn algo on training_dataset to try to predict values in actual_dataset. columns_indep_dep contain names
    of columns to work with: {'training_independent':'x','training_dependent','y','actual_indepedent':'z'.} Training_dependent
    will be name of column in outputdb '''
    # Create encoder labels for each categorical dataset
    lblEncodeX = LabelEncoder()
    lblEncodey = LabelEncoder()
    lblEncodeact = LabelEncoder()
    ordEncoderX = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    ordEncoderXAct = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)

    # Assign names for columns
    training_independent_col = columns_indep_dep['training_independent']
    dependent_col = columns_indep_dep['training_dependent']
    actual_independent_col = columns_indep_dep['actual_independent']

    
    # Split training data for model based on word frequency in transactions
    bank_trans_data ={
        'training_trans': training_dataset[training_independent_col],
        'trans_to_predict': actual_dataset[actual_independent_col]
    }
    bank_trans_array = np.concatenate((training_dataset[training_independent_col],actual_dataset[actual_independent_col]),axis=None)
    bank_trans_df = pd.DataFrame(bank_trans_data)
    # Encode transactions from banking statements 
    ordEncoderX.fit(bank_trans_array.reshape(-1,1))
    bank_trans_df[['training_trans']] = ordEncoderX.transform(bank_trans_df[['training_trans']])
    bank_trans_df[['trans_to_predict']] = ordEncoderX.transform(bank_trans_df[['trans_to_predict']])
   
    
    X = np.asarray(bank_trans_df['training_trans'])
    X = np.delete(X,np.where(X == -1)).reshape(-1,1)
    y = lblEncodey.fit_transform(training_dataset[dependent_col]).reshape(-1,1).ravel() # needed for proper shaping to the model
    x_actual = np.asarray(bank_trans_df['trans_to_predict'])
    x_actual = np.delete(x_actual,np.where(x_actual == -1)).reshape(-1,1)
    X_train,X_test,y_train,y_test = tss(X,y,test_size=0.2)

    # Create instance of the knn algorithm
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)

    
    # Test model accuracy 
    print(knn.score(X_test,y_test))

    # Add output column to actual dataframe
    actual_dataset[dependent_col] = lblEncodey.inverse_transform(knn.predict(x_actual))
    actual_dataset['amount'] = pd.to_numeric(actual_dataset['amount'])
    actual_dataset['datePosted'] = pd.to_datetime(actual_dataset['datePosted'])
    actual_dataset.sort_values(by='datePosted',inplace=True)
    actual_dataset.reset_index(inplace=True,drop=True)
    return actual_dataset

