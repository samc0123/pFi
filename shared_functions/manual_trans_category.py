import pandas as pd 
import numpy as np 
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tss
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

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

def transaction_LSVC_model(training_dataset:pd.DataFrame,actual_dataset:pd.DataFrame) :
    '''Run knn algo on training_dataset to try to predict values in actual_dataset'''
    # Create encoder labels for each categorical dataset
    lblEncodeX = LabelEncoder()
    lblEncodey = LabelEncoder()
    lblEncodeact = LabelEncoder()
    
    # Split training data for model based on word frequency in transactions
    X = lblEncodeX.fit_transform(training_dataset["nameTrans"]).reshape(-1,1)  
    y = lblEncodey.fit_transform(training_dataset["category"]).reshape(-1,1).ravel() # needed for proper shaping to the model
    x_actual = lblEncodeact.fit_transform(actual_dataset["Payee"]).reshape(-1,1) # actual dataset to predict on
    X_train,X_test,y_train,y_test = tss(X,y,test_size=0.2) # random state - matches the training set breakdown in\
        # test, stratify ensures breakdown across all columns 

    # Instatiate linear SVC 
    lsvc = LinearSVC(verbose=0, max_iter=2000)
    lsvc.fit(X_train,y_train)

    print(lsvc.score(X_test,y_test))

def transaction_linearReg_model(training_dataset:pd.DataFrame,actual_dataset:pd.DataFrame):
    '''Run linear reg algo to predict the bank transaction categories'''
    # Create encoder labels for each categorical dataset
    lblEncodeX = LabelEncoder()
    lblEncodey = LabelEncoder()
    lblEncodeact = LabelEncoder()
    
    # Split training data for model based on word frequency in transactions
    X = lblEncodeX.fit_transform(training_dataset["nameTrans"]).reshape(-1,1)  
    y = lblEncodey.fit_transform(training_dataset["category"]).reshape(-1,1) # needed for proper shaping to the model
    x_actual = lblEncodeact.fit_transform(actual_dataset["Payee"]).reshape(-1,1) # actual dataset to predict on
    X_train,X_test,y_train,y_test = tss(X,y,test_size=0.2) # random state - matches the training set breakdown in\
        # test, stratify ensures breakdown across all columns 

    # Scale variables 
    scaler_x = preprocessing.StandardScaler().fit(X_train)
    X_scaled = np.asarray(scaler_x.transform(X_train)).reshape(-1,1)

    scaler_y = preprocessing.StandardScaler().fit(y_train)
    y_scaled = np.asarray(scaler_y.transform(y_train))

    # Instantiate the linear reg model 
    linear_reg = LinearRegression()
    linear_reg.fit(X_scaled,y_scaled)

    # Scale outputs 
    X_test_scaler = preprocessing.StandardScaler().fit(X_test)
    X_test_scaled = np.asarray(X_test_scaler.transform(X_test)).reshape(-1,1)

    y_test_scaler = preprocessing.StandardScaler().fit(y_test)
    y_test_scaled = np.asarray(y_test_scaler.transform(y_test))


    # See model performance 
    print(linear_reg.score(X_test_scaled,y_test_scaled))

def transaction_bayes_model(training_dataset:pd.DataFrame,actual_dataset:pd.DataFrame):
     # Create encoder labels for each categorical dataset
    lblEncodeX = LabelEncoder()
    lblEncodey = LabelEncoder()
    lblEncodeact = LabelEncoder()
    ordEncoderX = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    ordEncoderXAct = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    
    # Split training data for model based on word frequency in transactions
    bank_trans_data ={
        'training_trans': training_dataset["nameTrans"],
        'trans_to_predict': actual_dataset["Payee"]
    }
    bank_trans_array = np.concatenate((training_dataset["nameTrans"],actual_dataset["Payee"]),axis=None)
    bank_trans_df = pd.DataFrame(bank_trans_data)
    # Encode transactions from banking statements 
    ordEncoderX.fit(bank_trans_array.reshape(-1,1))
    bank_trans_df[['training_trans']] = ordEncoderX.transform(bank_trans_df[['training_trans']])
    bank_trans_df[['trans_to_predict']] = ordEncoderX.transform(bank_trans_df[['trans_to_predict']])
   
    
    X = np.asarray(bank_trans_df['training_trans']).reshape(-1,1)
    y = lblEncodey.fit_transform(training_dataset["category"]).reshape(-1,1).ravel() # needed for proper shaping to the model
    x_actual = np.asarray(bank_trans_df['trans_to_predict'])
    x_actual = np.delete(x_actual,np.where(x_actual == -1)).reshape(-1,1)
    X_train,X_test,y_train,y_test = tss(X,y,test_size=0.2) # random state - matches the training set breakdown in\
        # test, stratify ensures breakdown across all columns 

    """  # Scale variables train
    scaler_x = preprocessing.StandardScaler().fit(X_train)
    X_scaled = np.asarray(scaler_x.transform(X_train)).reshape(-1,1)

    scaler_y = preprocessing.StandardScaler().fit(y_train)
    y_scaled = np.asarray(scaler_y.transform(y_train)) 
    
    # Sepeate ordinal encoders 
    ordEncoderX.fit(bank_trans_df[['training_trans']])
    bank_trans_df[['training_trans']] = ordEncoderX.transform(bank_trans_df[['training_trans']])

    ordEncoderXAct.fit(bank_trans_df[['trans_to_predict']])
    bank_trans_df[['trans_to_predict']] = ordEncoderXAct.transform(bank_trans_df[['trans_to_predict']])
    """

    # Instantiate the bayes model
    gnb = GaussianNB()
    gnb.fit(X_train,y_train)
    #print(gnb.score(X_test,y_test))
    #print(ordEncoderX.inverse_transform(X_test))
    #print(lblEncodey.inverse_transform(y_test))
     # Fit model with data 
    vals = gnb.predict(X=x_actual)
    print(ordEncoderX.inverse_transform(x_actual))
    print(lblEncodey.inverse_transform(vals))

    """ # Scale test
    X_test_scaler = preprocessing.StandardScaler().fit(X_test)
    X_test_scaled = np.asarray(X_test_scaler.transform(X_test)).reshape(-1,1)

    y_test_scaler = preprocessing.StandardScaler().fit(y_test)
    y_test_scaled = np.asarray(y_test_scaler.transform(y_test)) """

    print(gnb.score(X_test,y_test))
