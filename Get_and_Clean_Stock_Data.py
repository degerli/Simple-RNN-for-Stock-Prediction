import numpy as np
import pandas as pd
import quandl

def get_data_from_quandl (tickers, start_col, end_col):
    '''
    This function specifically gets stocks from Quandl's WIKI dataset. 
    The start and end columns refers to (adjusted) OHLC, volume.
    '''
    quandl.ApiConfig.api_key = 'APIKEYFILLER' # Insert your own API Key
    quandl.ApiConfig.api_version = '2015-04-09' # recommended version
    final_ticker = []
    for ticker in tickers:
        for i in range (start_col, end_col + 1):
            final_ticker.append ('WIKI/' + ticker + '.' + str (i))
    data = quandl.get(final_ticker).pct_change()
    return data

def create_train_and_test (df, ratio):
    data = df.dropna()
    num_rows_train = int (round (data.shape[0] * ratio))
    train, test = data[:num_rows_train], data[num_rows_train:]
    return train, test

def clean_data (train_data):
    data = train_data.replace (0, np.nan)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Imputer, StandardScaler
    trans_pipeline = Pipeline([
        ('imputer', Imputer(strategy='median')),
        #('std_scaler', StandardScaler()) # decided not to use it
    ])
    transformed_data = trans_pipeline.fit_transform (data)
    return pd.DataFrame (transformed_data, index = data.index, columns = data.columns)

def create_csv_file(train, test, file_name):
    train.to_csv('CleanedData/' + file_name + '_train.csv')
    test.to_csv('CleanedData/' + file_name + '_test.csv')
    
def get_clean_to_csv (tickers, start_col, end_col, ratio, file_name):
    data = get_data_from_quandl (tickers, start_col, end_col)
    train, test = create_train_and_test (data, ratio)
    cleaned_data = clean_data (train)
    create_csv_file(cleaned_data, test, file_name)
    return cleaned_data
