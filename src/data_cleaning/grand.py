#misc
import sys
#data processing
import numpy as np
import pandas as pd
#homemade
from data_cleaning_helpers import time_diff
sys.path.append('../../')
from utils.preprocessing import temp_forecasting_shape_processing,test_train_split
from utils.helpers import load_tsv

def clean_grand():
    filename = 'Grand_Geyser_Pool_Logger'
    df = load_tsv(filename)
    return df

def interpolate_grand(df):
    delta = time_diff(df['datetime'])
    assert (delta[1:]>0).all(),"Datetimes not monotonically inc"
    #to automate, look at value counts and the mod of the deltatime to 60. Pick small values
    #rn, we're just doing only specifically delta t = 121
    idx_121 = delta[delta==121.0].index[0]
    time_before = df.loc[idx_121-1]
    time_after = df.loc[idx_121]
    new_datetime = time_before['datetime'] + 60
    new_temp = (time_before['temp'] + time_after['temp'])/2
    #insert new row
    df.loc[len(df)] = [new_datetime,new_temp]
    #sort based on datetime
    #we can do this because we know datetime was monotonically inc.
    df = df.sort_values(by='datetime',axis=0).reset_index(drop=True)
    return df
    
def preprocess_grand(df,interpolate=True):
    if interpolate:
        #preprocessing - if delta time is 120, then just interpolate
        df = interpolate_grand(df)
    delta = time_diff(df['datetime'])  
    off_delta = delta[delta>62.0]
    off_delta_vals = df.loc[off_delta.index,:]['temp']
    df_partition = []
    for avoid_idx in off_delta.index.tolist()[1:]:
        df_partition.append(df.loc[:avoid_idx-1])
        df = df.loc[avoid_idx:]
    df_partition.append(df)
    return df_partition
def process_grand(df,lookbacklength,lookforwardlength,test_split,interpolate=True):
    df_partition = preprocess_grand(df,interpolate)
    X,y = [],[]
    for df in df_partition:
        #make sure df has enough records to satisfy lookbacklength + lookforwardlength
        if len(df) < lookbacklength + lookforwardlength:
            continue
        Xx,yy = temp_forecasting_shape_processing(df['temp'].to_frame(),lookbacklength,lookforwardlength)
        X.append(Xx)
        y.append(yy)
    X,y = np.concatenate(X),np.concatenate(y)
    X_train,X_test,y_train,y_test = [],[],[],[]
    Xx_train,Xx_test,Yy_train,Yy_test = test_train_split(X,y,test_split,random=False)
    X_train.append(Xx_train)
    X_test.append(Xx_test)
    y_train.append(Yy_train)
    y_test.append(Yy_test)
    X_train,X_test,y_train,y_test = np.concatenate(X_train),np.concatenate(X_test),np.concatenate(y_train),np.concatenate(y_test)
    assert X_train.shape[0] == y_train.shape[0],"Incorrect shapes of X_train,y_train"
    assert X_test.shape[0] == y_test.shape[0],"Incorrect shapes of X_test,y_test"
    assert X_train.shape[1:] == X_test.shape[1:],"Incorrect shapes of X_train,X_test"
    assert X_train.shape[1] == lookbacklength,"Incorrect input shape based on lookbacklength"
    return X_train,X_test,y_train,y_test

#lookforwardlength=1 for training, lookforwardlength=lookforwardlength during testing
def process_grand_recurrent(df,lookbacklength,lookforwardlength,test_split):
    time1 = df['datetime']
    time2 = df['datetime'].shift(1)
    delta = time2 - time1
    off_delta = delta[delta!=-60.0]
    off_delta_vals = df.loc[off_delta.index,:]['temp']
    df_partition = []
    tmp = df.copy()
    for avoid_idx in off_delta.index.tolist()[1:]:
        df_partition.append(tmp.loc[:avoid_idx-1])
        tmp = tmp.loc[avoid_idx:]
    split_idx = round(sum([len(x) for x in df_partition]) * (1-test_split))
    X_train,y_train, = [],[]
    count = 0
    for df_idx in range(len(df_partition)):
        df = df_partition[df_idx]
        count = count + len(df)
        #make sure df has enough records to satisfy lookbacklength + lookforwardlength
        if len(df) < lookbacklength + lookforwardlength:
            continue
        if count > split_idx:
            this_df_split_idx = count - split_idx
            train_df = df.iloc[:this_df_split_idx]
            test_df = df.iloc[this_df_split_idx:]
            Xx,yy = temp_forecasting_shape_processing(train_df['temp'].to_frame(),lookbacklength,1)
            X_train.append(Xx)
            y_train.append(yy)
            break
        Xx,yy = temp_forecasting_shape_processing(df['temp'].to_frame(),lookbacklength,1)
        X_train.append(Xx)
        y_train.append(yy)
    df_partition = df_partition[df_idx+1:]
    df_partition.append(test_df)
    X_test,y_test = [],[]
    for df_idx in range(len(df_partition)):
        df = df_partition[df_idx]
        count = count + len(df)
        #make sure df has enough records to satisfy lookbacklength + lookforwardlength
        if len(df) < lookbacklength + lookforwardlength:
            continue
        Xx,yy = temp_forecasting_shape_processing(df['temp'].to_frame(),lookbacklength,lookforwardlength)
        X_test.append(Xx)
        y_test.append(yy)
    X_train,X_test,y_train,y_test = np.concatenate(X_train),np.concatenate(X_test),np.concatenate(y_train),np.concatenate(y_test)
    assert X_train.shape[0] == y_train.shape[0],"Incorrect shapes of X_train,y_train"
    assert X_test.shape[0] == y_test.shape[0],"Incorrect shapes of X_test,y_test"
    assert X_train.shape[1:] == X_test.shape[1:],"Incorrect shapes of X_train,X_test"
    assert X_train.shape[1] == lookbacklength,"Incorrect input shape based on lookbacklength"
    return X_train,X_test,y_train,y_test