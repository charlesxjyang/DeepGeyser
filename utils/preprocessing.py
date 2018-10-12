import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from helpers import datetime_to_unix,unix_to_datetime


'''
Processes panda df that gives the eruption event times in epoch time
Only uses eruption data not interpolated time range
'''
def eruption_time_processing(df:pd.DataFrame,lookbacklength:int):
    x,y = temp_forecasting_shape_processing(df,lookbacklength,1)
    y = np.expand_dims(y,2)
    overall = np.concatenate([x,y],axis=1)
    #takes array and returns how far ago previous eruptions were
    def past_eruption_times(np_arr):
        np_arr = np_arr - np_arr[0,:]
'''
Processes panda df that gives the eruption event times in epoch time
Picks bunch of linearly spaced time points, calculates time away
from previous eruptions and time to next eruption
linearly dependent samples ?
'''
def eruption_time_processing_2(df:pd.DataFrame,lookbacklength:int,n=10**4):
    min_val,max_val = df['eruption_time_epoch'].min(),df['eruption_time_epoch'].max()
    n_points = np.linspace(min_val,max_val,n)
    def find_vals(time_val,list_of_eruptions,lookbacklength,start_idx):
        '''
        Given time value time_val and a list of time values,
        find lookbacklength number of values in list of time values
        that are closest to time_val but do not occur after as well
        as the value immediately after
        start_idx is a caching mechanism. Tells us where to start looking
        '''
        for i in range(start_idx,len(list_of_eruptions)):
            if list_of_eruptions[i]>=time_val:
                break
        i = i - 1
        return list_of_eruptions[i-lookbacklength:i],list_of_eruptions[i+1],i
    X,y = np.empty((n,lookbacklength)),np.empty((n,1))
    last_idx = 0
    for time in n_points:
        pre_x,pre_y,idx = find_vals(time,df['eruption_time_epoch'].values,lookbacklength,last_idx)
        if idx<lookbacklength:
            continue
        pre_x,pre_y = np.array(pre_x).reshape(1,-1),np.array([pre_y]).reshape(1,1)
        pre_x = np.abs(time - pre_x)
        pre_y = np.abs(time - pre_y)
        X = np.append(X,pre_x,axis=0)
        y = np.append(y,pre_y,axis=0)
        last_idx = idx
    X,y = np.asarray(X),np.array(y)
    assert len(X) == len(y),"X,y have different lengths"
    return X,y
    

def eruption_time_processing_datetime(df:pd.DataFrame,lookbacklength:int,n=10**4):
    min_val,max_val = df['eruption_time_epoch'].min(),df['eruption_time_epoch'].max()
    n_points = np.linspace(min_val,max_val,n)
    def find_vals(time_val,list_of_eruptions,lookbacklength,start_idx):
        '''
        Given time value time_val and a list of time values,
        find lookbacklength number of values in list of time values
        that are closest to time_val but do not occur after as well
        as the value immediately after
        start_idx is a caching mechanism. Tells us where to start looking
        '''
        for i in range(start_idx,len(list_of_eruptions)):
            if list_of_eruptions[i]>=time_val:
                break
        i = i - 1
        return list_of_eruptions[i-lookbacklength:i],list_of_eruptions[i+1],i
    X,y = np.empty((n,lookbacklength)),np.empty((n,1))
    last_idx = 0
    for time in n_points:
        dt = unix_to_datetime(time)
        pre_x,pre_y,idx = find_vals(dt,df['eruption_time'],lookbacklength,last_idx)
        if idx<lookbacklength:
            continue
        pre_x,pre_y = np.array(pre_x).reshape(1,-1),np.array([pre_y]).reshape(1,1)
        pre_x = dt - pre_x
        pre_y = dt - pre_y
        X = np.append(X,pre_x,axis=0)
        y = np.append(y,pre_y,axis=0)
        last_idx = idx
    X,y = np.asarray(X),np.array(y)
    assert len(X) == len(y),"X,y have different lengths"
    return X,y
    
'Processes panda df that gives temperature measurements'
def temp_forecasting_shape_processing(df:pd.DataFrame,lookbacklength:int,lookforwardlength:int,ewm=True,alpha=0.1):
    assert len(df) > lookbacklength+lookforwardlength, "Ensure dataframe has enough records to accomodate request"
    df = df.ewm(alpha=alpha).mean().dropna()
    m,n = df.shape
    y = df.iloc[lookbacklength+lookforwardlength-1:].copy().values
    new_df = df.iloc[:len(df)-lookforwardlength]
    X = []
    #reversed so that reading array from left to right is 
    #read from t_0 to t_n 
    for i in reversed(range(lookbacklength)):
        X.append(new_df.shift(i).values)
    x = np.array(X)
    x = x[:,lookbacklength-1:,:]
    x = np.swapaxes(x,0,1) 
    assert len(x) == len(y), "Incorrect Output shape"
    assert not np.isnan(x).any(), "No Nan values allowed"
    assert x[0,0,0] in df.iloc[0,:].values, "First value not included"
    #if we check all of the array, with very long lookbacklength, then odds are
    #there will randomly be one in there
    assert y[0] not in x[0,0,:lookforwardlength+1],"Y values in same X"
    return x,y
    
    
'''
Test train split on X,y
random determines whether to randomly shuffle or to partition
into two continuous segments
normalization standardizes features to have mean 0 std 1
'''    
def test_train_split(X,y,test_split=0.2,random=False,normalization=False):        
    if isinstance(X,pd.DataFrame):
        X = X.values
    if isinstance(y,pd.DataFrame):
        y = y.values
    if random:
        X_train,X_test,y_train,y_test = train_test_split(test_split)
        
    else:
        num_ex = len(X)
        split_idx = int(float(num_ex) * (1-test_split))
        X_train,X_test = X[:split_idx],X[split_idx:]
        y_train,y_test = y[:split_idx],y[split_idx:]
    if normalization:
        clf = StandardScaler()
        clf.fit(X_train)
        X_train,X_test = clf.transform(X_train),clf.transform(X_test)
        return X_train,X_test,y_train,y_test
    else:
        return X_train,X_test,y_train,y_test