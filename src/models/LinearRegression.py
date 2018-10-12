#misc
import sys
import argparse
#data processing
import numpy as np
#keras
from sklearn.linear_model import LinearRegression
#homemade
sys.path.append('../../utils')
from error_reporting import error_reporting_regression,error_histogram,error_time_series
from helpers import save_sklearn_model
sys.path.append('../data_cleaning')
from grand import process_grand,clean_grand


def linear_model(lookbacklength,lookforwardlength,test_split):
    df = clean_grand()
    X_train,X_test,y_train,y_test = process_grand(df,lookbacklength=lookbacklength,lookforwardlength=lookforwardlength,test_split=test_split)
    X_train,X_test = np.squeeze(X_train,axis=2),np.expand_dims(X_test,axis=2)
    clf = LinearRegression()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    
    save_sklearn_model(clf,'LinearRegressor')
    
    error_reporting_regression(y_test,y_pred)
    
    error_histogram(y_test,y_pred)
    
    error_time_series(y_test,y_pred)
    
    
    
    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("lookbacklength", help="How far back in time the model looks",
                    type=int)
    parser.add_argument("lookforwardlength",help='How far in future model tries to predict',
                        type=int)
    parser.add_argument("test_split",help="Proportion of data that goes into test set",type=float)
    args = parser.parse_args()
    linear_model(lookbacklength=args.lookbacklength,
           lookforwardlength=args.lookforwardlength,
           test_split=args.test_split)
    