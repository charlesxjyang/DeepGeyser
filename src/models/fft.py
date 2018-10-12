#misc
import sys
#data processing
import numpy as np
import pandas as pd
from scipy.fftpack import rfft
from scipy import optimize
#plotting
import matplotlib.pyplot as plt
#home-made
sys.path.append('../../utils')
from preprocessing import temp_forecasting_shape_processing,test_train_split
from error_reporting import error_reporting_regression,error_histogram,error_time_series,keras_training_loss_curve
from helpers import save_model,load_tsv
sys.path.append('../data_cleaning')
from grand import process_grand,clean_grand

def curve_func(x, a, b, c, d, e, f, g, h):
        return a * np.sin(b * x) + c * np.cos(d * x) + e * np.sin(f * x) + g * np.cos(h * x)
 
def curve_func_2(x, a, b):
    return a * np.sin(b * x)

def scipy_curve_fit(data,func):
    data = data.flatten()
    x_data = np.array(range(len(data)))
    y_data = data
    params, params_covariance = optimize.curve_fit(func, x_data, y_data,
                                               p0=[2, 2])
    pred = func(data,params[0],params[1]).flatten()
    error_reporting_regression(data,pred)
    return params,pred

def just_trying_fft(n=100):
    df = clean_grand()
    temp = df['temp'].values
    spectra = np.fft.rfft(temp,n=n)
    plt.plot(spectra)
    
def fft_to_pred(theta,cn_vec):
    n = len(cn_vec)
    total = 0
    for i in range(n):
        ea = np.exp(n*theta*np.array([0+1j]))
        total = cn_vec[i] *ea + total
    return total

def fft_model(lookbacklength,lookforwardlength,test_split):
    df = clean_grand()
    X_train,X_test,y_train,y_test = process_grand(df,lookbacklength=lookbacklength,lookforwardlength=lookforwardlength,test_split=test_split)
