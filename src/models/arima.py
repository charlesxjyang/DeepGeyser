#misc
import sys
#data processing
import numpy as np
import pandas as pd
from scipy.fftpack import rfft
from scipy import optimize
from statsmodels.tsa.arima_model import ARIMA
#plotting
import matplotlib.pyplot as plt
#home-made
sys.path.append('../../utils')
from preprocessing import temp_forecasting_shape_processing,test_train_split
from error_reporting import error_reporting_regression,error_histogram,error_time_series,keras_training_loss_curve
from helpers import save_model,load_tsv
sys.path.append('../data_cleaning')
from grand import process_grand,clean_grand

df = clean_grand()
series = df['temp'].values
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)
#error_reporting_regression
X = series
size = int(len(X) * 0.999)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
lookforwardlength=1
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(steps=lookforwardlength)
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    if i % 10 == 0:
        print(i)
error_reporting_regression(test,predictions)