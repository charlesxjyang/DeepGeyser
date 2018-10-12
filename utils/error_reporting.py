#data analysis
import pandas as pd
import numpy as np
#sklearn helpers
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
#plotting
import matplotlib.pyplot as plt


def error_reporting_regression(y_test,y_pred):
    assert len(y_test) == len(y_pred),"Dimensions do not match"
    y_test, y_pred = y_test.flatten(), y_pred.flatten()
    print("Error Reporting for Regression")
    num_ex = float(len(y_test)) - 1
    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    mae_arr = np.absolute(np.subtract(y_test,y_pred))
    sorted_mae = np.sort(mae_arr,axis=0)
    print("Mean Absolute Error: {0}".format(mae))
    print("Mean Squared Error: {0}".format(mse))
    print("R2: {0}".format(r2))
    print("Minimum Absolute Error: {0}".format(round(sorted_mae[0],4)))
    print("10% Minimum Absolute Error: {0}".format(round(sorted_mae[round(0.1*num_ex)],4)))
    print("25% Minimum Absolute Error: {0}".format(round(sorted_mae[round(0.25*num_ex)],4)))
    print("50%(Median) Minimum Absolute Error: {0}".format(round(sorted_mae[round(0.5*num_ex)],4)))
    print("75% Minimum Absolute Error: {0}".format(round(sorted_mae[round(0.75*num_ex)],4)))
    print("90% Minimum Absolute Error: {0}".format(round(sorted_mae[round(0.9*num_ex)],4)))
    print("Maximum Absolute Error: {0}".format(round(sorted_mae[int(num_ex)],4)))
    
    
def error_histogram(y_test,y_pred):
    mae = np.subtract(y_test,y_pred)
    fig,ax = plt.subplots()
    ax.hist(mae,bins=20,log=True)
    ax.set_xlabel("Actual - Predicted")
    ax.set_ylabel("Count")
    ax.set_title("Mean Absolute Error Histogram")
    
def error_time_series(y_test,y_pred):
    error = np.subtract(y_test,y_pred)
    fig,ax = plt.subplots()
    ax.plot(error)
    ax.set_xlabel("Index")
    ax.set_ylabel("Actual - Predicted")
    ax.set_title("Actual - Predicted Error")
    
def compare_time_series(y_test,y_pred):
    fig,ax = plt.subplots(2,sharex=True)
    ax[0].set_title("Plotting actual and predicted time series")
    ax[0].plot(y_test)
    ax[0].set_ylabel("Test")
    ax[1].plot(y_pred)
    ax[1].set_xlabel("Time(min)")
    ax[1].set_ylabel("Predicted")

def keras_training_loss_curve(history):
    loss = history.history['loss']
    fig,ax = plt.subplots()
    ax.plot(loss)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")