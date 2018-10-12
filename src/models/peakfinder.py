#misc
import sys
#data processing
import pandas as pd
import numpy as np
from scipy.signal import find_peaks,find_peaks_cwt
import matplotlib.pyplot as plt
# homemade
sys.path.append('../../utils')
from preprocessing import temp_forecasting_shape_processing,test_train_split
from error_reporting import error_reporting_regression,error_histogram,error_time_series,keras_training_loss_curve
from helpers import save_model,load_tsv
sys.path.append('../data_cleaning')
from grand import process_grand,clean_grand,preprocess_grand

df=clean_grand()
interpolate=True
df_list = preprocess_grand(df,interpolate)



#peaks = find_peaks(df['temp'].values,prominence=3)[0]
def partition_plot_intvl(df_partition,prom=10):
    for df in df_partition:
        series = df['temp'].values
        plot_time_intvl_between_peaks(series,prom)

def plot_time_intvl_between_peaks(series,prom=10):
    peaks = scipy_peaks(series,prom)
    time_diff = np.diff(peaks)[1:]
    plot_df_peaks(series,peaks,intvl=[0,10000])
    plt.show()
    plt.plot(time_diff)
    plt.show()
    plt.hist(time_diff,bins=20)
    plt.show()
    sorted_diff = np.sort(time_diff)
    print("Average: {0}".format(np.average(time_diff)))
    print("Median: {0}".format(np.median(time_diff)))
    print("Std Dev: {0}".format(np.std(time_diff)))
    print("Min: {0}".format(np.min(time_diff)))
    print("Max: {0}".format(np.max(time_diff)))
    print("10%: {0}".format(sorted_diff[int(0.1*len(sorted_diff))]))
    print("90%: {0}".format(sorted_diff[int(0.9*len(sorted_diff))]))

def scipy_peaks(arr,prom=3):
    return find_peaks(arr,prominence=prom)[0]

def plot_derivatives(arr,intvl=[0,1000]):
    arr = arr[intvl[0]:intvl[1]]
    fig,ax = plt.subplots(3,figsize=(10,10))
    ax[0].plot(arr)
    ax[0].set_title("Original Values")
    ax[1].plot(np.diff(arr))
    ax[1].set_title("First Derivative Values")
    ax[2].plot(np.diff(np.diff(arr)))
    ax[2].set_title("Second Derivative Values")
    
def peak_finder_grads(arr):
    first_deriv = np.abs(np.diff(arr))
    second_deriv = np.abs(np.diff(first_deriv))
    #value of 4 chosen empirically based on plots
    #note that derivatives have shifted values i.e. length of vector
    #decreases each time we take derivative
    peak_first_deriv = np.where(first_deriv>4)[0] + 1
    peak_second_deriv = np.where(second_deriv>4)[0] + 2
    return peak_first_deriv,peak_second_deriv

def plot_df_peaks(arr,peak_lst,intvl=[0,2000]):
    peak_lst = peak_lst[(peak_lst>intvl[0])&(peak_lst<intvl[1])]
    peak_x = peak_lst
    peak_y = arr[peak_lst]
    plt.figure()
    plt.plot(arr[intvl[0]:intvl[1]])
    plt.scatter(peak_x,peak_y,c='r')

def find_closest_peak(x,idx_of_peaks):
    '''
    x is index of peak, idx_of_peaks is list of index's of reference peaks,
    return idx of idx_of_peaks that is closest to x
    '''
    diff = np.abs(np.subtract(x,idx_of_peaks))
    return diff.argmin()

def find_diff_closest_peak(peaklst1,peaklst2):
    #assume peaklst1 has same shape as peaklst2
    assert len(peaklst1) == len(peaklst2),"Peak Lists should have same number of peaks"
    idxs = [find_closest_peak(x,peaklst2) for x in peaklst1]
    diff = [x-y for x,y in zip(peaklst1,peaklst2[idxs])]
    return np.array(diff)