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
from data_cleaning_helpers import time_diff


df = load_tsv('Grand_eruptions.tsv')
delta = time_diff(df['eruption_time_epoch'])
recent = delta[:30000]
plt.plot(recent)
plt.hist(recent)
recent.describe()