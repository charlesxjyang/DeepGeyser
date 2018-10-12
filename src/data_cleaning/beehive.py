#misc
import sys
#data processing
import numpy as np
import pandas as pd
#homemade
sys.path.append('../../utils')
from helpers import load_tsv


def clean_beehive():
    filename = 'Beehive_Logger'
    df = load_tsv(filename)
    print("Now cleaning Beehive Geyser Data")
    print("Initial Shape of df: {0}".format(df.shape))
    #drop rows with duplicate eruption_time_epoch records
    df = df.drop_duplicates(subset='eruption_time_epoch')
    print("Shape of df after dropping duplicates: {0}".format(df.shape))
    #drop rows with nan in eruption_time_epoch
    df = df.dropna(subset=['eruption_time_epoch'],axis=0)
    print("Shape of df after dropping nan: {0}".format(df.shape))
    if 'data' not in filename:
        filename = '../../data/' + filename
    df['eruption_time_epoch'] = pd.to_numeric(df['eruption_time_epoch'])
    #add datetime objects
    df['eruption_time'] = pd.to_datetime(df['eruption_time_epoch'],unit='s')
    assert (df['eruption_time_epoch']>=0).all(), "Shouldnt be negative numbers"
    df.to_pickle(filename+'.pkl')
    return df