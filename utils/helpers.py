import datetime
from keras import models
import pandas as pd
from sklearn.externals import joblib

#loads .tsv file and pickles it in data folder, returns panda dataframes
def load_tsv(filename:str):
    #make sure in data folder
    if 'tsv' not in filename:
        filename = filename + '.tsv'
    if 'data' not in filename:
        if 'eruption' in filename:
            filename = '../../data/eruption_data/' + filename
        if 'Logger' in filename:
            filename = '../../data/logger_data/' + filename
    try:
        df = pd.read_table(filename,sep='\t')
    except:
        #catch parser error
        df = pd.read_table(filename,sep='\t',engine='python')
    #save pickle to data folder
    save_filename = filename[:-3] + 'pkl'
    df.to_pickle(save_filename)
    return df
  
def unix_to_datetime(unix_epoch_time):
    return datetime.datetime.fromtimestamp(unix_epoch_time)

def datetime_to_unix(datetime):
    return datetime((datetime - datetime(1970, 1, 1))).total_seconds()

def save_keras_model(model,filename):
    if 'h5' not in filename:
        filename = filename + '.h5'
    model.save('../../data/saved_models/' + filename)
    
def save_sklearn_model(model,filename):
    if 'joblib' not in filename:
        filename = filename + '.joblib'
    if 'data' not in filename:
        filename = '../../data/saved_models/' + filename

    joblib.dump(model,filename)

def load_model(filename:str):
    #assume only filename, no rel. path specified
    if 'data' not in filename:
        filename = '../../data/saved_models/' + filename
    return models.load_model(filename)

def save_np_array(filename:str,arr):
    if 'data' not in filename:
        filename = '../../data/saved_predictions/' + filename
    return np.save(filename,arr)