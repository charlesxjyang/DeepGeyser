#misc
import sys
import argparse
#data processing
import numpy as np
#keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
#homemade
sys.path.append('../../utils')
from preprocessing import temp_forecasting_shape_processing,test_train_split
from error_reporting import error_reporting_regression,error_histogram,error_time_series,keras_training_loss_curve
from helpers import save_keras_model,load_tsv
sys.path.append('../data_cleaning')
from grand import process_grand,clean_grand

#dynamic gpu memory allocation
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
sess = tf.Session(config=config)
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
set_session(sess)  # set this TensorFlow session as the default session for Keras
#start code

##variables
#FLAGS = flags.FLAGS
#lookbacklength = 1000
#lookforwardlength = 1
#test_split = 0.1
#batchsize = 512
#nb_epochs = 30
#start model
def lstm_model(lookbacklength,lookforwardlength,test_split,batchsize,nb_epochs,shuffle):
    df = clean_grand()
    X_train,X_test,y_train,y_test = process_grand(df,lookbacklength=lookbacklength,lookforwardlength=lookforwardlength,test_split=test_split)
    #X_train,X_test = np.expand_dims(X_train,axis=3),np.expand_dims(X_test,axis=3)
    input_shape = X_train.shape[1:]
    #input_shape = input_shape + (1,)
    model = Sequential()
    model.add(LSTM(15,input_shape=input_shape))
    model.add(LeakyReLU())
    model.add(Dense(64))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dense(32))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dense(1))
    
    model.compile(loss='mse',
                  optimizer="adam")
    
    history = model.fit(X_train, y_train,
              batch_size=batchsize,
              epochs=nb_epochs,
              verbose=1,
              shuffle=shuffle)
    
    y_pred = model.predict(X_test)
    
    save_keras_model(model,'lstm')
    
    error_reporting_regression(y_test,y_pred)
    
    error_histogram(y_test,y_pred)
    
    error_time_series(y_test,y_pred)
    
    keras_training_loss_curve(history)
    
    
    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("lookbacklength", help="How far back in time the model looks",
                    type=int)
    parser.add_argument("lookforwardlength",help='How far in future model tries to predict',
                        type=int)
    parser.add_argument("nb_epochs",help="Number of epochs to train model",type=int)
    parser.add_argument("batchsize",help='Size of Training batches',type=int)
    parser.add_argument("test_split",help="Proportion of data that goes into test set",type=float)
    parser.add_argument("-shuffle",help="whether or not keras shuffles batches",type=bool,required=False,default=True)
    args = parser.parse_args()
    lstm_model(lookbacklength=args.lookbacklength,
           lookforwardlength=args.lookforwardlength,
           test_split=args.test_split,
           batchsize=args.batchsize,
           nb_epochs=args.nb_epochs,
           shuffle=args.shuffle)
