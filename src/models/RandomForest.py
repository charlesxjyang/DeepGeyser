#misc
import sys
import argparse
#data processing
import numpy as np
#keras
from sklearn.ensemble import RandomForestRegressor
#homemade
sys.path.append('../../utils')
from error_reporting import error_reporting_regression,error_histogram,error_time_series
from helpers import save_sklearn_model
sys.path.append('../data_cleaning')
from grand import process_grand,clean_grand


def rf_model(lookbacklength,lookforwardlength,test_split,n_decision_trees):
    df = clean_grand()
    X_train,X_test,y_train,y_test = process_grand(df,lookbacklength=lookbacklength,lookforwardlength=lookforwardlength,test_split=test_split)
    X_train,X_test = np.squeeze(X_train),np.squeeze(X_test)
    rf = RandomForestRegressor(n_estimators=n_decision_trees)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    
    save_sklearn_model(rf,'RandomForest')
    
    error_reporting_regression(y_test,y_pred)
    
    error_histogram(y_test,y_pred)
    
    error_time_series(y_test,y_pred)
    
    
    
    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("lookbacklength", help="How far back in time the model looks",
                    type=int)
    parser.add_argument("lookforwardlength",help='How far in future model tries to predict',
                        type=int)
    parser.add_argument("n_decision_trees",help="Number of epochs to train model",type=int)
    parser.add_argument("test_split",help="Proportion of data that goes into test set",type=float)
    args = parser.parse_args()
    rf_model(lookbacklength=args.lookbacklength,
           lookforwardlength=args.lookforwardlength,
           test_split=args.test_split,
           n_decision_trees=args.n_decision_trees)

#def main(argv=None):
#    lstm_model(lookbacklength=FLAGS.lookbacklength,
#               lookforwardlength=FLAGS.lookforwardlength,
#               test_split=FLAGS.test_split,
#               batchsize=FLAGS.batchsize,
#               nb_epochs=FLAGS.nb_epochs)


#if __name__ == '__main__':
#    flags.DEFINE_integer('nb_epochs', nb_epochs,
#                         'Number of epochs to train model')
#    flags.DEFINE_integer('batch_size', batchsize, 'Size of training batches')
#    flags.DEFINE_integer('lookbacklength', lookbacklength, 'How far back in time model looks')
#    flags.DEFINE_integer('lookforwardlength', lookforwardlength, 'How far in future model tries to predict')
#    flags.DEFINE_float("test_split",test_split,"Proportion of data that goes into test set")
#    tf.app.run()    
#    