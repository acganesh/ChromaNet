import numpy as np
import h5py
import scipy.io
import random 
"""
For reference the bottom gives how trainmat, validmat, and testmat are defined. 
DATA_PATH='/home/data/deepsea_train'
print 'loading data'
trainmat=h5py.FIle('%s/train.mat' % DATA_PATH)
validmat=scipy.io.loadmat('%s/valid.mat' % DATA_PATH)
testmat=scipy.io.loadmat('%s/test.mat' % DATA_PATH)
"""
def down_sample_data(trainmat, validmat, testmat,train_factor, valid_factor, test_factor):
    X_train_full=np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
    y_train_full=np.array(trainmat['traindata']).T
    X_valid_full=np.transpose(np.array(validmat['validxdata']),axes=(2,0,1))
    y_valid_full=np.array(validmat['validdata']).T
    X_test_full=np.transpose(np.array(testmat['testxdata']),axes=(2,0,1))
    y_test_full=np.array(testmat['testdata']).T

    new_train_size=(X_train_full.shape[0])*train_factor
    new_valid_size=(X_valid_full.shape[0])*valid_factor
    new_test_size=(X_test_full.shape[0])*test_factor

    X_train_new=np.zeros((new_train_size,1000,4))
    y_train_new=np.zeros((new_train_size,919))
    X_valid_new=np.zeros((new_valid_size,1000,4))
    y_valid_new=np.zeros((new_valid_size,919))
    X_test_new=np.zeros((new_test_size,1000,4))
    y_test_new=np.zeros((new_test_size,919))


    train_indices=random.sample(xrange(X_train_full.shape[0]),new_train_size)
    valid_indices=random.sample(xrange(X_valid_full.shape[0]),new_valid_size)
    test_indices=random.sample(xrange(X_test_full.shape[0]),new_test_size)



    for i in xrange(len(train_indices)):
        X_train_new[i,:,:]=X_train_full[train_indices[i],:,:]
        y_train_new[i,:]=y_train_full[train_indices[i],:]
    for i in xrange(len(valid_indices)):
        X_valid_new[i,:,:]=X_valid_full[valid_indices[i],:,:]
        y_valid_new[i,:]=y_valid_full[valid_indices[i],:]
    for i in xrange(len(test_indices)):
        X_test_new[i,:,:]=X_test_full[test_indices[i],:,:]
        y_test_new[i,:]=y_test_full[test_indices[i],:]

    return X_train_new, y_train_new, X_valid_new, y_valid_new, X_test_new, y_test_new



    
