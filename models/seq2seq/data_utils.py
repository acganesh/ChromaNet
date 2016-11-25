import numpy as np
import h5py
import scipy.io

# For reproducibility
np.random.seed(42)

DATA_PATH = '/home/data/deepsea_train'
print 'Loading data'


def load(data_path = DATA_PATH):
    trainmat = h5py.File('%s/train.mat' % DATA_PATH)
    validmat = scipy.io.loadmat('%s/valid.mat' % DATA_PATH)
    testmat = scipy.io.loadmat('%s/test.mat' % DATA_PATH)

    print 'Data loaded from disk'

    #X_train.shape: (4400000, 1000, 4)
    #y_train.shape: (4400000, 919)
    # 4400000 examples; each example is 1000 x 4 (1000 bp, one-hot encoding for each base)
    # Output is length 919 binary prediction task

    X_train = np.transpose(np.array(trainmat['trainxdata']), axes=(2, 0, 1))
    y_train = np.array(trainmat['traindata']).T

    print 'Data converted to NumPy arrays'

    return X_train, y_train

def bucket(X_train, y_train, buckets):
    """Need to fill this in"""
    pass
