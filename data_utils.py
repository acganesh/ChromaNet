import numpy as np
import h5py
import scipy.io
import random
import socket

def get_data_path():
    hostname = socket.gethostname()
    # Add entries to this dictionary as more machines are used 
    data_paths = {'acganesh-MS-7885': 
                  '/home/adithya/Stanford/cs273b/273b-project/data/deepsea_train/',
                  'group16': '/home/data/deepsea_train/'}
    return data_paths[hostname]

def load(dataset):
    datasets = {'main': ['X_train', 'y_train'],
                'valid': ['X_valid', 'y_valid'],
                'test': ['X_test', 'y_test'],
                'large': ['X_train_large', 'y_train_large'],
                'med': ['X_train_med', 'y_train_med'],
                'small': ['X_train_small', 'y_train_small'],
                'tiny': ['X_train_tiny', 'y_train_tiny']}

    X_str, y_str = datasets[dataset] 
    X = load_hdf5(X_str)
    y = load_hdf5(y_str)
    return X, y

def load_raw():
    DATA_PATH = get_data_path()
    # It is unclear why the DanQ code uses both h5py and scipy to load
    # the data files.  This could be because of differences in file
    # sizes.
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

def create_subsampled_data():
    DATA_PATH = get_data_path()
    DOWNSAMPLED_PATH = '%s/downsampled' % DATA_PATH

    trainmat = h5py.File('%s/train.mat' % DATA_PATH)
    validmat = scipy.io.loadmat('%s/valid.mat' % DATA_PATH)
    testmat = scipy.io.loadmat('%s/test.mat' % DATA_PATH)

    print 'Data loaded from disk'

    # X_train.shape: (4400000, 1000, 4)
    # y_train.shape = (4400000, 919)
    X_train = np.transpose(np.array(trainmat['trainxdata']), axes=(2, 0, 1))
    y_train = np.array(trainmat['traindata']).T

    # X_valid.shape: (8000, 1000, 4)
    # y_valid.shape: (8000, 919)
    X_valid = np.transpose(validmat['validxdata'],axes=(0,2,1))
    y_valid = validmat['validdata']

    # X_test.shape: (455024, 1000, 4)
    # y_test.shape: (455024, 919) 
    X_test = np.transpose(testmat['testxdata'],axes=(0,2,1))
    y_test = testmat['testdata']

    print 'Data converted to NumPy arrays'

    X_train_large = X_train[:1000000]
    y_train_large = y_train[:1000000]
    X_train_med = X_train[:100000]
    y_train_med = y_train[:100000]
    X_train_small = X_train[:10000]
    y_train_small = y_train[:10000]
    X_train_tiny = X_train[:1000]
    y_train_tiny = y_train[:1000] 
    print 'Subsampled data computed'

    create_hdf5(X_train, 'X_train')
    create_hdf5(y_train, 'y_train')
    print 'Saved train'

    create_hdf5(X_valid, 'X_valid')
    create_hdf5(y_valid, 'y_valid')
    print 'Saved validation'

    create_hdf5(X_test, 'X_test')
    create_hdf5(y_test, 'y_test')
    print 'Saved test'

    create_hdf5(X_train_large, 'X_train_large')
    create_hdf5(y_train_large, 'y_train_large')
    print 'Saved train large'

    create_hdf5(X_train_med, 'X_train_med')
    create_hdf5(y_train_med, 'y_train_med')
    print 'Saved train med'
    
    create_hdf5(X_train_small, 'X_train_small')
    create_hdf5(y_train_small, 'y_train_small')
    print 'Saved train small'

    create_hdf5(X_train_tiny, 'X_train_tiny')
    create_hdf5(y_train_tiny, 'y_train_tiny')
    print 'Saved train tiny'

def create_hdf5(data, name):
    DATA_PATH = get_data_path()
    DOWNSAMPLED_PATH = '%sdownsampled' % DATA_PATH

    h5f = h5py.File('%s/%s.h5' % (DOWNSAMPLED_PATH, name), 'w')
    h5f.create_dataset(name, data=data)
    h5f.close()

def load_hdf5(name):
    DATA_PATH = get_data_path()
    DOWNSAMPLED_PATH = '%sdownsampled' % DATA_PATH
    fname = '%s/%s.h5' % (DOWNSAMPLED_PATH, name)
    h5f = h5py.File('%s/%s.h5' % (DOWNSAMPLED_PATH, name), 'r')
    data = h5f[name][:]
    h5f.close()
    return data

def rnd_subsample(trainmat, validmat, testmat, new_train_size, new_valid_size, new_test_size):
    np.random.seed(42)

    X_train_full=np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
    y_train_full=np.array(trainmat['traindata']).T
    X_valid_full=np.transpose(np.array(validmat['validxdata']),axes=(0,2,1))
    y_valid_full=np.array(validmat['validdata']).T
    X_test_full=np.transpose(np.array(testmat['testxdata']),axes=(0,2,1))
    y_test_full=np.array(testmat['testdata']).T

    # Old implementation: use a downsampling factor 
    #new_train_size=(X_train_full.shape[0])*train_factor
    #new_valid_size=(X_valid_full.shape[0])*valid_factor
    #new_test_size=(X_test_full.shape[0])*test_factor

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
