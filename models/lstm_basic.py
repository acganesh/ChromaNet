import numpy as np
import h5py
import scipy.io
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM
from keras.layers.core import Dropout, Dense

import tensorflow
from tensorflow.python.ops import control_flow_ops 
tensorflow.python.control_flow_ops = control_flow_ops
np.random.seed(42)

DATA_PATH = '/home/adithya/Stanford/cs273b/273b-project/data/deepsea_train'

print 'loading data'
trainmat = h5py.File('%s/train.mat' % DATA_PATH)
validmat = scipy.io.loadmat('%s/valid.mat' % DATA_PATH)
testmat = scipy.io.loadmat('%s/test.mat' % DATA_PATH)

print 'data loaded'

#X_train.shape: (4400000, 1000, 4)
#y_train.shape: (4400000, 919)
# 4400000 examples; each example is 1000 x 4 (1000 bp, one-hot encoding for each base)
# Output is length 919 binary prediction task

X_train = np.transpose(np.array(trainmat['trainxdata']), axes=(2, 0, 1))
y_train = np.array(trainmat['traindata']).T

print 'data np-ified'

model = Sequential()
model.add(Bidirectional(LSTM(320, return_sequences=False), input_shape=(1000, 4)))
model.add(Dropout(0.5))
model.add(Dense(output_dim=919, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', class_mode='binary')


model.fit(X_train, y_train, batch_size=100, nb_epoch=60, shuffle=True, show_accuracy=True, validation_data=(np.transpose(validmat['validxdata'],axes=(0,2,1)), validmat['validdata']))

tresults = model.evaluate(np.transpose(testmat['testxdata'],axes=(0,2,1)), testmat['testdata'],show_accuracy=True)
