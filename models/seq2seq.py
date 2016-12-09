import numpy as np
import h5py
import scipy.io
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM
from keras.layers.core import Dropout, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

from data_utils import load

X_train, y_train = load()

model = Sequential()
model.add(TimeDistributed(Dense(64, input_shape=(1000, 4))))
model.add(LSTM(64, return_sequences=True))
model.add(TimeDistributed(Dense(4)))
model.add(Activation("softmax"))
model.compile(loss='binary_crossentropy', optimizer='Adagrad', class_mode='binary')


"""
model = Sequential()
model.add(Bidirectional(LSTM(320, return_sequences=False), input_shape=(1000, 4)))
model.add(Dropout(0.5))
model.add(Dense(output_dim=919, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', class_mode='binary')
"""

checkpointer = ModelCheckpoint(filepath="basic_lstm.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model.fit(X_train, y_train, batch_size=100, nb_epoch=60, shuffle=True, show_accuracy=True, validation_data=(np.transpose(validmat['validxdata'],axes=(0,2,1)), validmat['validdata']))

tresults = model.evaluate(np.transpose(testmat['testxdata'],axes=(0,2,1)), testmat['testdata'],show_accuracy=True)

print tresults
