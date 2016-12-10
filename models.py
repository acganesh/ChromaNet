from keras.models import Sequential
from keras.layers import Bidirectional, LSTM
from keras.layers.core import Dropout, Dense

class BidirectionalLSTMNet:
    def build_model(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(320, return_sequences=False), input_shape=(1000, 4)))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim=919, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', class_mode='binary', metrics=['accuracy'])
        return model

class TimeDistributedLSTMNet:
    def build_model(self):
        model = Sequential()
        model.add(TimeDistributed(Dense(64, input_shape=(1000, 4))))
        model.add(LSTM(320, return_sequences=False))
        model.add(TimeDistributed(Dense(919)))
        model.add(Activation('softmax'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')
        return model
