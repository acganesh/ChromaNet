from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.core import Dropout, Dense
from CWRNN import ClockworkRNN

class BidirectionalClockworkRNN:
    def build_model(self):
        model=Sequential()
        #fiddle around with period_spec
        model.add(Bidirectional(ClockworkRNN(320, period_spec=[1, 2, 4, 8, 16,
                                                               32,64,128,256]),
                                input_shape=(1000,4)))
        #optional
        model.add(Dropout(0.5))
        model.add(Dense(output_dim=919,activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', class_mode='binary', metrics=['accuracy'])
        return model
