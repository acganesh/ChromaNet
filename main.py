import sys
import os
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping


from data_utils import load
from models import BidirectionalLSTMNet

# Hack to fix Keras incompatibility issue;
# should change this
import tensorflow
from tensorflow.python.ops import control_flow_ops
tensorflow.python.control_flow_ops = control_flow_ops

# Random seeding for reproducibility
np.random.seed(42)

def main():
    # Disable output buffering
    #sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) 

    X_train_tiny, y_train_tiny = load('tiny')

    net = BidirectionalLSTMNet()
    model = net.build_model()

    checkpointer = ModelCheckpoint(filepath='./experiments/lstm_test.hdf5', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    model.fit(X_train_tiny, y_train_tiny, batch_size=100, class_weight='auto', nb_epoch=10, shuffle=True, validation_split=0.2, callbacks=[checkpointer, earlystopper])

if __name__ == '__main__':
    main()
