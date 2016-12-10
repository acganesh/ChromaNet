import numpy as np
from data_utils import load
from models import BidirectionalLSTMNet
import tensorflow

import sys
import os

# Hack to fix Keras incompatibility issue;
# should change this
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

    model.fit(X_train_tiny, y_train_tiny, batch_size=1, nb_epoch=10, shuffle=True, validation_split=0.2)

if __name__ == '__main__':
    main()
