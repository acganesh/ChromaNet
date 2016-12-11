# implement various metrics here
from keras import backend as K
from keras.models import load_model
from data_utils import load
import h5py

from sklearn.metrics import roc_auc_score, average_precision_score

"""
import tensorflow
from tensorflow.python.ops import control_flow_ops
tensorflow.python.control_flow_ops = control_flow_ops
"""

cnn_weights = '/home/adithya/Stanford/cs273b/273b-project/experiments/weights.10-0.08_0.005_NewWeighting_Nadam.hdf5'

def NLL_loss(y_true, y_pred):
    y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    y_t = K.reshape(y_true,(batch_sz * sampled,1))
    y_p = K.reshape(y_pred,(batch_sz * sampled,1))
    one_weights  = K.prod(K.concatenate([y_t,zeros], axis = 1), axis=1)
    zero_weights = K.prod(K.concatenate([1.0-y_t, ones], axis=1), axis=1)  # note the switch b/w zero and one label
    z_weights = K.reshape(zero_weights,(batch_sz,sampled))
    o_weights = K.reshape(one_weights ,(batch_sz,sampled))
    #assert K.dot(y_t, 1.0-y_t)==K.dot(y_t, 1.0-y_t) 
    #print K.eval(y_true), K.eval(y_pred)
    #sys.stdout.flush()
    #return K.sum(-(o_weights * K.log(y_pred) + z_weights * K.log(1.0 - y_pred)))
    #return -(K.dot(one_weights, K.log(y_p)) + K.dot(zero_weights,K.log(1-y_p)))
    #return -(K.dot(K.transpose(y_t), K.log(y_p))+K.dot(K.transpose(y_t), K.log(1-y_p)))
    #return K.mean(binary_crossentropy(y_t, y_p))
    #return -K.mean(one_weights * K.log(y_p) + zero_weights*K.log(1-y_p))
    return -K.mean(o_weights * K.log(y_pred) + z_weights*K.log(1-y_pred))

def get_metrics():
    model = load_model(cnn_weights)
    X_test, y_test = load('test')
    y_pred = model.predict_proba(X_test)

    # Output predictions to h5 file
    h5f = h5py.File('y_pred.h5', 'w')
    h5f.create_dataset('y_pred', data=y_pred)
    h5f.close()

    roc_auc = roc_auc_score(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_pred)

    import pdb; pdb.set_trace()

if __name__=='__main__':
    get_metrics()
