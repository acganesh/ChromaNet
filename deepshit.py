import numpy as np
import sys
import h5py
import scipy.io
np.random.seed(111) # for reproducibility                                             

from keras import backend as K 
from keras.preprocessing import sequence
from keras.optimizers import RMSprop, Nadam, SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.callbacks import Callback,LambdaCallback, ModelCheckpoint, EarlyStopping, RemoteMonitor
import tensorflow as tf
from keras.metrics import binary_accuracy, fbeta_score, binary_crossentropy
import matplotlib.pyplot as plt
import random 
import theano
import theano.tensor as T
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import hmean
_FLOATX = 'float32'
_EPSILON = 10e-8



random.seed('123')

sampled = 5
index = random.sample(xrange(919), sampled)
batch_sz = 16

print 'loading data'
#trainmat = h5py.File('../smallData.mat')                                   
trainmat = scipy.io.loadmat('../midData.m.mat')
validmat = scipy.io.loadmat('../valid.mat')                            
#testmat = scipy.io.loadmat('../test.mat')                              

#X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))                
X_train = np.array(trainmat['trainxdata'])
y_train = np.array(trainmat['traindata'])
y_train = y_train[:, index]       

ones = np.mean(y_train,0)
#ones[ones<.01] = 0.01
zeros = 1 - ones
os = np.array([ones for shit in range(batch_sz)])
zs = np.array([zeros for shit in range(batch_sz)])

os.shape = (batch_sz * sampled,1)
zs.shape = (batch_sz * sampled,1)
ones, zeros = os, zs 

y_train = y_train[:16, :]
X_train = X_train[:16,:,:]

#meanOnes = np.mean(y_train, 0)
#w = np.array([1.0/item for item in meanOnes])
#meanW = np.mean(w)
#regularizedW = np.array([np.mean([item, meanW]) for item in w])
#reg = np.array([np.mean([item, 1]) for item in regularizedW])
#
#weights = np.array([reg for i in range(batch_sz)]) 
#weights.shape = (batch_sz * sampled, 1)
##weights.shape = (sampled, 1)
#w_max = np.nanmax(weights[np.isfinite(weights)])
#weights[weights==np.inf] = w_max
#
def NLL_loss(y_true, y_pred):
    y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    y_t = K.reshape(y_true,(batch_sz * sampled,1))
    one_weights  = K.prod(K.concatenate([y_t,zeros], axis = 1), axis=1)
    zero_weights = K.prod(K.concatenate([1.0-y_t, ones], axis=1), axis=1)
    z_weights = K.reshape(zero_weights,(batch_sz, sampled))
    o_weights = K.reshape(one_weights ,(batch_sz, sampled))
    #assert K.dot(y_t, 1.0-y_t)==K.dot(y_t, 1.0-y_t) 
    return -(o_weights * K.log(y_pred) + z_weights * K.log(1.0 - y_pred))


lr, decay, p = 0.005, 8e-7, 0.5

filters = [320, 480, 960]
width = 1000
convWindows = [8, 8, 8]
poolWindows = [4,4,4]
dropouts    = [0.0,0.0,0.0,0.0]
num_layers = 3

thef = open('test.txt', 'a')

model = Sequential()

model.add(Convolution1D(filters[0], input_dim=4, input_length=1000, filter_length=convWindows[0],
  border_mode="valid", activation="relu",subsample_length=1))
model.add(MaxPooling1D(pool_length=poolWindows[0]))
model.add(Dropout(dropouts[0]))
nchannel = np.floor((width - convWindows[0])/poolWindows[0])


for i in xrange(num_layers - 2):
  i = i + 1
  model.add(Convolution1D(filters[i], convWindows[i], activation = "relu"))
  model.add(MaxPooling1D(pool_length=poolWindows[i]))
  model.add(Dropout(dropouts[i]))
  nchannel = np.floor((nchannel - convWindows[i])/poolWindows[i])

model.add(Convolution1D(filters[i+1], convWindows[i+1], activation = "relu"))
nchannel =- convWindows[i+1]
model.add(Dropout(dropouts[i+1]))

# add fully connected layer. We may want to keep this one shallow                     

model.add(Flatten())
model.add(Dense(925, input_dim=int(filters[i+1]*nchannel)))
model.add(Dropout(dropouts[3]))
model.add(Dense(sampled, input_dim=925, activation='sigmoid'))


model.compile(loss=NLL_loss, optimizer=Nadam(lr=lr),  metrics=[binary_accuracy])

class Histories(Callback):
  def __init__(self):
    self.pr=0.0
    self.roc=0.0
    self.pr_auc=[]
  def on_train_begin(self, logs={}):
    self.pr_auc = []
    self.pr = 0.0
    self.roc_auc = []
    self.roc = 0.0
    self.loss = []
    self.entropyloss = []
  def on_epoch_end(self,epoch, logs={}):
    y_pred = self.model.predict(self.model.validation_data[0])
    y_true = self.model.validation_data[1]
    y_pred = y_pred[:, np.sum(y_true, axis=0) > 0]
    y_true = y_true[:,np.sum(y_true, axis=0) > 0]
   # self.roc = roc_auc_score(y_true, y_pred)
   # self.roc_auc.append(self.roc)
   # self.pr = average_precision_score(y_true, y_pred)
   # self.pr_auc.append(self.pr)   
   # self.loss.append(NLL_loss(y_pred, y_true)) 
   # self.entropyloss.append(binary_crossentropy(y_true, y_pred))
    return

h1 = Histories()


checkpointer = ModelCheckpoint(filepath="weights/weightstest.{epoch:02d}-{val_loss:.2f}_"+str(lr)+"_NewWeighting_Nadam.hdf5", verbose=1, save_best_only=False,save_weights_only=False)
earlystopper = EarlyStopping(monitor='val_loss', patience=40, verbose=1, mode='min')


history = model.fit(X_train, y_train, batch_size=batch_sz, nb_epoch=120, shuffle=True, verbose = 1, validation_data=(np.transpose(validmat['validxdata'],axes=(0,2,1))[:16,:,:], validmat['validdata'][:16,index]), callbacks=[checkpointer,earlystopper,h1], show_accuracy=True)

print >> thef, "lr=",lr, decay, p, 'Nadam default'
print >> thef, history.history['val_loss'], history.history['val_binary_accuracy']
print >> thef, h1.pr_auc, h1.roc_auc



#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
#
#
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
