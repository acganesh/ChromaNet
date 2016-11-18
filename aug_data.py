import sys 
import csv
import h5py
import numpy as np
import scipy.io as sio
import re

bedCurrentData = '../deepsea_train/allTFs.pos.bed'
bedFinal       = '../all_sites_cleaned.bed'
i1 = 2200000
i2 = 2204000

dummyTup = ('chr1', '10000', '10200')

dataDic = {}
trainmat = h5py.File('../deepsea_train/train.mat')
y_train = np.array(trainmat['traindata']).T
validmat = sio.loadmat('../deepsea_train/valid.mat')
y_valid = validmat['validdata']
testmat = sio.loadmat('../deepsea_train/test.mat')
y_test = testmat['testdata']

def aGb(tup1, tup2):
  #hacky compare for two bed type tuples 
  m1 = re.match('chr([0-9+])',tup1[0])
  m2 = re.match('chr([0-9+])',tup2[0])
  if m1 is None or m2 is None :return False
  c1 = m1.group(1)
  c2 = m2.group(1)
  if int(c1) > int(c2):
    return True
  elif int(c2) > int(c1):
    return False
  if int(tup1[1]) > int(tup2[1]):
    return True
  return False

with open(bedCurrentData, 'r') as fin:
  reader = csv.reader(fin, delimiter='\t')
  i, k = 0, 0
  for chrm, start, end, dot, zero, star in reader:
    if i < i1:
      dataDic[(chrm, start, end)] = y_train[2*i,:]
      lastTrain = (chrm, start, end)
    elif i < i2:
      j = i - i1
      dataDic[(chrm, start, end)] = y_valid[2*j,:]
      lastValid = (chrm, start, end)
    elif chrm == 'chr8' or chrm == 'chr9':
      dataDic[(chrm, start, end)] = y_test[2*k,:]
      k += 1
    if i%100000 == 0:
      print i
    i += 1
del y_test, testmat, y_valid, validmat, y_train

# now read the other bed file and start writing
with open(bedFinal, 'r') as fin:
  reader = csv.reader(fin, delimiter='\t')
  i = 0
  train, valid, test, being_used = [],[],[], []
  for chrm, start, end, dot, zero, star in reader:
    pos = (chrm, start, end)
    if aGb(lastTrain, pos):
      if pos in dataDic:
        train.append(dataDic[pos])
      else :
        train.append(dataDic[dummyTup]*0)
      being_used.append(pos)
    elif aGb(lastValid, pos): 
      if pos in dataDic:
        valid.append(dataDic[pos])
      else: 
        valid.append(dataDic[dummyTup]*0)
      being_used.append(pos)
    elif chrm == 'chr8' or chrm == 'chr9':
      if (chrm, start, end) in dataDic:
        test.append(dataDic[pos])
      else:
        test.append(dataDic[dummyTup]*0)
      being_used.append(pos)
    i += 1
    if i% 100000 ==0: print i

#write the shit 

with open ('sites_used', 'w') as fout:
  writer = csv.writer(fout, delimiter = '\t')
  for row in being_used:
    writer.writerow(row)


sio.savemat('data.mat', {'train_y':train, 'valid_y':valid, 'test_y':test})
